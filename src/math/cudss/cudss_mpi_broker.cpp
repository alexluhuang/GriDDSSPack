/*
 * Copyright (c) 2026 Battelle Memorial Institute
 * Licensed under the modified BSD License.
 */

#include "cudss/cudss_mpi_broker.hpp"

#include "gridpack/utilities/exception.hpp"

#include <algorithm>
#include <chrono>
#include <deque>
#include <exception>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>

namespace gridpack {
namespace math {
namespace {

const std::uint64_t kMagic = 0x4750435544535342ULL; // "GPCUDSSB"
const std::uint64_t kVersion = 1;
const int kRequestHeaderTag = 100;
const int kRowOffsetsTag = 101;
const int kColumnIndicesTag = 102;
const int kValuesTag = 103;
const int kRightHandSideTag = 104;
const int kResponseHeaderTag = 110;
const int kSolutionTag = 111;
const std::size_t kRequestFields = 10;
const std::size_t kResponseFields = 5;

enum MessageType
{
  kRegisterSolve = 1,
  kSolve = 2,
  kDone = 3
};

enum ResponseStatus
{
  kSolved = 1,
  kFallback = 2,
  kRetryWithStructure = 3,
  kError = 4
};

typedef std::chrono::steady_clock Clock;

void fail(const std::string& message)
{
  throw gridpack::Exception("cuDSS MPI broker: " + message);
}

void checkMPI(int status, const std::string& operation)
{
  if (status != MPI_SUCCESS) {
    char text[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(status, text, &length);
    fail(operation + " failed: " + std::string(text, length));
  }
}

int mpiCount(std::size_t count, const std::string& label)
{
  if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    fail(label + " exceeds MPI int count");
  }
  return static_cast<int>(count);
}

std::uint64_t hashBytes(std::uint64_t hash, std::uint32_t value)
{
  const std::uint64_t prime = 1099511628211ULL;
  for (std::size_t byte = 0; byte < sizeof(value); ++byte) {
    hash ^= static_cast<unsigned char>((value >> (8 * byte)) & 0xff);
    hash *= prime;
  }
  return hash;
}

std::uint64_t patternHash(const RealCsrSystem& system)
{
  std::uint64_t hash = 14695981039346656037ULL;
  hash = hashBytes(hash, system.rows);
  hash = hashBytes(hash, system.columns);
  hash = hashBytes(hash, system.nonzeros);
  for (std::size_t index = 0; index < system.rowOffsets.size(); ++index) {
    hash = hashBytes(hash, system.rowOffsets[index]);
  }
  for (std::size_t index = 0; index < system.columnIndices.size(); ++index) {
    hash = hashBytes(hash, system.columnIndices[index]);
  }
  return hash;
}

void validateSystem(const RealCsrSystem& system)
{
  if (system.rows == 0 || system.columns != system.rows ||
      system.nonzeros == 0 || system.rightHandSideCount != 1 ||
      system.rows >= static_cast<std::uint32_t>(
          std::numeric_limits<int>::max()) ||
      system.nonzeros > static_cast<std::uint32_t>(
          std::numeric_limits<int>::max()) ||
      system.rowOffsets.size() !=
        static_cast<std::size_t>(system.rows) + 1 ||
      system.columnIndices.size() != system.nonzeros ||
      system.values.size() != system.nonzeros ||
      system.rightHandSides.size() != system.rows) {
    fail("invalid real CSR system");
  }
  if (system.rowOffsets.front() != 0 ||
      system.rowOffsets.back() != system.nonzeros) {
    fail("invalid CSR row-offset bounds");
  }
  for (std::size_t row = 0; row < system.rows; ++row) {
    const std::uint32_t begin = system.rowOffsets[row];
    const std::uint32_t end = system.rowOffsets[row + 1];
    if (begin > end || end > system.nonzeros) {
      fail("CSR row offsets must be monotonic and bounded");
    }
    for (std::uint32_t entry = begin; entry < end; ++entry) {
      if (system.columnIndices[entry] >= system.columns ||
          (entry > begin && system.columnIndices[entry - 1] >=
                            system.columnIndices[entry])) {
        fail("CSR columns must be bounded and strictly increasing by row");
      }
    }
  }
}

bool samePattern(const RealCsrSystem& left, const RealCsrSystem& right)
{
  return left.rows == right.rows && left.columns == right.columns &&
    left.nonzeros == right.nonzeros &&
    left.rowOffsets == right.rowOffsets &&
    left.columnIndices == right.columnIndices;
}

void sendHeader(MPI_Comm communicator, int destination, int tag,
                const std::uint64_t *header, std::size_t count)
{
  checkMPI(MPI_Send(const_cast<std::uint64_t*>(header), mpiCount(count,
      "header"), MPI_UINT64_T, destination, tag, communicator),
      "MPI_Send(header)");
}

void receiveHeader(MPI_Comm communicator, int source, int tag,
                   std::uint64_t *header, std::size_t count,
                   MPI_Status *status)
{
  checkMPI(MPI_Recv(header, mpiCount(count, "header"), MPI_UINT64_T,
      source, tag, communicator, status), "MPI_Recv(header)");
}

} // anonymous namespace

class CUDSSBrokerClient::Impl
{
  struct PatternToken
  {
    RealCsrSystem pattern;
    std::uint64_t token;
    std::uint64_t lastUse;
  };

  public:
    Impl(MPI_Comm communicator, const CUDSSBrokerOptions& options)
      : p_communicator(communicator), p_options(options), p_rank(-1),
        p_size(0), p_nextRequest(1), p_clock(0), p_done(false), p_patterns()
    {
      checkMPI(MPI_Comm_rank(p_communicator, &p_rank), "MPI_Comm_rank");
      checkMPI(MPI_Comm_size(p_communicator, &p_size), "MPI_Comm_size");
      if (p_options.ownerRank < 0 || p_options.ownerRank >= p_size) {
        fail("owner rank is outside broker communicator");
      }
      if (p_rank == p_options.ownerRank) {
        fail("broker owner rank cannot construct a client");
      }
      if (p_options.maximumRegisteredPatterns == 0) {
        fail("client pattern-cache capacity must be positive");
      }
    }

    ~Impl(void) {}

    bool solve(const RealCsrSystem& system, std::uint64_t taskId,
               std::vector<double>& solution)
    {
      if (p_done) fail("solve requested after client completion");
      validateSystem(system);
      const int rowOffsetCount = mpiCount(system.rowOffsets.size(),
                                          "row offsets");
      const int columnCount = mpiCount(system.columnIndices.size(),
                                       "column indices");
      const int valueCount = mpiCount(system.values.size(), "matrix values");
      const int rhsCount = mpiCount(system.rightHandSides.size(),
                                    "right-hand side");
      PatternToken *pattern = find(system);
      for (int attempt = 0; attempt < 2; ++attempt) {
        const bool registration = pattern == NULL || pattern->token == 0;
        const std::uint64_t request = p_nextRequest++;
        std::uint64_t header[kRequestFields] = {};
        header[0] = kMagic;
        header[1] = kVersion;
        header[2] = registration ? kRegisterSolve : kSolve;
        header[3] = request;
        header[4] = registration ? 0 : pattern->token;
        header[5] = system.rows;
        header[6] = system.nonzeros;
        header[7] = patternHash(system);
        header[8] = taskId;
        sendHeader(p_communicator, p_options.ownerRank, kRequestHeaderTag,
                   header, kRequestFields);
        if (registration) {
          checkMPI(MPI_Send(const_cast<std::uint32_t*>(
              system.rowOffsets.data()), rowOffsetCount, MPI_UINT32_T,
              p_options.ownerRank,
              kRowOffsetsTag, p_communicator), "MPI_Send(row offsets)");
          checkMPI(MPI_Send(const_cast<std::uint32_t*>(
              system.columnIndices.data()), columnCount,
              MPI_UINT32_T, p_options.ownerRank, kColumnIndicesTag,
              p_communicator), "MPI_Send(column indices)");
        }
        checkMPI(MPI_Send(const_cast<double*>(system.values.data()),
            valueCount, MPI_DOUBLE,
            p_options.ownerRank, kValuesTag, p_communicator),
            "MPI_Send(matrix values)");
        checkMPI(MPI_Send(const_cast<double*>(
            system.rightHandSides.data()), rhsCount,
            MPI_DOUBLE, p_options.ownerRank, kRightHandSideTag,
            p_communicator), "MPI_Send(right-hand side)");

        std::uint64_t response[kResponseFields] = {};
        receiveHeader(p_communicator, p_options.ownerRank,
            kResponseHeaderTag, response, kResponseFields,
            MPI_STATUS_IGNORE);
        if (response[0] != kMagic || response[1] != kVersion ||
            response[2] != request) {
          fail("response header does not match request");
        }
        const ResponseStatus status =
          static_cast<ResponseStatus>(response[3]);
        const std::uint64_t token = response[4];
        if (registration && token != 0) {
          if (pattern == NULL) {
            evictIfNeeded();
            PatternToken created;
            created.pattern = system;
            created.pattern.values.clear();
            created.pattern.rightHandSides.clear();
            created.token = token;
            created.lastUse = ++p_clock;
            p_patterns.push_back(created);
            pattern = &p_patterns.back();
          } else {
            pattern->token = token;
            pattern->lastUse = ++p_clock;
          }
        }
        if (status == kSolved) {
          solution.resize(system.rows);
          checkMPI(MPI_Recv(solution.data(), mpiCount(solution.size(),
              "solution"), MPI_DOUBLE, p_options.ownerRank, kSolutionTag,
              p_communicator, MPI_STATUS_IGNORE), "MPI_Recv(solution)");
          return true;
        }
        if (status == kFallback) return false;
        if (status == kRetryWithStructure) {
          if (pattern != NULL) pattern->token = 0;
          continue;
        }
        fail("broker reported a device or protocol error");
      }
      fail("broker token retry limit exceeded");
      return false;
    }

    void done(void)
    {
      if (p_done) return;
      std::uint64_t header[kRequestFields] = {};
      header[0] = kMagic;
      header[1] = kVersion;
      header[2] = kDone;
      sendHeader(p_communicator, p_options.ownerRank, kRequestHeaderTag,
                 header, kRequestFields);
      p_done = true;
    }

  private:
    MPI_Comm p_communicator;
    CUDSSBrokerOptions p_options;
    int p_rank;
    int p_size;
    std::uint64_t p_nextRequest;
    std::uint64_t p_clock;
    bool p_done;
    std::vector<PatternToken> p_patterns;

    PatternToken *find(const RealCsrSystem& system)
    {
      for (std::vector<PatternToken>::iterator iter = p_patterns.begin();
           iter != p_patterns.end(); ++iter) {
        if (samePattern(iter->pattern, system)) {
          iter->lastUse = ++p_clock;
          return &*iter;
        }
      }
      return NULL;
    }

    void evictIfNeeded(void)
    {
      if (p_patterns.size() < p_options.maximumRegisteredPatterns) return;
      std::vector<PatternToken>::iterator victim = p_patterns.begin();
      for (std::vector<PatternToken>::iterator iter = p_patterns.begin();
           iter != p_patterns.end(); ++iter) {
        if (iter->lastUse < victim->lastUse) victim = iter;
      }
      p_patterns.erase(victim);
    }
};

class CUDSSBrokerServer::Impl
{
  struct Pending
  {
    int source;
    std::uint64_t request;
    std::uint64_t task;
    std::vector<double> values;
    std::vector<double> rightHandSide;
    Clock::time_point arrival;
  };

  struct Pattern
  {
    std::uint64_t token;
    std::uint64_t hash;
    std::uint64_t lastUse;
    bool gpuDisabled;
    RealCsrSystem structure;
    std::deque<Pending> queue;
  };

  public:
    Impl(MPI_Comm communicator, const CUDSSBrokerOptions& options)
      : p_communicator(communicator), p_options(options), p_rank(-1),
        p_size(0), p_workerCount(0), p_doneCount(0), p_nextToken(1),
        p_clock(0), p_fatal(false), p_statistics(), p_patterns(),
        p_workerDone(), p_solver(batchOptions(options))
    {
      checkMPI(MPI_Comm_rank(p_communicator, &p_rank), "MPI_Comm_rank");
      checkMPI(MPI_Comm_size(p_communicator, &p_size), "MPI_Comm_size");
      if (p_rank != p_options.ownerRank) {
        fail("only the configured owner rank can construct the broker server");
      }
      if (p_options.batchSize == 0 ||
          p_options.batchSize > static_cast<std::size_t>(
              std::numeric_limits<int>::max()) ||
          p_options.minimumGpuBatchSize == 0 ||
          p_options.minimumGpuBatchSize > p_options.batchSize ||
          p_options.maximumRegisteredPatterns == 0 ||
          p_options.maximumDevicePatterns == 0) {
        fail("invalid broker batch or cache configuration");
      }
      p_workerCount = p_size - 1;
      p_workerDone.assign(p_size, false);
      if (!p_solver.available() && p_options.strict) p_fatal = true;
    }

    void run(void)
    {
      try {
        while (p_doneCount < p_workerCount || hasQueuedRequests()) {
          bool received = false;
          for (;;) {
            int available = 0;
            MPI_Status status;
            checkMPI(MPI_Iprobe(MPI_ANY_SOURCE, kRequestHeaderTag,
                p_communicator, &available, &status), "MPI_Iprobe");
            if (!available) break;
            receive(status.MPI_SOURCE);
            received = true;
          }
          // Drain every header already waiting before applying the latency
          // limit. A Texas payload is large enough that flushing after each
          // serial receive can split an otherwise complete synchronized wave.
          flushReady(p_doneCount == p_workerCount);
          if (!received) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
          }
        }
      } catch (const std::exception& error) {
        // A framing or MPI failure can leave synchronous workers blocked in a
        // payload send. There is no safe protocol-level recovery at that
        // point, so terminate this isolated broker communicator collectively.
        std::cerr << "fatal cuDSS broker server error: "
                  << error.what() << std::endl;
        MPI_Abort(p_communicator, 1);
        throw;
      } catch (...) {
        std::cerr << "fatal unknown cuDSS broker server error" << std::endl;
        MPI_Abort(p_communicator, 1);
        throw;
      }
      p_statistics.completedWorkers = p_doneCount;
      p_statistics.batch = p_solver.statistics();
    }

    CUDSSBrokerStatistics statistics(void) const
    {
      CUDSSBrokerStatistics result = p_statistics;
      result.batch = p_solver.statistics();
      return result;
    }

  private:
    MPI_Comm p_communicator;
    CUDSSBrokerOptions p_options;
    int p_rank;
    int p_size;
    int p_workerCount;
    int p_doneCount;
    std::uint64_t p_nextToken;
    std::uint64_t p_clock;
    bool p_fatal;
    CUDSSBrokerStatistics p_statistics;
    std::vector<std::unique_ptr<Pattern> > p_patterns;
    std::vector<bool> p_workerDone;
    CUDSSBatchSolver p_solver;

    static CUDSSBatchOptions batchOptions(const CUDSSBrokerOptions& options)
    {
      CUDSSBatchOptions result;
      result.device = options.device;
      result.batchSize = options.batchSize;
      result.maximumCachedPatterns = options.maximumDevicePatterns;
      result.validateResiduals = options.validateResiduals;
      result.residualTolerance = options.residualTolerance;
      return result;
    }

    bool hasQueuedRequests(void) const
    {
      for (std::size_t index = 0; index < p_patterns.size(); ++index) {
        if (!p_patterns[index]->queue.empty()) return true;
      }
      return false;
    }

    Pattern *findToken(std::uint64_t token)
    {
      for (std::size_t index = 0; index < p_patterns.size(); ++index) {
        if (p_patterns[index]->token == token) return p_patterns[index].get();
      }
      return NULL;
    }

    Pattern *findExact(std::uint64_t hash, const RealCsrSystem& system)
    {
      for (std::size_t index = 0; index < p_patterns.size(); ++index) {
        Pattern *pattern = p_patterns[index].get();
        if (pattern->hash == hash && samePattern(pattern->structure, system)) {
          return pattern;
        }
      }
      return NULL;
    }

    bool evictRegistration(void)
    {
      if (p_patterns.size() < p_options.maximumRegisteredPatterns) return true;
      std::vector<std::unique_ptr<Pattern> >::iterator victim = p_patterns.end();
      for (std::vector<std::unique_ptr<Pattern> >::iterator iter =
             p_patterns.begin(); iter != p_patterns.end(); ++iter) {
        if (!(*iter)->queue.empty()) continue;
        if (victim == p_patterns.end() ||
            (*iter)->lastUse < (*victim)->lastUse) victim = iter;
      }
      if (victim == p_patterns.end()) return false;
      p_patterns.erase(victim);
      return true;
    }

    void receive(int source)
    {
      std::uint64_t header[kRequestFields] = {};
      receiveHeader(p_communicator, source, kRequestHeaderTag, header,
                    kRequestFields, MPI_STATUS_IGNORE);
      if (header[0] != kMagic || header[1] != kVersion) {
        fail("invalid request magic or protocol version");
      }
      const MessageType type = static_cast<MessageType>(header[2]);
      if (type == kDone) {
        if (source == p_options.ownerRank || p_workerDone[source]) {
          fail("duplicate or owner DONE message");
        }
        p_workerDone[source] = true;
        ++p_doneCount;
        return;
      }
      if (type != kRegisterSolve && type != kSolve) {
        fail("unknown request type");
      }
      if (p_workerDone[source]) {
        fail("request received after worker DONE message");
      }
      if (header[5] == 0 ||
          header[5] >= static_cast<std::uint64_t>(
              std::numeric_limits<int>::max()) ||
          header[6] == 0 ||
          header[6] > static_cast<std::uint64_t>(
              std::numeric_limits<int>::max())) {
        fail("request dimensions exceed CSR protocol limits");
      }
      RealCsrSystem system;
      system.rows = static_cast<std::uint32_t>(header[5]);
      system.columns = system.rows;
      system.nonzeros = static_cast<std::uint32_t>(header[6]);
      system.rightHandSideCount = 1;
      if (type == kRegisterSolve) {
        system.rowOffsets.resize(static_cast<std::size_t>(system.rows) + 1);
        system.columnIndices.resize(system.nonzeros);
        checkMPI(MPI_Recv(system.rowOffsets.data(), mpiCount(
            system.rowOffsets.size(), "row offsets"), MPI_UINT32_T, source,
            kRowOffsetsTag, p_communicator, MPI_STATUS_IGNORE),
            "MPI_Recv(row offsets)");
        checkMPI(MPI_Recv(system.columnIndices.data(), mpiCount(
            system.columnIndices.size(), "column indices"), MPI_UINT32_T,
            source, kColumnIndicesTag, p_communicator, MPI_STATUS_IGNORE),
            "MPI_Recv(column indices)");
      }
      system.values.resize(system.nonzeros);
      system.rightHandSides.resize(system.rows);
      checkMPI(MPI_Recv(system.values.data(), mpiCount(system.values.size(),
          "matrix values"), MPI_DOUBLE, source, kValuesTag, p_communicator,
          MPI_STATUS_IGNORE), "MPI_Recv(matrix values)");
      checkMPI(MPI_Recv(system.rightHandSides.data(), mpiCount(
          system.rightHandSides.size(), "right-hand side"), MPI_DOUBLE,
          source, kRightHandSideTag, p_communicator, MPI_STATUS_IGNORE),
          "MPI_Recv(right-hand side)");

      Pattern *pattern = NULL;
      if (type == kRegisterSolve) {
        validateSystem(system);
        const std::uint64_t computedHash = patternHash(system);
        if (computedHash != header[7]) {
          sendResponse(source, header[3], kError, 0, NULL);
          ++p_statistics.errorResponses;
          return;
        }
        pattern = findExact(computedHash, system);
        if (pattern == NULL) {
          if (!evictRegistration()) {
            const ResponseStatus response =
              p_options.strict ? kError : kFallback;
            sendResponse(source, header[3], response, 0, NULL);
            if (response == kFallback) ++p_statistics.fallbackResponses;
            else ++p_statistics.errorResponses;
            return;
          }
          std::unique_ptr<Pattern> created(new Pattern);
          created->token = p_nextToken++;
          created->hash = computedHash;
          created->lastUse = ++p_clock;
          created->gpuDisabled = false;
          created->structure.rows = system.rows;
          created->structure.columns = system.columns;
          created->structure.nonzeros = system.nonzeros;
          created->structure.rightHandSideCount = 1;
          created->structure.rowOffsets.swap(system.rowOffsets);
          created->structure.columnIndices.swap(system.columnIndices);
          pattern = created.get();
          p_patterns.push_back(std::move(created));
          ++p_statistics.registrations;
        }
      } else {
        pattern = findToken(header[4]);
        if (pattern == NULL) {
          sendResponse(source, header[3], kRetryWithStructure, 0, NULL);
          ++p_statistics.retryResponses;
          return;
        }
        if (pattern->hash != header[7] ||
            pattern->structure.rows != system.rows ||
            pattern->structure.nonzeros != system.nonzeros) {
          sendResponse(source, header[3], kRetryWithStructure, 0, NULL);
          ++p_statistics.retryResponses;
          return;
        }
      }
      pattern->lastUse = ++p_clock;
      Pending pending;
      pending.source = source;
      pending.request = header[3];
      pending.task = header[8];
      pending.values.swap(system.values);
      pending.rightHandSide.swap(system.rightHandSides);
      pending.arrival = Clock::now();
      pattern->queue.push_back(std::move(pending));
      ++p_statistics.solveRequests;
    }

    void sendResponse(int destination, std::uint64_t request,
                      ResponseStatus status, std::uint64_t token,
                      const std::vector<double> *solution)
    {
      std::uint64_t header[kResponseFields] = {};
      header[0] = kMagic;
      header[1] = kVersion;
      header[2] = request;
      header[3] = status;
      header[4] = token;
      sendHeader(p_communicator, destination, kResponseHeaderTag, header,
                 kResponseFields);
      if (status == kSolved && solution != NULL) {
        checkMPI(MPI_Send(const_cast<double*>(solution->data()),
            mpiCount(solution->size(), "solution"), MPI_DOUBLE, destination,
            kSolutionTag, p_communicator), "MPI_Send(solution)");
      }
    }

    void flushReady(bool force)
    {
      const Clock::time_point now = Clock::now();
      for (std::size_t index = 0; index < p_patterns.size(); ++index) {
        Pattern& pattern = *p_patterns[index];
        while (!pattern.queue.empty()) {
          const std::uint64_t age = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
              now - pattern.queue.front().arrival).count());
          const bool full = pattern.queue.size() >= p_options.batchSize;
          const bool gpuUnavailable =
            p_fatal || pattern.gpuDisabled || !p_solver.available();
          if (!force && !full && !gpuUnavailable &&
              age < p_options.batchWaitMicroseconds) break;
          const std::size_t count = std::min(pattern.queue.size(),
                                             p_options.batchSize);
          if (gpuUnavailable) {
            respondWithoutGpu(pattern, count,
                p_options.strict ? kError : kFallback);
          } else if (count < p_options.minimumGpuBatchSize) {
            respondWithoutGpu(pattern, count,
                p_options.strict ? kError : kFallback);
          } else {
            executeBatch(pattern, count);
          }
        }
      }
    }

    void respondWithoutGpu(Pattern& pattern, std::size_t count,
                           ResponseStatus status)
    {
      for (std::size_t index = 0; index < count; ++index) {
        Pending pending = std::move(pattern.queue.front());
        pattern.queue.pop_front();
        sendResponse(pending.source, pending.request, status, pattern.token,
                     NULL);
        if (status == kFallback) ++p_statistics.fallbackResponses;
        else ++p_statistics.errorResponses;
      }
    }

    void executeBatch(Pattern& pattern, std::size_t count)
    {
      std::vector<Pending*> pending;
      std::vector<CUDSSBatchSystemView> systems;
      pending.reserve(count);
      systems.reserve(count);
      for (std::size_t index = 0; index < count; ++index) {
        pending.push_back(&pattern.queue[index]);
        systems.push_back(CUDSSBatchSystemView(
            pattern.queue[index].values,
            pattern.queue[index].rightHandSide));
      }
      std::vector<std::vector<double> > solutions;
      try {
        p_solver.solve(pattern.structure, systems, solutions);
      } catch (const std::exception&) {
        pattern.gpuDisabled = true;
        if (p_options.strict) p_fatal = true;
        respondWithoutGpu(pattern, count,
            p_options.strict ? kError : kFallback);
        return;
      }
      if (count == p_options.batchSize) ++p_statistics.fullBatches;
      else ++p_statistics.partialBatches;
      for (std::size_t index = 0; index < count; ++index) {
        sendResponse(pending[index]->source, pending[index]->request, kSolved,
                     pattern.token, &solutions[index]);
      }
      pattern.queue.erase(pattern.queue.begin(),
                          pattern.queue.begin() + count);
    }
};

CUDSSBrokerClient::CUDSSBrokerClient(
    MPI_Comm communicator, const CUDSSBrokerOptions& options)
  : p_impl(new Impl(communicator, options))
{}

CUDSSBrokerClient::~CUDSSBrokerClient(void)
{}

bool CUDSSBrokerClient::solve(const RealCsrSystem& system,
                              std::uint64_t taskId,
                              std::vector<double>& solution)
{
  return p_impl->solve(system, taskId, solution);
}

void CUDSSBrokerClient::done(void)
{
  p_impl->done();
}

CUDSSBrokerServer::CUDSSBrokerServer(
    MPI_Comm communicator, const CUDSSBrokerOptions& options)
  : p_impl(new Impl(communicator, options))
{}

CUDSSBrokerServer::~CUDSSBrokerServer(void)
{}

void CUDSSBrokerServer::run(void)
{
  p_impl->run();
}

CUDSSBrokerStatistics CUDSSBrokerServer::statistics(void) const
{
  return p_impl->statistics();
}

} // namespace math
} // namespace gridpack
