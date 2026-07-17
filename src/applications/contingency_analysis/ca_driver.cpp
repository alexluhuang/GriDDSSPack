/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
// -------------------------------------------------------------
/**
 * @file   ca_driver.cpp
 * @author Bruce Palmer
 * @date   2017-12-08 13:12:46 d3g096
 *
 * @updated Yousu Chen
 * - N-1 auto-generation for branch and generator contingencies
 * - Automatic slack bus transfer and capacity check
 * - Q-limit support integration
 * @date  2026-01-31
 *
 * @brief Driver for contingency analysis calculation that make use of the
 *        powerflow module to implement individual power flow simulations for
 *        each contingency. The different contingencies are distributed across
 *        separate communicators using the task manager.
 *
 *
 */
// -------------------------------------------------------------

#include "gridpack/include/gridpack.hpp"
#include "gridpack/applications/modules/powerflow/pf_app_module.hpp"
#include "gridpack/math/cudss/cudss_mpi_broker.hpp"
#include "gridpack/math/petsc/petsc_csr_exporter.hpp"
#include "gridpack/utilities/results_exporter.hpp"
#include "ca_driver.hpp"
#include "ca_pattern_scheduler.hpp"

#include <ga.h>

#include <boost/scoped_ptr.hpp>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>

#define USE_SUCCESS
// Sets up multiple communicators so that individual contingency calculations
// can be run concurrently

namespace {

enum CAProfileStatus
{
  PROFILE_CONVERGED = 1,
  PROFILE_DIVERGENT = 2,
  PROFILE_ISLANDED = 3,
  PROFILE_REJECTED = 4,
  PROFILE_SLACK_OVERLOAD = 5
};

struct CAProfileRecord
{
  int status;
  int powerFlowSolveCalls;
  int linearSolveCalls;
  int completedNewtonUpdates;
  int reportedNewtonIterations;
  int controllerLoopPasses;
  int areaInterchangePasses;
  int mapperWorkspaceRebuilds;
  int mapperWorkspaceReuses;
  double finalTolerance;
};

class CAContingencyTaskDispatcher
{
  public:
    CAContingencyTaskDispatcher(
        gridpack::parallel::TaskManager& manager,
        const std::vector<
          gridpack::contingency_analysis::PatternScheduleEpoch>& epochs)
      : p_manager(manager), p_epochs(epochs), p_epoch(0), p_epochStarted(false)
    {}

    bool nextTask(gridpack::parallel::Communicator& communicator, int *task)
    {
      while (p_epoch < p_epochs.size()) {
        if (!p_epochStarted) {
          p_manager.set(static_cast<int>(p_epochs[p_epoch].taskIds.size()));
          p_epochStarted = true;
        }
        int epochTask;
        if (p_manager.nextTask(communicator, &epochTask)) {
          *task = p_epochs[p_epoch].taskIds[epochTask];
          return true;
        }
        ++p_epoch;
        p_epochStarted = false;
      }
      *task = -1;
      return false;
    }

  private:
    gridpack::parallel::TaskManager& p_manager;
    const std::vector<
      gridpack::contingency_analysis::PatternScheduleEpoch>& p_epochs;
    std::size_t p_epoch;
    bool p_epochStarted;
};

std::vector<CAProfileRecord> gatherProfileRecords(
    const gridpack::parallel::Communicator& communicator,
    const std::vector<int>& localIndices,
    const std::vector<CAProfileRecord>& localRecords,
    int taskCount)
{
  const std::size_t integerFields = 9;
  if (taskCount < 0 || localIndices.size() != localRecords.size() ||
      static_cast<std::size_t>(taskCount) >
        static_cast<std::size_t>(std::numeric_limits<int>::max()) /
          integerFields) {
    throw gridpack::Exception("invalid contingency profile dimensions");
  }

  const std::size_t tasks = static_cast<std::size_t>(taskCount);
  std::vector<int> presence(tasks, 0);
  std::vector<int> fields(tasks * integerFields, 0);
  std::vector<double> tolerances(tasks, 0.0);
  for (std::size_t local = 0; local < localIndices.size(); ++local) {
    const int task = localIndices[local];
    if (task < 0 || task >= taskCount) {
      throw gridpack::Exception("contingency profile task is out of range");
    }
    const std::size_t index = static_cast<std::size_t>(task);
    const CAProfileRecord& record = localRecords[local];
    ++presence[index];
    int *destination = &fields[index * integerFields];
    destination[0] = record.status;
    destination[1] = record.powerFlowSolveCalls;
    destination[2] = record.linearSolveCalls;
    destination[3] = record.completedNewtonUpdates;
    destination[4] = record.reportedNewtonIterations;
    destination[5] = record.controllerLoopPasses;
    destination[6] = record.areaInterchangePasses;
    destination[7] = record.mapperWorkspaceRebuilds;
    destination[8] = record.mapperWorkspaceReuses;
    tolerances[index] = record.finalTolerance;
  }

  MPI_Comm mpiCommunicator = static_cast<MPI_Comm>(communicator);
  const int taskElements = taskCount;
  const int fieldElements = static_cast<int>(fields.size());
  int presenceError = MPI_SUCCESS;
  int toleranceError = MPI_SUCCESS;
  int fieldError = MPI_SUCCESS;
  if (taskElements > 0) {
    presenceError = MPI_Allreduce(MPI_IN_PLACE, presence.data(), taskElements,
                                  MPI_INT, MPI_SUM, mpiCommunicator);
    toleranceError = MPI_Allreduce(MPI_IN_PLACE, tolerances.data(),
                                   taskElements, MPI_DOUBLE, MPI_SUM,
                                   mpiCommunicator);
  }
  if (fieldElements > 0) {
    fieldError = MPI_Allreduce(MPI_IN_PLACE, fields.data(), fieldElements,
                               MPI_INT, MPI_SUM, mpiCommunicator);
  }
  if (presenceError != MPI_SUCCESS || toleranceError != MPI_SUCCESS ||
      fieldError != MPI_SUCCESS) {
    throw gridpack::Exception("unable to gather contingency profile records");
  }

  std::vector<CAProfileRecord> result(tasks);
  for (std::size_t index = 0; index < tasks; ++index) {
    if (presence[index] != 1) {
      throw gridpack::Exception(
          "contingency profile task was not produced exactly once");
    }
    const int *source = &fields[index * integerFields];
    result[index].status = source[0];
    result[index].powerFlowSolveCalls = source[1];
    result[index].linearSolveCalls = source[2];
    result[index].completedNewtonUpdates = source[3];
    result[index].reportedNewtonIterations = source[4];
    result[index].controllerLoopPasses = source[5];
    result[index].areaInterchangePasses = source[6];
    result[index].mapperWorkspaceRebuilds = source[7];
    result[index].mapperWorkspaceReuses = source[8];
    result[index].finalTolerance = tolerances[index];
  }
  return result;
}

void gatherContingencyStatus(
    const gridpack::parallel::Communicator& communicator,
    const std::vector<int>& localIndices,
    std::vector<bool>& success,
    std::vector<int>& violation,
    std::vector<bool>& isolated,
    int taskCount)
{
  if (taskCount < 0 || localIndices.size() != success.size() ||
      localIndices.size() != violation.size() ||
      localIndices.size() != isolated.size()) {
    throw gridpack::Exception("invalid contingency status dimensions");
  }
  const std::size_t tasks = static_cast<std::size_t>(taskCount);
  std::vector<int> presence(tasks, 0);
  std::vector<int> successValues(tasks, 0);
  std::vector<int> violationValues(tasks, 0);
  std::vector<int> isolatedValues(tasks, 0);
  for (std::size_t local = 0; local < localIndices.size(); ++local) {
    const int task = localIndices[local];
    if (task < 0 || task >= taskCount) {
      throw gridpack::Exception("contingency status task is out of range");
    }
    const std::size_t index = static_cast<std::size_t>(task);
    ++presence[index];
    successValues[index] = success[local] ? 1 : 0;
    violationValues[index] = violation[local];
    isolatedValues[index] = isolated[local] ? 1 : 0;
  }

  MPI_Comm mpiCommunicator = static_cast<MPI_Comm>(communicator);
  int errors[4] = {MPI_SUCCESS, MPI_SUCCESS, MPI_SUCCESS, MPI_SUCCESS};
  if (taskCount > 0) {
    errors[0] = MPI_Allreduce(MPI_IN_PLACE, presence.data(), taskCount,
                              MPI_INT, MPI_SUM, mpiCommunicator);
    errors[1] = MPI_Allreduce(MPI_IN_PLACE, successValues.data(), taskCount,
                              MPI_INT, MPI_SUM, mpiCommunicator);
    errors[2] = MPI_Allreduce(MPI_IN_PLACE, violationValues.data(), taskCount,
                              MPI_INT, MPI_SUM, mpiCommunicator);
    errors[3] = MPI_Allreduce(MPI_IN_PLACE, isolatedValues.data(), taskCount,
                              MPI_INT, MPI_SUM, mpiCommunicator);
  }
  for (std::size_t index = 0; index < 4; ++index) {
    if (errors[index] != MPI_SUCCESS) {
      throw gridpack::Exception("unable to gather contingency status");
    }
  }

  success.assign(tasks, false);
  violation.assign(tasks, 0);
  isolated.assign(tasks, false);
  for (std::size_t index = 0; index < tasks; ++index) {
    if (presence[index] != 1 ||
        (successValues[index] != 0 && successValues[index] != 1) ||
        violationValues[index] < 0 || violationValues[index] > 4 ||
        (isolatedValues[index] != 0 && isolatedValues[index] != 1)) {
      throw gridpack::Exception(
          "contingency status task was not produced exactly once");
    }
    success[index] = successValues[index] != 0;
    violation[index] = violationValues[index];
    isolated[index] = isolatedValues[index] != 0;
  }
}

void accumulateSolveMetrics(
    CAProfileRecord& record,
    const gridpack::powerflow::PFSolveMetrics& metrics,
    const gridpack::utility::ConvergenceSummary& convergence)
{
  record.powerFlowSolveCalls++;
  record.linearSolveCalls += metrics.linearSolveCalls;
  record.completedNewtonUpdates += metrics.completedNewtonUpdates;
  record.reportedNewtonIterations += convergence.iterations;
  record.controllerLoopPasses += metrics.controllerLoopPasses;
  record.areaInterchangePasses += metrics.areaInterchangePasses;
  record.mapperWorkspaceRebuilds += metrics.mapperWorkspaceRebuilds;
  record.mapperWorkspaceReuses += metrics.mapperWorkspaceReuses;
  record.finalTolerance = convergence.finalTolerance;
}

const char *profileStatusName(int status)
{
  switch (status) {
    case PROFILE_CONVERGED:
      return "converged";
    case PROFILE_DIVERGENT:
      return "divergent";
    case PROFILE_ISLANDED:
      return "islanded";
    case PROFILE_REJECTED:
      return "rejected";
    case PROFILE_SLACK_OVERLOAD:
      return "slack_overload";
    default:
      return "unknown";
  }
}

std::string csvField(const std::string& value)
{
  std::string result("\"");
  for (std::string::const_iterator iter = value.begin();
       iter != value.end(); ++iter) {
    if (*iter == '"') result += '"';
    result += *iter;
  }
  result += '"';
  return result;
}

std::string lowerCase(std::string value)
{
  for (std::string::iterator iter = value.begin(); iter != value.end(); ++iter) {
    *iter = static_cast<char>(
        std::tolower(static_cast<unsigned char>(*iter)));
  }
  return value;
}

void brokerDiagnostic(bool enabled, int rank, const char *stage)
{
  if (enabled) {
    std::cerr << "p[" << rank << "] CUDSSBroker: " << stage << std::endl;
  }
}

void requireKluFallback(gridpack::utility::Configuration *configuration)
{
  gridpack::utility::Configuration::CursorPtr powerflow =
    configuration->getCursor("Configuration.Powerflow");
  gridpack::utility::Configuration::CursorPtr solver;
  if (powerflow) solver = powerflow->getCursor("LinearSolver");

  std::string backend("petsc");
  std::string petscOptions;
  if (solver) {
    backend = solver->get("Backend", backend);
    solver->get("PETScOptions", &petscOptions);
  }
  const char *environment =
    std::getenv("GRIDPACK_LINEAR_SOLVER_BACKEND");
  if (environment != NULL && *environment != '\0') backend = environment;
  if (lowerCase(backend) != "petsc") {
    throw gridpack::Exception(
        "CUDSSBroker requires Powerflow.LinearSolver Backend=petsc so "
        "fallback never launches scalar cuDSS on worker ranks");
  }
  if (lowerCase(petscOptions).find("klu") == std::string::npos) {
    throw gridpack::Exception(
        "CUDSSBroker requires KLU in Powerflow.LinearSolver.PETScOptions");
  }
}

class CABrokerLinearSystemExecutor
  : public gridpack::powerflow::PFLinearSystemExecutor
{
  public:
    CABrokerLinearSystemExecutor(
        gridpack::math::CUDSSBrokerClient& client,
        MPI_Comm communicator, bool strict)
      : p_client(client), p_communicator(communicator), p_strict(strict),
        p_taskId(std::numeric_limits<std::uint64_t>::max())
    {}

    void taskId(std::uint64_t value)
    {
      p_taskId = value;
    }

    bool solve(gridpack::powerflow::PFLinearSystem& system)
    {
      gridpack::math::RealCsrSystem csr;
      try {
        csr = gridpack::math::extractPetscRealCsrSystem(
            system.matrix, system.rightHandSide);
      } catch (const std::exception& error) {
        if (!p_strict) return false;
        abort(error.what());
      }

      try {
        std::vector<double> solution;
        if (!p_client.solve(csr, p_taskId, solution)) return false;
        gridpack::math::writePetscRealVector(system.solution, solution);
        return true;
      } catch (const std::exception& error) {
        abort(error.what());
      }
      return false;
    }

  private:
    gridpack::math::CUDSSBrokerClient& p_client;
    MPI_Comm p_communicator;
    bool p_strict;
    std::uint64_t p_taskId;

    void abort(const std::string& message)
    {
      int rank = -1;
      MPI_Comm_rank(p_communicator, &rank);
      std::cerr << "p[" << rank << "] fatal cuDSS broker error: "
                << message << std::endl;
      MPI_Abort(p_communicator, 2);
      throw gridpack::Exception("fatal cuDSS broker error: " + message);
    }
};

class CABrokerAbortGuard
{
  public:
    CABrokerAbortGuard(void)
      : p_communicator(MPI_COMM_NULL), p_active(false)
    {}

    ~CABrokerAbortGuard(void)
    {
      if (p_active && p_communicator != MPI_COMM_NULL) {
        std::cerr << "fatal error while the cuDSS broker was active; "
                  << "aborting its MPI communicator" << std::endl;
        MPI_Abort(p_communicator, 3);
      }
    }

    void arm(MPI_Comm communicator)
    {
      p_communicator = communicator;
      p_active = true;
    }

    void complete(void)
    {
      p_active = false;
    }

  private:
    MPI_Comm p_communicator;
    bool p_active;
    CABrokerAbortGuard(const CABrokerAbortGuard&);
    CABrokerAbortGuard& operator=(const CABrokerAbortGuard&);
};

class CAGADefaultPgroupGuard
{
  public:
    explicit CAGADefaultPgroupGuard(int group)
      : p_previous(GA_Pgroup_get_default()), p_active(true)
    {
      GA_Pgroup_set_default(group);
    }

    ~CAGADefaultPgroupGuard(void)
    {
      restore();
    }

    void restore(void)
    {
      if (p_active) {
        GA_Pgroup_set_default(p_previous);
        p_active = false;
      }
    }

  private:
    int p_previous;
    bool p_active;
    CAGADefaultPgroupGuard(const CAGADefaultPgroupGuard&);
    CAGADefaultPgroupGuard& operator=(const CAGADefaultPgroupGuard&);
};

} // anonymous namespace

/**
 * Basic constructor
 */
gridpack::contingency_analysis::CADriver::CADriver(void)
{
}

/**
 * Basic destructor
 */
gridpack::contingency_analysis::CADriver::~CADriver(void)
{
}

/**
 * Get list of contingencies from external file
 * @param cursor pointer to contingencies in input deck
 * @return vector of contingencies
 */
std::vector<gridpack::powerflow::Contingency>
  gridpack::contingency_analysis::CADriver::getContingencies(
      gridpack::utility::Configuration::ChildCursors contingencies)
{
  // The contingencies ChildCursors argument is a vector of configuration
  // pointers. Each element in the vector is pointing at a seperate Contingency
  // block within the Contingencies block in the input file.
  std::vector<gridpack::powerflow::Contingency> ret;
  int size = contingencies.size();
  int i, idx;
  // Create string utilities object to help parse file
  gridpack::utility::StringUtils utils;
  // Loop over all child cursors
  for (idx = 0; idx < size; idx++) {
    std::string ca_type;
    contingencies[idx]->get("contingencyType",&ca_type);
    // Contingency name is used to direct output to different files for each
    // contingency
    std::string ca_name;
    contingencies[idx]->get("contingencyName",&ca_name);
    if (ca_type == "Line") {
      std::string buses;
      contingencies[idx]->get("contingencyLineBuses",&buses);
      std::string names;
      contingencies[idx]->get("contingencyLineNames",&names);
      // Tokenize bus string to get a list of individual buses
      std::vector<std::string> string_vec = utils.blankTokenizer(buses);
      // Convert buses from character strings to ints
      std::vector<int> bus_ids;
      for (i=0; i<string_vec.size(); i++) {
        bus_ids.push_back(atoi(string_vec[i].c_str()));
      }
      string_vec.clear();
      // Tokenize names string to get a list of individual line tags
      string_vec = utils.blankTokenizer(names);
      std::vector<std::string> line_names;
      // clean up line tags so that they are exactly two characters
      for (i=0; i<string_vec.size(); i++) {
        line_names.push_back(utils.clean2Char(string_vec[i]));
      }
      // Check to make sure we found everything
      if (bus_ids.size() == 2*line_names.size()) {
        // Add contingency parameters to contingency struct
        gridpack::powerflow::Contingency contingency;
        contingency.p_name = ca_name;
        contingency.p_type = Branch;
        int i;
        for (i = 0; i < line_names.size(); i++) {
          contingency.p_from.push_back(bus_ids[2*i]);
          contingency.p_to.push_back(bus_ids[2*i+1]);
          contingency.p_ckt.push_back(line_names[i]);
          contingency.p_saveLineStatus.push_back(true);
        }
        // Add branch contingency to contingency list
        ret.push_back(contingency);
      }
    } else if (ca_type == "Generator") {
      std::string buses;
      contingencies[idx]->get("contingencyBuses",&buses);
      std::string gens;
      contingencies[idx]->get("contingencyGenerators",&gens);
      // Tokenize bus string to get a list of individual buses
      std::vector<std::string> string_vec = utils.blankTokenizer(buses);
      std::vector<int> bus_ids;
      // Convert buses from character strings to ints
      for (i=0; i<string_vec.size(); i++) {
        bus_ids.push_back(atoi(string_vec[i].c_str()));
      }
      string_vec.clear();
      // Tokenize gens string to get a list of individual generator tags
      string_vec = utils.blankTokenizer(gens);
      std::vector<std::string> gen_ids;
      // clean up generator tags so that they are exactly two characters
      for (i=0; i<string_vec.size(); i++) {
        gen_ids.push_back(utils.clean2Char(string_vec[i]));
      }
      // Check to make sure we found everything
      if (bus_ids.size() == gen_ids.size()) {
        gridpack::powerflow::Contingency contingency;
        contingency.p_name = ca_name;
        contingency.p_type = Generator;
        int i;
        for (i = 0; i < bus_ids.size(); i++) {
          contingency.p_busid.push_back(bus_ids[i]);
          contingency.p_genid.push_back(gen_ids[i]);
          contingency.p_saveGenStatus.push_back(true);
        }
        // Add generator contingency to contingency list
        ret.push_back(contingency);
      }
    }
  }
  return ret;
}

/**
 * Auto-generate N-1 contingencies from the network
 * @param pf_app power flow application module with loaded network
 * @param gen_branches generate branch contingencies
 * @param gen_generators generate generator contingencies
 * @return vector of auto-generated contingencies
 */
std::vector<gridpack::powerflow::Contingency>
  gridpack::contingency_analysis::CADriver::generateN1Contingencies(
      gridpack::powerflow::PFAppModule &pf_app,
      bool gen_branches, bool gen_generators)
{
  std::vector<gridpack::powerflow::Contingency> ret;
  gridpack::utility::StringUtils utils;

  // Generate N-1 branch contingencies
  if (gen_branches) {
    std::vector<std::string> branch_data = pf_app.writeBranchString("flow_str");
    int branch_count = 0;

    for (size_t i=0; i<branch_data.size(); i++) {
      std::vector<std::string> tokens = utils.blankTokenizer(branch_data[i]);
      if (tokens.size()%8 != 0) {
        continue;  // Skip malformed data
      }

      int nline = tokens.size()/8;
      for (int j=0; j<nline; j++) {
        int from_bus = atoi(tokens[j*8].c_str());
        int to_bus = atoi(tokens[j*8+1].c_str());
        std::string ckt_id = tokens[j*8+2];

        // Create contingency for this branch
        gridpack::powerflow::Contingency contingency;
        char name_buf[64];
        sprintf(name_buf, "BR_%d_%d_%s", from_bus, to_bus,
                utils.clean2Char(ckt_id).c_str());
        contingency.p_name = name_buf;
        contingency.p_type = Branch;
        contingency.p_from.push_back(from_bus);
        contingency.p_to.push_back(to_bus);
        contingency.p_ckt.push_back(utils.clean2Char(ckt_id));
        contingency.p_saveLineStatus.push_back(true);

        ret.push_back(contingency);
        branch_count++;
      }
    }

    printf("Auto-generated %d N-1 branch contingencies\n", branch_count);
  }

  // Generate N-1 generator contingencies
  if (gen_generators) {
    std::vector<std::string> gen_data = pf_app.writeBusString("power");
    int gen_count = 0;

    for (size_t i=0; i<gen_data.size(); i++) {
      std::vector<std::string> tokens = utils.blankTokenizer(gen_data[i]);
      if (tokens.size()%4 != 0) {
        continue;  // Skip malformed data
      }

      int ngen = tokens.size()/4;
      for (int j=0; j<ngen; j++) {
        int bus_id = atoi(tokens[j*4].c_str());
        std::string gen_id = tokens[j*4+1];

        // Create contingency for this generator
        gridpack::powerflow::Contingency contingency;
        char name_buf[64];
        sprintf(name_buf, "GN_%d_%s", bus_id,
                utils.clean2Char(gen_id).c_str());
        contingency.p_name = name_buf;
        contingency.p_type = Generator;
        contingency.p_busid.push_back(bus_id);
        contingency.p_genid.push_back(utils.clean2Char(gen_id));
        contingency.p_saveGenStatus.push_back(true);

        ret.push_back(contingency);
        gen_count++;
      }
    }

    printf("Auto-generated %d N-1 generator contingencies\n", gen_count);
  }

  return ret;
}

/**
 * Check if a contingency is a duplicate of any in the existing list
 * @param contingency the contingency to check
 * @param existing_list vector of existing contingencies
 * @return true if duplicate found, false otherwise
 */
bool gridpack::contingency_analysis::CADriver::isDuplicateContingency(
    const gridpack::powerflow::Contingency &contingency,
    const std::vector<gridpack::powerflow::Contingency> &existing_list)
{
  for (size_t i = 0; i < existing_list.size(); i++) {
    const gridpack::powerflow::Contingency &existing = existing_list[i];

    // Check if same type
    if (existing.p_type != contingency.p_type) {
      continue;
    }

    if (contingency.p_type == Branch) {
      // For branch contingencies, check if same branches are tripped
      // A duplicate means all branches match (order doesn't matter)
      if (existing.p_from.size() != contingency.p_from.size()) {
        continue;
      }

      bool all_match = true;
      for (size_t j = 0; j < contingency.p_from.size(); j++) {
        bool found = false;
        for (size_t k = 0; k < existing.p_from.size(); k++) {
          if (contingency.p_from[j] == existing.p_from[k] &&
              contingency.p_to[j] == existing.p_to[k] &&
              contingency.p_ckt[j] == existing.p_ckt[k]) {
            found = true;
            break;
          }
        }
        if (!found) {
          all_match = false;
          break;
        }
      }

      if (all_match) {
        return true;  // Duplicate found
      }

    } else if (contingency.p_type == Generator) {
      // For generator contingencies, check if same generators are tripped
      if (existing.p_busid.size() != contingency.p_busid.size()) {
        continue;
      }

      bool all_match = true;
      for (size_t j = 0; j < contingency.p_busid.size(); j++) {
        bool found = false;
        for (size_t k = 0; k < existing.p_busid.size(); k++) {
          if (contingency.p_busid[j] == existing.p_busid[k] &&
              contingency.p_genid[j] == existing.p_genid[k]) {
            found = true;
            break;
          }
        }
        if (!found) {
          all_match = false;
          break;
        }
      }

      if (all_match) {
        return true;  // Duplicate found
      }
    }
  }

  return false;  // Not a duplicate
}

/**
 * Execute application. argc and argv are standard runtime parameters
 */
void gridpack::contingency_analysis::CADriver::execute(int argc, char** argv)
{
  // Create world communicator for entire simulation
  gridpack::parallel::Communicator world;
  gridpack::parallel::Communicator launch_world(world);
  gridpack::parallel::Communicator task_comm;
  MPI_Comm broker_communicator = MPI_COMM_NULL;
  bool broker_enabled = false;
  bool broker_diagnostics = false;
  bool schedule_by_expected_pattern = false;
  int broker_worker_count = 0;
  gridpack::math::CUDSSBrokerOptions broker_options;
  boost::scoped_ptr<gridpack::math::CUDSSBrokerClient> broker_client;
  boost::scoped_ptr<CABrokerLinearSystemExecutor> broker_executor;
  boost::scoped_ptr<gridpack::parallel::Communicator> launch_task_comm;
  boost::scoped_ptr<gridpack::parallel::Communicator> role_world;
  boost::shared_ptr<gridpack::powerflow::PFNetwork> pf_network;
  boost::scoped_ptr<gridpack::parallel::TaskManager> taskmgr;
  boost::scoped_ptr<gridpack::analysis::StatBlock> vmag_stats;
  boost::scoped_ptr<gridpack::analysis::StatBlock> vang_stats;
  boost::scoped_ptr<gridpack::analysis::StatBlock> pgen_stats;
  boost::scoped_ptr<gridpack::analysis::StatBlock> qgen_stats;
  boost::scoped_ptr<gridpack::analysis::StatBlock> pflow_stats;
  boost::scoped_ptr<gridpack::analysis::StatBlock> qflow_stats;
  boost::scoped_ptr<gridpack::analysis::StatBlock> perf_stats;
  // PFAppModule contains a default GridPACK communicator. Construct it while
  // every launch rank is still participating; its communicator is replaced by
  // the network communicator in readNetwork().
  gridpack::powerflow::PFAppModule pf_app;
  // Keep this guard later in declaration order than every long-lived GA owner.
  // It therefore aborts the broker protocol before those objects unwind.
  CABrokerAbortGuard broker_abort_guard;
  boost::scoped_ptr<CAGADefaultPgroupGuard> ga_default_pgroup;

  // Get timer instance for timing entire calculation
  gridpack::utility::CoarseTimer *timer =
    gridpack::utility::CoarseTimer::instance();
  int t_total = timer->createCategory("Total Application");
  timer->start(t_total);

  // Read configuration file (user specified, otherwise assume that it is
  // call input.xml)
  gridpack::utility::Configuration *config
    = gridpack::utility::Configuration::configuration();
  if (argc >= 2 && argv[1] != NULL) {
    char inputfile[256];
    sprintf(inputfile,"%s",argv[1]);
    config->open(inputfile,world);
  } else {
    config->open("input.xml",world);
  }

  // Get size of group (communicator) that individual contingency calculations
  // will run on and create a task communicator. Each process is part of only
  // one task communicator, even though the world communicator is broken up into
  // many task communicators
  gridpack::utility::Configuration::CursorPtr cursor;
  cursor = config->getCursor("Configuration.Contingency_analysis");
  int grp_size;
  double Vmin, Vmax;
  // Check to find out if files should be printed for individual power flow
  // calculations
  bool print_calcs;
  std::string tmp_bool;
  gridpack::utility::StringUtils util;
  if (!cursor->get("printCalcFiles",&tmp_bool)) {
    print_calcs = true;
  } else {
    util.toLower(tmp_bool);
    if (tmp_bool == "false") {
      print_calcs = false;
    } else {
      print_calcs = true;
    }
  }
  bool write_stats = cursor->get("writeStats", true);
  bool profile = cursor->get("profile", false);
  const char *profile_env = std::getenv("GRIDPACK_CA_PROFILE");
  if (profile_env != NULL) {
    std::string profile_value(profile_env);
    util.toLower(profile_value);
    profile = profile_value != "0" && profile_value != "false" &&
      profile_value != "off";
  }
  std::string profile_file = "ca_profile.csv";
  cursor->get("profileFile", &profile_file);
  if (!cursor->get("groupSize",&grp_size)) {
    grp_size = 1;
  }
  if (!cursor->get("minVoltage",&Vmin)) {
    Vmin = 0.9;
  }
  if (!cursor->get("maxVoltage",&Vmax)) {
    Vmax = 1.1;
  }
  // Check for Q limit violations (qlim: true=enabled, false=disabled)
  bool check_Qlim = cursor->get("qlim", true);
  // Output format: "json", "csv", or "text" (default)
  std::string outputFormat = "text";
  cursor->get("outputFormat", &outputFormat);
  std::string outputFile = "ca_results";
  cursor->get("outputFile", &outputFile);

  gridpack::utility::Configuration::CursorPtr broker_cursor =
    cursor->getCursor("CUDSSBroker");
  if (broker_cursor) {
    broker_enabled = broker_cursor->get("Enabled", false);
    broker_diagnostics = broker_cursor->get("Diagnostics", false);
    schedule_by_expected_pattern = broker_cursor->get(
        "ScheduleByExpectedPattern", false);
  }
  if (broker_enabled) {
    int batch_size = 8;
    int minimum_batch_size = 0;
    int wait_microseconds = 0;
    int registered_patterns = 64;
    int device_patterns = 16;
    broker_cursor->get("BatchSize", &batch_size);
    minimum_batch_size = batch_size;
    broker_cursor->get("MinimumGpuBatchSize", &minimum_batch_size);
    broker_cursor->get("BatchWaitMicroseconds", &wait_microseconds);
    broker_cursor->get("MaximumRegisteredPatterns", &registered_patterns);
    broker_cursor->get("MaximumDevicePatterns", &device_patterns);
    broker_cursor->get("Device", &broker_options.device);
    broker_options.validateResiduals = broker_cursor->get(
        "ValidateResiduals", broker_options.validateResiduals);
    broker_options.residualTolerance = broker_cursor->get(
        "ResidualTolerance", broker_options.residualTolerance);
    broker_options.strict = broker_cursor->get(
        "Strict", broker_options.strict);

    if (grp_size != 1) {
      throw gridpack::Exception(
          "CUDSSBroker requires Contingency_analysis.groupSize=1");
    }
    if (launch_world.size() < 3) {
      throw gridpack::Exception(
          "CUDSSBroker requires at least two worker ranks and one owner rank");
    }
    const int worker_count = launch_world.size() - 1;
    broker_worker_count = worker_count;
    if (batch_size <= 0 || minimum_batch_size <= 0 ||
        minimum_batch_size > batch_size || batch_size > worker_count ||
        minimum_batch_size > worker_count || wait_microseconds < 0 ||
        registered_patterns <= 0 || device_patterns <= 0 ||
        broker_options.device < 0 ||
        !std::isfinite(broker_options.residualTolerance) ||
        broker_options.residualTolerance <= 0.0) {
      throw gridpack::Exception("invalid CUDSSBroker configuration");
    }
    broker_options.ownerRank = launch_world.size() - 1;
    broker_options.batchSize = static_cast<std::size_t>(batch_size);
    broker_options.minimumGpuBatchSize =
      static_cast<std::size_t>(minimum_batch_size);
    broker_options.batchWaitMicroseconds =
      static_cast<std::uint64_t>(wait_microseconds);
    broker_options.maximumRegisteredPatterns =
      static_cast<std::size_t>(registered_patterns);
    broker_options.maximumDevicePatterns =
      static_cast<std::size_t>(device_patterns);
    requireKluFallback(config);

    if (broker_diagnostics) {
      std::cerr << "p[" << launch_world.rank()
                << "] CUDSSBroker: configuration validated" << std::endl;
    }

    if (broker_diagnostics) {
      std::cerr << "p[" << launch_world.rank()
                << "] CUDSSBroker: duplicating protocol communicator"
                << std::endl;
    }
    if (MPI_Comm_dup(static_cast<MPI_Comm>(launch_world),
                     &broker_communicator) != MPI_SUCCESS) {
      throw gridpack::Exception(
          "unable to duplicate the cuDSS broker communicator");
    }
    // Once the owner can enter server.run(), an escaping exception on any
    // rank would otherwise strand peers in the protocol or final barrier.
    broker_abort_guard.arm(broker_communicator);
    const bool broker_owner =
      launch_world.rank() == broker_options.ownerRank;
    if (broker_diagnostics) {
      std::cerr << "p[" << launch_world.rank()
                << "] CUDSSBroker: splitting worker/owner roles" << std::endl;
    }
    // GridPACK communicator construction creates GA process groups. Build all
    // singleton task groups collectively from the launch world before ranks
    // enter disjoint worker/owner control paths.
    launch_task_comm.reset(new gridpack::parallel::Communicator(
        launch_world.divide(grp_size)));
    role_world.reset(new gridpack::parallel::Communicator(
        launch_world.split(broker_owner ? 1 : 0)));
    // GA_Create_handle() uses the process-local default process group before
    // callers attach a handle to its final group. Keep worker-only GridPACK
    // allocations off the launch group after the owner enters the broker.
    ga_default_pgroup.reset(
        new CAGADefaultPgroupGuard(role_world->getGroup()));
    if (broker_diagnostics) {
      std::cerr << "p[" << launch_world.rank()
                << "] CUDSSBroker: role and task communicators ready"
                << std::endl;
    }
    if (broker_owner) {
      if (broker_diagnostics) {
        std::cerr << "p[" << launch_world.rank()
                  << "] CUDSSBroker: entering owner loop" << std::endl;
      }
      gridpack::math::CUDSSBrokerServer server(
          broker_communicator, broker_options);
      server.run();
      const gridpack::math::CUDSSBrokerStatistics statistics =
        server.statistics();
      std::cout << "CUDSS_BROKER_SUMMARY"
        << " requests=" << statistics.solveRequests
        << " registrations=" << statistics.registrations
        << " full_batches=" << statistics.fullBatches
        << " partial_batches=" << statistics.partialBatches
        << " fallbacks=" << statistics.fallbackResponses
        << " errors=" << statistics.errorResponses
        << " completed=" << statistics.batch.completedSystems
        << std::endl;
      MPI_Barrier(broker_communicator);
      ga_default_pgroup.reset();
      broker_abort_guard.complete();
      MPI_Comm_free(&broker_communicator);
      timer->stop(t_total);
      return;
    }
    world = *role_world;
    task_comm = *launch_task_comm;
    broker_client.reset(new gridpack::math::CUDSSBrokerClient(
        broker_communicator, broker_options));
    broker_executor.reset(new CABrokerLinearSystemExecutor(
        *broker_client, broker_communicator, broker_options.strict));
    if (broker_diagnostics) {
      std::cerr << "p[" << launch_world.rank()
                << "] CUDSSBroker: worker client ready" << std::endl;
    }
  }

  // Set static flag for PFBus class BEFORE network creation.
  // This controls how Q values are reported in output functions:
  // - When check_Qlim = false: output uses calculated Q from p_Qinj
  // - When check_Qlim = true: output uses p_qg (set by chkQlim())
  gridpack::powerflow::PFBus::setQlim(check_Qlim);
  if (!broker_enabled) task_comm = world.divide(grp_size);

  // Keep track of failed calculations
#ifdef USE_SUCCESS
  std::vector<int> contingency_idx;
  std::vector<bool> contingency_success;
  std::vector<int> contingency_violation;
  std::vector<bool> contingency_isolated;
#endif
  // Create powerflow applications on each task communicator
  pf_network.reset(new gridpack::powerflow::PFNetwork(task_comm));
  // Read in the network from an external file and partition it over the
  // processors in the task communicator. This will read in power flow
  // parameters from the Powerflow block in the input
  pf_app.readNetwork(pf_network,config);
  // Finish initializing the network
  pf_app.initialize();
  if (broker_executor) {
    broker_executor->taskId(std::numeric_limits<std::uint64_t>::max());
    pf_app.setLinearSystemExecutor(broker_executor.get());
  }
  //  Set minimum and maximum voltage limits on all buses
  pf_app.setVoltageLimits(Vmin, Vmax);
  // Solve the base power flow calculation. This calculation is replicated on
  // all task communicators
  int t_base = timer->createCategory("Contingency: Base Case");
  timer->start(t_base);
  bool base_solve_ok = pf_app.solve();
  CAProfileRecord base_profile = {};
  base_profile.finalTolerance = std::numeric_limits<double>::quiet_NaN();
  if (profile) {
    accumulateSolveMetrics(base_profile, pf_app.getSolveMetrics(),
        pf_app.getConvergence());
  }
  // Check for Qlimit violations
  if (check_Qlim && !pf_app.checkQlimViolations()) {
    base_solve_ok = pf_app.solve();
    if (profile) {
      accumulateSolveMetrics(base_profile, pf_app.getSolveMetrics(),
          pf_app.getConvergence());
    }
  }
  timer->stop(t_base);
  // Some buses may violate the voltage limits in the base problem. Flag these
  // buses to ignore voltage violations on them.
  pf_app.ignoreVoltageViolations();

  // Collect base case results for export
  gridpack::utility::PowerFlowResults baseCaseResults;
  if (outputFormat != "text") {
    baseCaseResults = pf_app.collectResults();
  }

  // Check if auto-generation of N-1 contingencies is enabled
  // FullBranchN1: generate N-1 contingencies for all branches
  // FullGeneratorN1: generate N-1 contingencies for all generators
  cursor = config->getCursor("Configuration.Contingency_analysis");
  bool full_branch_n1 = false;
  bool full_generator_n1 = false;

  if (!cursor->get("FullBranchN1",&tmp_bool)) {
    full_branch_n1 = false;
  } else {
    util.toLower(tmp_bool);
    full_branch_n1 = (tmp_bool == "true");
  }

  if (!cursor->get("FullGeneratorN1",&tmp_bool)) {
    full_generator_n1 = false;
  } else {
    util.toLower(tmp_bool);
    full_generator_n1 = (tmp_bool == "true");
  }

  bool auto_generate_n1 = full_branch_n1 || full_generator_n1;

  std::vector<gridpack::powerflow::Contingency> events;
  int auto_generated_count = 0;
  int file_loaded_count = 0;
  int duplicates_skipped = 0;

  // Step 1: Auto-generate N-1 contingencies if requested
  if (auto_generate_n1) {
    if (world.rank() == 0) {
      printf("\n==================================================================\n");
      printf("Auto-generating N-1 contingencies from network\n");
      printf("  FullBranchN1: %s\n", full_branch_n1 ? "YES" : "NO");
      printf("  FullGeneratorN1: %s\n", full_generator_n1 ? "YES" : "NO");
      printf("==================================================================\n\n");
    }
    events = generateN1Contingencies(pf_app, full_branch_n1, full_generator_n1);
    auto_generated_count = events.size();
  }

  // Step 2: Load contingencies from file if specified
  // This allows combining auto-generated N-1 with custom N-2+ contingencies
  std::string contingencyfile;
  bool has_contingency_file = cursor->get("contingencyList",&contingencyfile);

  if (has_contingency_file || !auto_generate_n1) {
    // Set default filename if not specified
    if (!has_contingency_file) {
      contingencyfile = "contingencies.xml";
    }

    if (world.rank() == 0) {
      if (auto_generate_n1) {
        printf("Loading additional contingencies from file: %s\n", contingencyfile.c_str());
        printf("(Duplicates of auto-generated contingencies will be skipped)\n\n");
      } else {
        printf("Contingency List: %s\n", contingencyfile.c_str());
      }
    }

    // Open contingency file
    bool ok = config->open(contingencyfile,world);

    if (ok) {
      // Get a list of contingencies from file
      cursor = config->getCursor(
          "ContingencyList.Contingency_analysis.Contingencies");
      gridpack::utility::Configuration::ChildCursors contingencies;
      if (cursor) cursor->children(contingencies);
      std::vector<gridpack::powerflow::Contingency> file_contingencies =
          getContingencies(contingencies);

      // If auto-generation was used, check for duplicates before adding
      if (auto_generate_n1) {
        for (size_t i = 0; i < file_contingencies.size(); i++) {
          if (isDuplicateContingency(file_contingencies[i], events)) {
            duplicates_skipped++;
            if (world.rank() == 0) {
              printf("  Skipping duplicate: %s\n", file_contingencies[i].p_name.c_str());
            }
          } else {
            events.push_back(file_contingencies[i]);
            file_loaded_count++;
          }
        }
        if (world.rank() == 0 && file_loaded_count > 0) {
          printf("\nAdded %d unique contingencies from file\n", file_loaded_count);
          if (duplicates_skipped > 0) {
            printf("Skipped %d duplicates\n", duplicates_skipped);
          }
        }
      } else {
        // No auto-generation, just use file contingencies
        events = file_contingencies;
        file_loaded_count = events.size();
      }
    }
  }

  // Print summary
  if (world.rank() == 0) {
    printf("\n==================================================================\n");
    printf("Total contingencies to analyze: %d\n", (int)events.size());
    if (auto_generate_n1) {
      printf("  Auto-generated: %d\n", auto_generated_count);
      if (file_loaded_count > 0) {
        printf("  From file: %d\n", file_loaded_count);
      }
      if (duplicates_skipped > 0) {
        printf("  Duplicates skipped: %d\n", duplicates_skipped);
      }
    }
    printf("==================================================================\n\n");
  }

  // Print contingency details
  if (world.rank() == 0) {
    int idx;
    for (idx = 0; idx < events.size(); idx++) {
      printf("Name: %s\n",events[idx].p_name.c_str());
      if (events[idx].p_type == Branch) {
        int nlines = events[idx].p_from.size();
        int j;
        for (j=0; j<nlines; j++) {
          printf(" Line: (from) %d (to) %d (line) \'%s\'\n",
              events[idx].p_from[j],events[idx].p_to[j],
              events[idx].p_ckt[j].c_str());
        }
      } else if (events[idx].p_type == Generator) {
        int nbus = events[idx].p_busid.size();
        int j;
        for (j=0; j<nbus; j++) {
          printf(" Generator: (bus) %d (generator ID) \'%s\'\n",
              events[idx].p_busid[j],events[idx].p_genid[j].c_str());
        }
      }
    }
  }


  // Set up task manager on the world communicator. The number of tasks is
  // equal to the number of contingencies
  taskmgr.reset(new gridpack::parallel::TaskManager(world));
  int ntasks = events.size();
  std::vector<gridpack::contingency_analysis::PatternScheduleEpoch>
    task_epochs;
  if (schedule_by_expected_pattern && ntasks > 0) {
    std::vector<std::string> expected_pattern_classes;
    expected_pattern_classes.reserve(events.size());
    gridpack::contingency_analysis::ExpectedJacobianPatternClassifier
      pattern_classifier(pf_network);
    for (std::size_t event = 0; event < events.size(); ++event) {
      expected_pattern_classes.push_back(pattern_classifier.classify(
          events[event]));
    }
    const std::size_t batch_size = broker_options.batchSize;
    const std::size_t epoch_size =
      (static_cast<std::size_t>(broker_worker_count) / batch_size) *
        batch_size;
    task_epochs = gridpack::contingency_analysis::buildPatternSchedule(
        expected_pattern_classes, epoch_size);
    if (world.rank() == 0) {
      std::set<std::string> distinct_classes(
          expected_pattern_classes.begin(), expected_pattern_classes.end());
      std::cout << "CUDSS_PATTERN_SCHEDULE"
                << " tasks=" << ntasks
                << " classes=" << distinct_classes.size()
                << " epochs=" << task_epochs.size()
                << " epoch_size=" << epoch_size
                << std::endl;
    }
  } else if (ntasks > 0) {
    gridpack::contingency_analysis::PatternScheduleEpoch epoch;
    epoch.expectedPatternClass = "input-order";
    epoch.taskIds.reserve(events.size());
    for (int task = 0; task < ntasks; ++task) {
      epoch.taskIds.push_back(task);
    }
    task_epochs.push_back(epoch);
  }

  int nbus = pf_network->totalBuses();
  // Get bus voltage information for base case
  int i, j;
  std::vector<std::string> v_vals;
  int nsize = 0;
  std::vector<double> vmag, vang, pgen, qgen, pflow, qflow, perf;
  std::vector<int> mask, mag_mask;
  int t_store = timer->createCategory("Store Statistics");
  if (write_stats) {
    timer->start(t_store);
    v_vals = pf_app.writeBusString("vr_str");
    nsize = v_vals.size();
    std::vector<int> mag_ids;
    std::vector<int> ids;
    std::vector<std::string> mag_tags;
    std::vector<std::string> tags;
    // Find bus IDs and create a dummy tag label and get voltage magnitude
    // and angle for base case
    for (i=0; i<nsize; i++) {
      std::vector<std::string> tokens = util.blankTokenizer(v_vals[i]);
      int not_isolated = atoi(tokens[3].c_str());
      if (not_isolated == 1) {
        mag_ids.push_back(atoi(tokens[0].c_str()));
        mag_tags.push_back("1 ");
        vmag.push_back(atof(tokens[2].c_str()));
        if (atoi(tokens[4].c_str()) != 0) {
          mag_mask.push_back(2);
        } else {
          mag_mask.push_back(1);
        }
      }
      ids.push_back(atoi(tokens[0].c_str()));
      tags.push_back("1 ");
      vang.push_back(atof(tokens[1].c_str()));
      mask.push_back(1);
    }
    int nmags = vmag.size();
    world.max(&nmags,1);
    world.max(&nbus,1);
    // Create StatBlock objects for voltage magnitude and angles and add
    // bus IDs to it as well as base case values
    vmag_stats.reset(new gridpack::analysis::StatBlock(world,nmags,ntasks+1));
    vang_stats.reset(new gridpack::analysis::StatBlock(world,nbus,ntasks+1));
    if (world.rank() == 0) {
      vmag_stats->addRowLabels(mag_ids, mag_tags);
      vang_stats->addRowLabels(ids, tags);
      vmag_stats->addColumnValues(0,vmag,mag_mask);
      vang_stats->addColumnValues(0,vang,mask);
    }
    // Get generator power information
    v_vals.clear();
    ids.clear();
    tags.clear();
    mask.clear();
    v_vals = pf_app.writeBusString("power");
    nsize = v_vals.size();
    // Find bus IDs and tags for generators and evaluate Pg and Qg for base case
    for (i=0; i<nsize; i++) {
      std::vector<std::string> tokens = util.blankTokenizer(v_vals[i]);
      if (tokens.size()%4 != 0) {
        printf("Incorrect generator listing\n");
        continue;
      }
      int ngen = tokens.size()/4;
      for (j=0; j<ngen; j++) {
        ids.push_back(atoi(tokens[j*4].c_str()));
        tags.push_back(tokens[j*4+1]);
        pgen.push_back(atof(tokens[j*4+2].c_str()));
        qgen.push_back(atof(tokens[j*4+3].c_str()));
        mask.push_back(1);
      }
    }
    nsize = pgen.size();
    world.max(&nsize,1);
    // Create StatBlock objects for Pg and Qg and add labels and base case values
    pgen_stats.reset(new gridpack::analysis::StatBlock(world,nsize,ntasks+1));
    qgen_stats.reset(new gridpack::analysis::StatBlock(world,nsize,ntasks+1));
    if (world.rank() == 0) {
      pgen_stats->addRowLabels(ids, tags);
      qgen_stats->addRowLabels(ids, tags);
      pgen_stats->addColumnValues(0,pgen,mask);
      qgen_stats->addColumnValues(0,qgen,mask);
    }

    // Find flow parameters for all branch lines
    v_vals.clear();
    ids.clear();
    tags.clear();
    mask.clear();
    std::vector<int> id1;
    std::vector<int> id2;
    std::vector<double> pmin, pmax;
    v_vals = pf_app.writeBranchString("flow_str");
    nsize = v_vals.size();
    // Parse branch line endpoints as well as line IDs and values of P and Q for
    // base case
    for (i=0; i<nsize; i++) {
      std::vector<std::string> tokens = util.blankTokenizer(v_vals[i]);
      if (tokens.size()%8 != 0) {
        printf("Incorrect branch power flow listing\n");
        continue;
      }
      int nline = tokens.size()/8;
      for (j=0; j<nline; j++) {
        id1.push_back(atoi(tokens[j*8].c_str()));
        id2.push_back(atoi(tokens[j*8+1].c_str()));
        tags.push_back(tokens[j*8+2]);
        pflow.push_back(atof(tokens[j*8+3].c_str()));
        qflow.push_back(atof(tokens[j*8+4].c_str()));
        perf.push_back(atof(tokens[j*8+5].c_str()));
        pmin.push_back(-atof(tokens[j*8+6].c_str()));
        pmax.push_back(atof(tokens[j*8+6].c_str()));
        if (atoi(tokens[j*8+7].c_str()) == 0) {
          mask.push_back(1);
        } else {
          mask.push_back(2);
        }
      }
    }
    nsize = pflow.size();
    world.max(&nsize,1);
    // Create StatBlock objects for flow parameters and add labels and base case
    // values
    pflow_stats.reset(new gridpack::analysis::StatBlock(world,nsize,ntasks+1));
    qflow_stats.reset(new gridpack::analysis::StatBlock(world,nsize,ntasks+1));
    perf_stats.reset(new gridpack::analysis::StatBlock(world,nsize,ntasks+1));
    if (world.rank() == 0) {
      pflow_stats->addRowLabels(id1, id2, tags);
      qflow_stats->addRowLabels(id1, id2, tags);
      perf_stats->addRowLabels(id1, id2, tags);
      pflow_stats->addColumnValues(0,pflow,mask);
      qflow_stats->addColumnValues(0,qflow,mask);
      perf_stats->addColumnValues(0,perf,mask);
      pflow_stats->addRowMinValue(pmin);
      qflow_stats->addRowMinValue(pmin);
      pflow_stats->addRowMaxValue(pmax);
      qflow_stats->addRowMaxValue(pmax);
    }
    timer->stop(t_store);
  }
  if (check_Qlim) pf_app.clearQlimViolations();
  // Clear any Q limit warnings from base case before starting contingencies
  gridpack::powerflow::PFBus::clearQlimWarnings();

  // Local contingency results storage for JSON/CSV export
  std::vector<gridpack::utility::ContingencyResult> localContingencies;
  std::vector<int> local_profile_indices;
  std::vector<CAProfileRecord> local_profile_records;

  // Evaluate contingencies using the task manager
  int task_id;
  int scheduled_local_tasks = 0;
  char sbuf[128];
  int t_tasks = timer->createCategory("Contingency: Execute Tasks");
  timer->start(t_tasks);
  // The dispatcher synchronizes workers between pattern epochs and returns
  // original task IDs on every process in task_comm.
  CAContingencyTaskDispatcher task_dispatcher(*taskmgr, task_epochs);
  while (task_dispatcher.nextTask(task_comm, &task_id)) {
    if (schedule_by_expected_pattern && task_comm.rank() == 0) {
      ++scheduled_local_tasks;
    }
    if (broker_executor) {
      broker_executor->taskId(static_cast<std::uint64_t>(task_id));
    }
    CAProfileRecord profile_record = {};
    profile_record.finalTolerance =
      std::numeric_limits<double>::quiet_NaN();
    printf("Executing task %d on process %d\n",task_id,world.rank());
    // Trim trailing spaces from contingency name for filename
    std::string fname = events[task_id].p_name;
    size_t end = fname.find_last_not_of(' ');
    if (end != std::string::npos) fname = fname.substr(0, end + 1);
    sprintf(sbuf,"%s.out",fname.c_str());
    // Open a new file, based on the contingency name, to store results from
    // this particular contingency calculation
    if (print_calcs) pf_app.open(sbuf);
    // Write out information to the top of the output file providing some
    // information on the contingency
    sprintf(sbuf,"\nRunning task on %d processes\n",task_comm.size());
    if (print_calcs) pf_app.writeHeader(sbuf);
    if (events[task_id].p_type == Branch) {
      int nlines = events[task_id].p_from.size();
      int j;
      for (j=0; j<nlines; j++) {
        sprintf(sbuf," Line: (from) %d (to) %d (line) \'%s\'\n",
            events[task_id].p_from[j],events[task_id].p_to[j],
            events[task_id].p_ckt[j].c_str());
        printf("p[%d] Line: (from) %d (to) %d (line) \'%s\'\n",
            pf_network->communicator().rank(),
            events[task_id].p_from[j],events[task_id].p_to[j],
            events[task_id].p_ckt[j].c_str());
      }
    } else if (events[task_id].p_type == Generator) {
      int nbus = events[task_id].p_busid.size();
      int j;
      for (j=0; j<nbus; j++) {
        sprintf(sbuf," Generator: (bus) %d (generator ID) \'%s\'\n",
            events[task_id].p_busid[j],events[task_id].p_genid[j].c_str());
        printf("p[%d] Generator: (bus) %d (generator ID) \'%s\'\n",
            pf_network->communicator().rank(),
            events[task_id].p_busid[j],events[task_id].p_genid[j].c_str());
      }
    }
    if (print_calcs) pf_app.writeHeader(sbuf);
    // Reset all voltages back to their original values
    pf_app.resetVoltages();
    // Sync ghost bus data after voltage reset to ensure branches connected to
    // ghost buses use the correct reset voltages in power flow calculation
    pf_network->updateBuses();
    // Set contingency
    bool contingencyFound = pf_app.setContingency(events[task_id]);
    if (!contingencyFound) {
      printf("WARNING: Contingency '%s' - elements not found or no valid slack bus\n",
             events[task_id].p_name.c_str());
    }
    // Check for islanding before attempting to solve
    // Note: lone bus isolation is handled separately as a warning, not a failure
    int islandCount = pf_app.getIslandCount();
    bool hasLoneBus = pf_app.hasLoneBus();
    bool islandDetected = (islandCount > 1);
    // Solve power flow equations for this system
#ifdef USE_SUCCESS
    contingency_idx.push_back(task_id);
#endif
    // Skip power flow if contingency setup failed (no valid slack) or islanding detected
    bool slackCapacityOk = true;  // Will be checked after solve
    bool solveOk = false;
    if (contingencyFound && !islandDetected) {
      try {
        solveOk = pf_app.solve();
        if (profile) {
          accumulateSolveMetrics(profile_record, pf_app.getSolveMetrics(),
              pf_app.getConvergence());
        }
        if (solveOk && check_Qlim && !pf_app.checkQlimViolations()) {
          solveOk = pf_app.solve();
          if (profile) {
            accumulateSolveMetrics(profile_record, pf_app.getSolveMetrics(),
                pf_app.getConvergence());
          }
        }
      } catch (const std::exception& e) {
        printf("p[%d] hit exception: %s\n", world.rank(), e.what());
        printf("Solver failure\n");
        solveOk = false;
      } catch (...) {
        printf("p[%d] hit unknown exception during solve\n", world.rank());
        solveOk = false;
      }
    }
    if (profile) {
      if (!contingencyFound) {
        profile_record.status = PROFILE_REJECTED;
      } else if (islandDetected) {
        profile_record.status = PROFILE_ISLANDED;
      } else if (!solveOk) {
        profile_record.status = PROFILE_DIVERGENT;
      }
    }
    if (solveOk) {
      // Write PV->PQ conversion warnings to output file
      if (print_calcs && check_Qlim) {
        std::vector<std::string>& warnings = gridpack::powerflow::PFBus::getQlimWarnings();
        for (size_t w = 0; w < warnings.size(); w++) {
          pf_app.print(warnings[w].c_str());
        }
      }
      // Check if slack bus generator exceeds capacity
      slackCapacityOk = pf_app.checkSlackCapacity();
      if (!slackCapacityOk) {
        if (profile) profile_record.status = PROFILE_SLACK_OVERLOAD;
        // Slack generator exceeds Pmax - insufficient generation capacity
        // This is treated as a failure, similar to divergence
#ifdef USE_SUCCESS
        contingency_success.push_back(false);
        contingency_violation.push_back(0);
        contingency_isolated.push_back(false);
#endif
        if (outputFormat != "text") {
          gridpack::utility::ContingencyResult ctResult = {};
          ctResult.name = events[task_id].p_name;
          ctResult.type = (events[task_id].p_type == Branch) ? "branch" : "generator";
          ctResult.hasVoltageViolation = false;
          ctResult.hasBranchViolation = false;
          ctResult.solution.convergence.converged = false;
          localContingencies.push_back(ctResult);
        }
        sprintf(sbuf,"\nInsufficient generation capacity for contingency %s\n",
            events[task_id].p_name.c_str());
        if (print_calcs) pf_app.print(sbuf);
      } else {
        if (profile) profile_record.status = PROFILE_CONVERGED;
        // Power flow solved and slack within capacity
#ifdef USE_SUCCESS
        contingency_success.push_back(true);
        contingency_isolated.push_back(hasLoneBus);
#endif
        // If power flow solution is successful, write out voltages and currents
        if (print_calcs) pf_app.write();
        // Check for violations
        bool ok1 = pf_app.checkVoltageViolations();
        bool ok2 = pf_app.checkLineOverloadViolations();
        bool ok = ok1 && ok2;
        // Collect results for JSON/CSV export
        if (outputFormat != "text") {
          gridpack::utility::ContingencyResult ctResult = {};
          ctResult.name = events[task_id].p_name;
          ctResult.type = (events[task_id].p_type == Branch) ? "branch" : "generator";
          ctResult.hasVoltageViolation = !ok1;
          ctResult.hasBranchViolation = !ok2;
          ctResult.solution = pf_app.collectResults();
          localContingencies.push_back(ctResult);
        }
      // Include results of violation checks in output
      if (ok) {
        sprintf(sbuf,"\nNo violation for contingency %s\n",
            events[task_id].p_name.c_str());
#ifdef USE_SUCCESS
        contingency_violation.push_back(1);
#endif
      }
      // Report bus voltage violations
      if (!ok1) {
        sprintf(sbuf,"\nBus Violation for contingency %s\n",
            events[task_id].p_name.c_str());
      } else if (!ok) {
        sprintf(sbuf,"\nNo Bus Violation for contingency %s\n",
            events[task_id].p_name.c_str());
      }
      if (print_calcs) pf_app.print(sbuf);
      if (print_calcs) pf_app.writeCABus();
      // Report branch overload violations
      if (!ok2) {
        sprintf(sbuf,"\nBranch Violation for contingency %s\n",
            events[task_id].p_name.c_str());
      } else if (!ok) {
        sprintf(sbuf,"\nNo Branch Violation for contingency %s\n",
            events[task_id].p_name.c_str());
      }

#ifdef USE_SUCCESS
      if (!ok1 && !ok2) {
        contingency_violation.push_back(4);
      } else if (!ok1) {
        contingency_violation.push_back(2);
      } else if (!ok2) {
        contingency_violation.push_back(3);
      }
#endif

      if (print_calcs) pf_app.print(sbuf);
      if (print_calcs) pf_app.writeCABranch();
      // Get strings of data from power flow calculation and parse them to
      // extract numerical values. Store these values in vectors and then
      // add them to StatBlock objects
      if (write_stats) {
        timer->start(t_store);
        vmag.clear();
        vang.clear();
        mask.clear();
        mag_mask.clear();
        v_vals.clear();
        v_vals = pf_app.writeBusString("vr_str");
        nsize = v_vals.size();
        for (i=0; i<nsize; i++) {
          std::vector<std::string> tokens = util.blankTokenizer(v_vals[i]);
          int not_isolated = atoi(tokens[3].c_str());
          if (not_isolated == 1) {
            vmag.push_back(atof(tokens[2].c_str()));
            if (atoi(tokens[4].c_str()) != 0) {
              mag_mask.push_back(2);
            } else {
              mag_mask.push_back(1);
            }
          }
          vang.push_back(atof(tokens[1].c_str()));
          mask.push_back(1);
        }
        if (task_comm.rank() == 0) {
          vmag_stats->addColumnValues(task_id+1,vmag,mag_mask);
          vang_stats->addColumnValues(task_id+1,vang,mask);
        }
        pgen.clear();
        qgen.clear();
        mask.clear();
        v_vals.clear();
        v_vals = pf_app.writeBusString("power");
        nsize = v_vals.size();
        for (i=0; i<nsize; i++) {
          std::vector<std::string> tokens = util.blankTokenizer(v_vals[i]);
          if (tokens.size()%4 != 0) {
            printf("Incorrect generator listing\n");
            continue;
          }
          int ngen = tokens.size()/4;
          for (j=0; j<ngen; j++) {
            pgen.push_back(atof(tokens[j*4+2].c_str()));
            qgen.push_back(atof(tokens[j*4+3].c_str()));
            mask.push_back(1);
          }
        }
        if (task_comm.rank() == 0) {
          pgen_stats->addColumnValues(task_id+1,pgen,mask);
          qgen_stats->addColumnValues(task_id+1,qgen,mask);
        }
        pflow.clear();
        qflow.clear();
        perf.clear();
        mask.clear();
        v_vals.clear();
        v_vals = pf_app.writeBranchString("flow_str");
        nsize = v_vals.size();
        for (i=0; i<nsize; i++) {
          std::vector<std::string> tokens = util.blankTokenizer(v_vals[i]);
          if (tokens.size()%8 != 0) {
            printf("Incorrect branch power flow listing\n");
            continue;
          }
          int nline = tokens.size()/8;
          for (j=0; j<nline; j++) {
            pflow.push_back(atof(tokens[j*8+3].c_str()));
            qflow.push_back(atof(tokens[j*8+4].c_str()));
            perf.push_back(atof(tokens[j*8+5].c_str()));
            if (atoi(tokens[j*8+7].c_str()) == 0) {
              mask.push_back(1);
            } else {
              mask.push_back(2);
            }
          }
        }
        if (task_comm.rank() == 0) {
          pflow_stats->addColumnValues(task_id+1,pflow,mask);
          qflow_stats->addColumnValues(task_id+1,qflow,mask);
          perf_stats->addColumnValues(task_id+1,perf,mask);
        }
        timer->stop(t_store);
      }
        // Note: clearQlimViolations() moved after unSetContingency() below
      }  // end slackCapacityOk block
    } else {
#ifdef USE_SUCCESS
      contingency_success.push_back(false);
      contingency_violation.push_back(0);
      contingency_isolated.push_back(false);
#endif
      if (outputFormat != "text") {
        gridpack::utility::ContingencyResult ctResult = {};
        ctResult.name = events[task_id].p_name;
        ctResult.type = (events[task_id].p_type == Branch) ? "branch" : "generator";
        ctResult.hasVoltageViolation = false;
        ctResult.hasBranchViolation = false;
        ctResult.solution.convergence.converged = false;
        localContingencies.push_back(ctResult);
      }
      if (islandDetected) {
        sprintf(sbuf,"\nIslanding detected for contingency %s (%d islands)\n",
            events[task_id].p_name.c_str(), islandCount);
      } else if (!contingencyFound) {
        sprintf(sbuf,"\nNo valid slack bus for contingency %s\n",
            events[task_id].p_name.c_str());
      } else {
        sprintf(sbuf,"\nDivergent for contingency %s\n",
            events[task_id].p_name.c_str());
      }
      if (print_calcs) pf_app.print(sbuf);
      // Add dummy values to StatBlock object. Mask value is set to 0 for all
      // network elements to indicate calculation failure
      if (write_stats) {
        timer->start(t_store);
        vmag.clear();
        vang.clear();
        mask.clear();
        mag_mask.clear();
        v_vals.clear();
        v_vals = pf_app.writeBusString("vfail_str");
        nsize = v_vals.size();
        for (i=0; i<nsize; i++) {
          std::vector<std::string> tokens = util.blankTokenizer(v_vals[i]);
          int not_isolated = atoi(tokens[3].c_str());
          if (not_isolated == 1) {
            vmag.push_back(0.0);
            mag_mask.push_back(0);
          }
          vang.push_back(0.0);
          mask.push_back(0);
        }
        if (task_comm.rank() == 0) {
          vmag_stats->addColumnValues(task_id+1,vmag,mag_mask);
          vang_stats->addColumnValues(task_id+1,vang,mask);
        }
        pgen.clear();
        qgen.clear();
        mask.clear();
        v_vals.clear();
        v_vals = pf_app.writeBusString("pfail_str");
        nsize = v_vals.size();
        for (i=0; i<nsize; i++) {
          std::vector<std::string> tokens = util.blankTokenizer(v_vals[i]);
          if (tokens.size()%4 != 0) {
            printf("Incorrect generator listing\n");
            continue;
          }
          int ngen = tokens.size()/4;
          for (j=0; j<ngen; j++) {
            pgen.push_back(0.0);
            qgen.push_back(0.0);
            mask.push_back(0);
          }
        }
        if (task_comm.rank() == 0) {
          pgen_stats->addColumnValues(task_id+1,pgen,mask);
          qgen_stats->addColumnValues(task_id+1,qgen,mask);
        }
        pflow.clear();
        qflow.clear();
        perf.clear();
        mask.clear();
        v_vals.clear();
        v_vals = pf_app.writeBranchString("fail_str");
        nsize = v_vals.size();
        for (i=0; i<nsize; i++) {
          std::vector<std::string> tokens = util.blankTokenizer(v_vals[i]);
          if (tokens.size()%8 != 0) {
            printf("Incorrect branch power flow listing\n");
            continue;
          }
          int nline = tokens.size()/8;
          for (j=0; j<nline; j++) {
            pflow.push_back(0.0);
            qflow.push_back(0.0);
            perf.push_back(0.0);
            mask.push_back(0);
          }
        }
        if (task_comm.rank() == 0) {
          pflow_stats->addColumnValues(task_id+1,pflow,mask);
          qflow_stats->addColumnValues(task_id+1,qflow,mask);
          perf_stats->addColumnValues(task_id+1,perf,mask);
        }
        timer->stop(t_store);
      }
    }
    // Return network to its original base case state
    pf_app.unSetContingency(events[task_id]);
    // Clear Q limit violations AFTER unSetContingency so generators are restored first.
    // This ensures clearQlim() sees the correct generator status when deciding
    // whether to restore p_isPV (PV bus status).
    if (check_Qlim) pf_app.clearQlimViolations();
    // Clear Q limit warnings for next contingency
    gridpack::powerflow::PFBus::clearQlimWarnings();
    // Close output file for this contingency
    if (print_calcs) pf_app.close();
    if (profile && task_comm.rank() == 0) {
      local_profile_indices.push_back(task_id);
      local_profile_records.push_back(profile_record);
    }
  }
  timer->stop(t_tasks);
  if (broker_client) {
    pf_app.setLinearSystemExecutor(NULL);
    broker_client->done();
    broker_executor.reset();
    broker_client.reset();
  }
  int t_output = timer->createCategory("Contingency: Output Pipeline");
  timer->start(t_output);
  // Print statistics from task manager describing the number of tasks performed
  // per processor
  if (schedule_by_expected_pattern) {
    std::vector<int> tasks_per_process(world.size(), 0);
    tasks_per_process[world.rank()] = scheduled_local_tasks;
    MPI_Allreduce(MPI_IN_PLACE, &tasks_per_process[0], world.size(), MPI_INT,
                  MPI_SUM, static_cast<MPI_Comm>(world));
    if (world.rank() == 0) {
      printf("\nNumber of tasks per processors\n");
      for (int process = 0; process < world.size(); ++process) {
        printf("  Number of tasks on process %6d: %6d\n", process,
               tasks_per_process[process]);
      }
    }
  } else {
    taskmgr->printStats();
  }
  brokerDiagnostic(broker_diagnostics, launch_world.rank(),
                   "task statistics complete");

  // Gather stats on successful contingency calculations
#ifdef USE_SUCCESS
  if (task_comm.rank() != 0) {
    contingency_idx.clear();
    contingency_success.clear();
    contingency_violation.clear();
    contingency_isolated.clear();
  }
  gatherContingencyStatus(world, contingency_idx, contingency_success,
                          contingency_violation, contingency_isolated,
                          ntasks);
  brokerDiagnostic(broker_diagnostics, launch_world.rank(),
                   "contingency status gathered");
  contingency_idx.clear();
  for (i=0; i<ntasks; i++) contingency_idx.push_back(i);
  // Write out stats on successful calculations
  if (world.rank() == 0) {
    std::ofstream fout;
    fout.open("success.txt");
    for (i=0; i<ntasks; i++) {
      if (contingency_success[i]) {
        fout << "contingency: " << i+1 << " success: true";
        if (contingency_violation[i] == 1) {
          fout << " violation: none";
        } else if (contingency_violation[i] == 2) {
          fout << " violation: bus";
        } else if (contingency_violation[i] == 3) {
          fout << " violation: branch";
        } else if (contingency_violation[i] == 4) {
          fout << " violation: bus and branch";
        }
        if (contingency_isolated[i]) {
          fout << " warning: isolated";
        }
        fout << std::endl;
      } else {
        fout << "contingency: " << i+1 << " success: false" << std::endl;
      }
    }
    fout.close();
  }
#endif

  if (profile) {
    brokerDiagnostic(broker_diagnostics, launch_world.rank(),
                     "profile records staged");
    std::vector<CAProfileRecord> all_profile_records =
      gatherProfileRecords(world, local_profile_indices,
                           local_profile_records, ntasks);
    brokerDiagnostic(broker_diagnostics, launch_world.rank(),
                     "profile records gathered");
    int base_replicas = task_comm.rank() == 0 ? 1 : 0;
    world.sum(&base_replicas, 1);
    brokerDiagnostic(broker_diagnostics, launch_world.rank(),
                     "profile base count reduced");
    int profile_write_ok = 1;
    if (world.rank() == 0) {
      std::ofstream profile_out(profile_file.c_str());
      if (!profile_out) profile_write_ok = 0;
      profile_out
        << "task_id,contingency,type,status,power_flow_solves,"
        << "linear_solves,newton_updates,reported_newton_iterations,"
        << "controller_passes,area_interchange_passes,mapper_rebuilds,"
        << "mapper_reuses,final_tolerance\n";
      profile_out << -1
        << ",\"base_case_world_rank_0_representative\",base_case,"
        << (base_solve_ok ? "converged" : "divergent") << ","
        << base_profile.powerFlowSolveCalls << ","
        << base_profile.linearSolveCalls << ","
        << base_profile.completedNewtonUpdates << ","
        << base_profile.reportedNewtonIterations << ","
        << base_profile.controllerLoopPasses << ","
        << base_profile.areaInterchangePasses << ","
        << base_profile.mapperWorkspaceRebuilds << ","
        << base_profile.mapperWorkspaceReuses << ","
        << std::setprecision(17) << base_profile.finalTolerance << "\n";

      int status_counts[PROFILE_SLACK_OVERLOAD + 1] = {};
      long long linear_solves = 0;
      long long newton_updates = 0;
      long long controller_passes = 0;
      long long mapper_rebuilds = 0;
      long long mapper_reuses = 0;
      for (i=0; i<ntasks; i++) {
        const CAProfileRecord& record = all_profile_records[i];
        if (record.status >= PROFILE_CONVERGED &&
            record.status <= PROFILE_SLACK_OVERLOAD) {
          status_counts[record.status]++;
        }
        linear_solves += record.linearSolveCalls;
        newton_updates += record.completedNewtonUpdates;
        controller_passes += record.controllerLoopPasses;
        mapper_rebuilds += record.mapperWorkspaceRebuilds;
        mapper_reuses += record.mapperWorkspaceReuses;
        profile_out << i << "," << csvField(events[i].p_name) << ","
          << (events[i].p_type == Branch ? "branch" : "generator") << ","
          << profileStatusName(record.status) << ","
          << record.powerFlowSolveCalls << ","
          << record.linearSolveCalls << ","
          << record.completedNewtonUpdates << ","
          << record.reportedNewtonIterations << ","
          << record.controllerLoopPasses << ","
          << record.areaInterchangePasses << ","
          << record.mapperWorkspaceRebuilds << ","
          << record.mapperWorkspaceReuses << ","
          << std::setprecision(17) << record.finalTolerance << "\n";
      }
      profile_out.close();
      if (!profile_out) profile_write_ok = 0;
      std::cout << "CA_PROFILE_SUMMARY"
        << " converged=" << status_counts[PROFILE_CONVERGED]
        << " divergent=" << status_counts[PROFILE_DIVERGENT]
        << " islanded=" << status_counts[PROFILE_ISLANDED]
        << " rejected=" << status_counts[PROFILE_REJECTED]
        << " slack_overload=" << status_counts[PROFILE_SLACK_OVERLOAD]
        << " base_replicas=" << base_replicas
        << " base_linear_solves_per_replica="
        << base_profile.linearSolveCalls
        << " contingency_linear_solves=" << linear_solves
        << " contingency_newton_updates=" << newton_updates
        << " contingency_controller_passes=" << controller_passes
        << " contingency_mapper_rebuilds=" << mapper_rebuilds
        << " contingency_mapper_reuses=" << mapper_reuses
        << std::endl;
    }
    world.min(&profile_write_ok, 1);
    brokerDiagnostic(broker_diagnostics, launch_world.rank(),
                     "profile output complete");
    if (profile_write_ok == 0) {
      throw gridpack::Exception(
          "Unable to write contingency profile file " + profile_file);
    }
  }

  // Sync GA before MPI collectives to flush any pending one-sided operations
  world.sync();
  brokerDiagnostic(broker_diagnostics, launch_world.rank(),
                   "output GA synchronization complete");

  // Export CA results to JSON or CSV.
  // Only rank 0 writes output files. Non-zero ranks send their serialized
  // data to rank 0 using point-to-point MPI send/recv.
  if (outputFormat == "json") {
    // Each process serializes its contingency results as JSON text
    std::ostringstream localJsonStream;
    for (size_t ci = 0; ci < localContingencies.size(); ci++) {
      gridpack::utility::ResultsExporter::writeContingencyResultJSON(
          localJsonStream, localContingencies[ci]);
      if (ci + 1 < localContingencies.size()) {
        localJsonStream << ",\n";
      }
    }
    std::string localJson = localJsonStream.str();

    // Gather all JSON fragments on rank 0 using point-to-point send/recv
    MPI_Comm mpi_comm = static_cast<MPI_Comm>(world);
    std::vector<std::string> allFragments(world.size());
    allFragments[0] = localJson;  // rank 0's own data
    if (world.rank() == 0) {
      for (int p = 1; p < world.size(); p++) {
        int len;
        MPI_Recv(&len, 1, MPI_INT, p, 0, mpi_comm, MPI_STATUS_IGNORE);
        allFragments[p].resize(len);
        if (len > 0) {
          MPI_Recv(&allFragments[p][0], len, MPI_CHAR, p, 1, mpi_comm,
                   MPI_STATUS_IGNORE);
        }
      }
    } else {
      int len = static_cast<int>(localJson.size());
      MPI_Send(&len, 1, MPI_INT, 0, 0, mpi_comm);
      if (len > 0) {
        MPI_Send(const_cast<char*>(localJson.c_str()), len, MPI_CHAR, 0, 1,
                 mpi_comm);
      }
    }

    // Rank 0 writes the final JSON file
    if (world.rank() == 0) {
      std::string jsonFile = outputFile + ".json";
      std::ofstream jout(jsonFile.c_str());
      gridpack::utility::ResultsExporter::writeCAJSONHeader(jout,
          baseCaseResults);
      bool firstFragment = true;
      for (int p = 0; p < world.size(); p++) {
        if (!allFragments[p].empty()) {
          if (!firstFragment) jout << ",\n";
          jout << allFragments[p];
          firstFragment = false;
        }
      }
      jout << "\n";
      gridpack::utility::ResultsExporter::writeCAJSONFooter(jout);
      jout.close();
    }
  }

  if (outputFormat == "csv") {
    // Each process serializes its contingency CSV data into 4 strings
    // (buses, branches, generators, convergence rows without headers)
    std::ostringstream localBus, localBranch, localGen, localConv;
    localBus << std::fixed;
    localBranch << std::fixed;
    localGen << std::fixed;
    for (size_t ci = 0; ci < localContingencies.size(); ci++) {
      const gridpack::utility::ContingencyResult& ct = localContingencies[ci];
      const gridpack::utility::PowerFlowResults& r = ct.solution;
      for (size_t bi = 0; bi < r.buses.size(); bi++) {
        const gridpack::utility::BusResult& b = r.buses[bi];
        localBus << ct.name << ","
           << b.busId << "," << b.type << ","
           << b.area << "," << b.zone << ","
           << std::setprecision(2) << b.baseKV << ","
           << std::setprecision(6) << b.voltage << ","
           << std::setprecision(6) << b.angle << ","
           << std::setprecision(4) << b.pInjection << ","
           << std::setprecision(4) << b.qInjection << ","
           << std::setprecision(4) << b.pLoad << ","
           << std::setprecision(4) << b.qLoad << ","
           << std::setprecision(4) << b.pGen << ","
           << std::setprecision(4) << b.qGen << ","
           << std::setprecision(4) << b.shuntMvar << "\n";
      }
      for (size_t bi = 0; bi < r.branches.size(); bi++) {
        const gridpack::utility::BranchResult& br = r.branches[bi];
        localBranch << ct.name << ","
           << br.fromBus << "," << br.toBus << ","
           << br.circuitId << ","
           << std::setprecision(4) << br.pFrom << ","
           << std::setprecision(4) << br.qFrom << ","
           << std::setprecision(4) << br.pTo << ","
           << std::setprecision(4) << br.qTo << ","
           << std::setprecision(4) << br.pLoss << ","
           << std::setprecision(4) << br.qLoss << ","
           << std::setprecision(4) << br.mvaFrom << ","
           << std::setprecision(4) << br.mvaTo << ","
           << std::setprecision(4) << br.rateA << ","
           << std::setprecision(2) << br.loadingPercent << "\n";
      }
      for (size_t gi = 0; gi < r.generators.size(); gi++) {
        const gridpack::utility::GeneratorResult& g = r.generators[gi];
        localGen << ct.name << ","
           << g.busId << "," << g.genId << ","
           << std::setprecision(4) << g.pGen << ","
           << std::setprecision(4) << g.qGen << ","
           << std::setprecision(4) << g.qMax << ","
           << std::setprecision(4) << g.qMin << ","
           << std::setprecision(6) << g.voltageSetpoint << ","
           << g.status << "\n";
      }
      localConv << ct.name << ","
         << (r.convergence.converged ? "true" : "false") << ","
         << r.convergence.iterations << ","
         << std::scientific << r.convergence.finalTolerance << ","
         << std::fixed
         << r.convergence.finalMismatch.maxPBus << ","
         << std::setprecision(4) << r.convergence.finalMismatch.maxPMismatch << ","
         << r.convergence.finalMismatch.maxQBus << ","
         << std::setprecision(4) << r.convergence.finalMismatch.maxQMismatch << "\n";
    }

    // Gather all CSV fragments on rank 0 using point-to-point send/recv
    MPI_Comm mpi_comm = static_cast<MPI_Comm>(world);
    std::vector<std::string> allBus(world.size()), allBranch(world.size());
    std::vector<std::string> allGen(world.size()), allConv(world.size());
    allBus[0] = localBus.str();
    allBranch[0] = localBranch.str();
    allGen[0] = localGen.str();
    allConv[0] = localConv.str();
    if (world.rank() == 0) {
      for (int p = 1; p < world.size(); p++) {
        int lens[4];
        MPI_Recv(lens, 4, MPI_INT, p, 0, mpi_comm, MPI_STATUS_IGNORE);
        allBus[p].resize(lens[0]);
        allBranch[p].resize(lens[1]);
        allGen[p].resize(lens[2]);
        allConv[p].resize(lens[3]);
        if (lens[0] > 0)
          MPI_Recv(&allBus[p][0], lens[0], MPI_CHAR, p, 1, mpi_comm,
                   MPI_STATUS_IGNORE);
        if (lens[1] > 0)
          MPI_Recv(&allBranch[p][0], lens[1], MPI_CHAR, p, 2, mpi_comm,
                   MPI_STATUS_IGNORE);
        if (lens[2] > 0)
          MPI_Recv(&allGen[p][0], lens[2], MPI_CHAR, p, 3, mpi_comm,
                   MPI_STATUS_IGNORE);
        if (lens[3] > 0)
          MPI_Recv(&allConv[p][0], lens[3], MPI_CHAR, p, 4, mpi_comm,
                   MPI_STATUS_IGNORE);
      }
    } else {
      std::string sBus = localBus.str(), sBranch = localBranch.str();
      std::string sGen = localGen.str(), sConv = localConv.str();
      int lens[4] = {(int)sBus.size(), (int)sBranch.size(),
                     (int)sGen.size(), (int)sConv.size()};
      MPI_Send(lens, 4, MPI_INT, 0, 0, mpi_comm);
      if (lens[0] > 0)
        MPI_Send(const_cast<char*>(sBus.c_str()), lens[0], MPI_CHAR, 0, 1,
                 mpi_comm);
      if (lens[1] > 0)
        MPI_Send(const_cast<char*>(sBranch.c_str()), lens[1], MPI_CHAR, 0, 2,
                 mpi_comm);
      if (lens[2] > 0)
        MPI_Send(const_cast<char*>(sGen.c_str()), lens[2], MPI_CHAR, 0, 3,
                 mpi_comm);
      if (lens[3] > 0)
        MPI_Send(const_cast<char*>(sConv.c_str()), lens[3], MPI_CHAR, 0, 4,
                 mpi_comm);
    }

    // Rank 0 writes the CSV files
    if (world.rank() == 0) {
      // Write base case first (creates files with headers)
      gridpack::utility::ResultsExporter::writePFCSV(outputFile,
          baseCaseResults, "base_case");
      // Append contingency data from all processes
      {
        std::ofstream out((outputFile + "_buses.csv").c_str(), std::ios::app);
        for (size_t p = 0; p < allBus.size(); p++) out << allBus[p];
      }
      {
        std::ofstream out((outputFile + "_branches.csv").c_str(), std::ios::app);
        for (size_t p = 0; p < allBranch.size(); p++) out << allBranch[p];
      }
      {
        std::ofstream out((outputFile + "_generators.csv").c_str(), std::ios::app);
        for (size_t p = 0; p < allGen.size(); p++) out << allGen[p];
      }
      {
        std::ofstream out((outputFile + "_convergence.csv").c_str(), std::ios::app);
        for (size_t p = 0; p < allConv.size(); p++) out << allConv[p];
      }
#ifdef USE_SUCCESS
      // Write summary CSV
      std::string summaryFile = outputFile + "_summary.csv";
      std::ofstream sout(summaryFile.c_str());
      sout << "contingency,type,converged,has_voltage_violation,has_branch_violation\n";
      for (int ci = 0; ci < ntasks; ci++) {
        bool converged = (contingency_violation[ci] > 0);
        sout << events[ci].p_name << ","
             << (events[ci].p_type == Branch ? "branch" : "generator") << ","
             << (converged ? "true" : "false") << ","
             << ((contingency_violation[ci] == 2 || contingency_violation[ci] == 4)
                 ? "true" : "false") << ","
             << ((contingency_violation[ci] == 3 || contingency_violation[ci] == 4)
                 ? "true" : "false") << "\n";
      }
      sout.close();
#endif
    }
  }

  // Print out statistics on contingencies
  if (write_stats) {
    int t_stats = timer->createCategory("Write Statistics");
    timer->start(t_stats);
    vmag_stats->writeMeanAndRMS("vmag.txt",1,false);
    vmag_stats->writeMinAndMax("vmag_mm.txt",1,false);
    if (check_Qlim) vmag_stats->writeMaskValueCount("pq_change_cnt.txt",2,false);
    vang_stats->writeMeanAndRMS("vang.txt",1,false);
    vang_stats->writeMinAndMax("vang_mm.txt",1,false);
    pgen_stats->writeMeanAndRMS("pgen.txt",1);
    pgen_stats->writeMinAndMax("pgen_mm.txt",1);
    qgen_stats->writeMeanAndRMS("qgen.txt",1);
    qgen_stats->writeMinAndMax("qgen_mm.txt",1);
    pflow_stats->writeMeanAndRMS("pflow.txt",1);
    pflow_stats->writeMinAndMax("pflow_mm.txt",1);
    pflow_stats->writeMaskValueCount("line_flt_cnt.txt",2);
    qflow_stats->writeMeanAndRMS("qflow.txt",1);
    qflow_stats->writeMinAndMax("qflow_mm.txt",1);
    perf_stats->writeMinAndMax("perf_mm.txt",1);
    perf_stats->sumColumnValues("perf_sum.txt",1);
    timer->stop(t_stats);
  }
  timer->stop(t_output);
  if (broker_enabled) {
    brokerDiagnostic(broker_diagnostics, launch_world.rank(),
                     "entering final protocol barrier");
    MPI_Barrier(broker_communicator);
    ga_default_pgroup.reset();
    broker_abort_guard.complete();
    MPI_Comm_free(&broker_communicator);
  }
  timer->stop(t_total);
  // If all processors executed at least one task, then print out timing
  // statistics (this printout does not work if some processors do not define
  // all timing variables)
  // CoarseTimer::dump() always reduces over GA_MPI_Comm (the launch world).
  // In broker mode the dedicated GPU owner has already left the CA worker
  // path, so invoking that collective here would deadlock during teardown.
  if (!broker_enabled && events.size()*grp_size >= world.size()) {
    timer->dump();
  }
}
