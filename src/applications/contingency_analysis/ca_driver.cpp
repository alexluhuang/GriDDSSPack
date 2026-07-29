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
 * @updated Yousu Chen
 * - csv_flat / csv_delta per-(contingency,branch) outputs
 * - monitorBranchesFile / monitorAreas / monitorKvMin/Max filters
 * @date  2026-06-21
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
#include "gridpack/utilities/results_exporter.hpp"
#include "gridpack/math/linear_solver_backend.hpp"
#include "ca_driver.hpp"
#include "ca_async_writer.hpp"
#ifdef GRIDPACK_WITH_CUDSS
#include "gridpack/applications/modules/powerflow/pf_batch_ca.hpp"
#include "gridpack/applications/modules/powerflow/pf_batch_ca_assembler.hpp"
#endif
#include <map>

#include <boost/scoped_ptr.hpp>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cctype>
#include <climits>
#include <set>

// Statistical-summary output (vmag.txt, pflow.txt, etc.) used to be controlled
// by a USE_STATBLOCK build-time macro; it is now a runtime XML option,
// `Configuration.Contingency_analysis.writeStats`, defaulting to true to
// preserve existing behavior.
// Sets up multiple communicators so that individual contingency calculations
// can be run concurrently

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
      if (!contingencies[idx]->get("CKT",&names)) {
        contingencies[idx]->get("contingencyLineNames",&names);
      }
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
      if (!contingencies[idx]->get("GenID",&gens)) {
        contingencies[idx]->get("contingencyGenerators",&gens);
      }
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

    if (gridpack::parallel::Communicator().rank() == 0) {
      printf("Auto-generated %d N-1 branch contingencies\n", branch_count);
    }
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

    if (gridpack::parallel::Communicator().rank() == 0) {
      printf("Auto-generated %d N-1 generator contingencies\n", gen_count);
    }
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

  // Get timer instance for timing entire calculation
  gridpack::utility::CoarseTimer *timer =
    gridpack::utility::CoarseTimer::instance();
  int t_total = timer->createCategory("Total Application");
  timer->start(t_total);

  // Compatibility-preserving CA profile. The legacy GridPACK categories below
  // remain unchanged; these six non-overlapping phases partition the successful
  // CADriver::execute interval so CPU, optimized CPU, and GPU runs have common
  // apples-to-apples boundaries.
  int t_ca_config = timer->createCategory("CA: Configuration");
  int t_ca_model_setup =
    timer->createCategory("CA: Model and Output Setup");
  int t_ca_base = timer->createCategory("CA: Base Case");
  int t_ca_contingency_setup =
    timer->createCategory("CA: Contingency Setup");
  int t_ca_processing =
    timer->createCategory("CA: Contingency Processing");
  int t_ca_finalization =
    timer->createCategory("CA: Result Finalization");

  // Common implementation detail. These categories retain identical meanings
  // on the CPU-only and GPU-with-fallback paths.
  int t_ca_task_dispatch = timer->createCategory("CA: Task Dispatch");
  int t_ca_case_setup = timer->createCategory("CA: Case Setup");
  int t_ca_exact_solve =
    timer->createCategory("CA: Exact Per-Case Solve");
  int t_ca_case_output =
    timer->createCategory("CA: Case Evaluation and Output");
  int t_ca_case_restore = timer->createCategory("CA: Case Restore");
  int t_ca_flat_format =
    timer->createCategory("CA: Flat Result Formatting");
  int t_ca_flat_submit = timer->createCategory("CA: Flat Output Submit");
  int t_ca_flat_finalize =
    timer->createCategory("CA: Flat Output Finalize");
  int t_ca_convergence_output =
    timer->createCategory("CA: Convergence Output");

  // GPU-specific diagnostics. They are deliberately separate from the common
  // CA phases and must not be compared directly with legacy CPU categories.
  int t_ca_gpu_invariants =
    timer->createCategory("CA GPU: Invariant Setup");
  int t_ca_gpu_wave_prepare =
    timer->createCategory("CA GPU: Wave Preparation");
  int t_ca_gpu_batch_newton =
    timer->createCategory("CA GPU: Batch Newton");
  int t_ca_gpu_controllers =
    timer->createCategory("CA GPU: Controller Checks");
  int t_ca_gpu_overlay =
    timer->createCategory("CA GPU: Result Overlay");
  int t_ca_gpu_restore =
    timer->createCategory("CA GPU: State Restore");
  timer->createCategory("CA GPU: Solver Setup and Symbolic Analysis");
  timer->createCategory("CA GPU: Host Assembly and Update");
  timer->createCategory(
      "CA GPU: Numeric Factorization and Triangular Solve");

  timer->start(t_ca_config);

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
  // Statistical-summary output (vmag.txt, pflow.txt, etc. via StatBlock).
  // Default true to preserve existing behavior; set false to skip the
  // per-case StatBlock work and the 13 post-loop global writes.
  bool write_stats = true;
  if (cursor->get("writeStats",&tmp_bool)) {
    util.toLower(tmp_bool);
    write_stats = (tmp_bool != "false");
  }
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
  double qlim_deadband = cursor->get("qlimDeadband", 0.1);
  // Exhaustive per-(contingency, branch) rows are the default contract.
  std::string outputFormat = "csv_flat";
  cursor->get("outputFormat", &outputFormat);
  std::string outputFile = "ca_results";
  cursor->get("outputFile", &outputFile);
  // Phase-5 I/O overlap: when true, per-contingency csv_flat rows are written by
  // a background thread (a spare Grace core) so disk I/O overlaps the next
  // solve.  Output is byte-identical to the synchronous default (single FIFO
  // consumer, same row order); default false preserves exact behavior.
  bool overlapIO = false;
  bool sharedFlatFile = true;
  bool bufferFlatOutput = false;
  {
    std::string t;
    if (cursor->get("overlapIO", &t)) { util.toLower(t); overlapIO = (t == "true"); }
    t.clear();
    if (cursor->get("sharedFlatFile", &t)) {
      util.toLower(t);
      sharedFlatFile = (t != "false");
    }
    t.clear();
    if (cursor->get("bufferFlatOutput", &t)) {
      util.toLower(t);
      bufferFlatOutput = (t == "true");
    }
  }
  // Optional CSV allowlist (from_bus,to_bus,ckt). Empty -> emit all.
  std::string monitorBranchesFile;
  cursor->get("monitorBranchesFile", &monitorBranchesFile);
  // Optional area/kV gates. Empty/zero/missing -> no restriction on that
  // dimension. Filters AND together with monitorBranchesFile.
  std::string monitorAreasStr;
  cursor->get("monitorAreas", &monitorAreasStr);
  double monitorKvMin = 0.0;
  cursor->get("monitorKvMin", &monitorKvMin);
  double monitorKvMax = 0.0;
  cursor->get("monitorKvMax", &monitorKvMax);
  std::set<int> monitorAreas;
  {
    std::vector<std::string> tok = util.blankTokenizer(monitorAreasStr);
    for (size_t i = 0; i < tok.size(); i++) {
      if (!tok[i].empty()) monitorAreas.insert(atoi(tok[i].c_str()));
    }
  }
  // Which rating column csv_flat/csv_delta emit.  In csv_flat, rate_mva,
  // loading_percent, and viol all use this same selected limit. A|B|C,
  // default C.
  // Missing B falls back to A; missing C falls back to B and then A.
  std::string contingencyRating = "C";
  cursor->get("contingencyRating", &contingencyRating);
  util.toUpper(contingencyRating);
  if (contingencyRating != "A" && contingencyRating != "B" &&
      contingencyRating != "C") {
    if (world.rank() == 0) {
      printf("WARNING: contingencyRating='%s' not A/B/C; defaulting to C\n",
             contingencyRating.c_str());
    }
    contingencyRating = "C";
  }
  // ---------------------------------------------------------------------
  // GPU (NVIDIA cuDSS) linear-solver backend selection.  Pure opt-in: the GPU
  // path is chosen only when input.xml requests it AND the binary was built
  // with cuDSS AND a CUDA device is visible; otherwise the run uses the
  // existing PETSc/direct-LU CPU path unchanged.  This must be set here, before
  // any LinearSolver is constructed inside the power-flow solves.  The CLI
  // (mpirun -n K ca.x input.xml) is unaffected.
  // ---------------------------------------------------------------------
  bool gpuEnabled = false;
  {
    std::string tmp_gpu;
    gridpack::utility::Configuration::CursorPtr gpucur =
      config->getCursor("Configuration.Contingency_analysis.GPU");
    if (gpucur && gpucur->get("enabled", &tmp_gpu)) {
      util.toLower(tmp_gpu);
      gpuEnabled = (tmp_gpu == "true");
    }
    // The linear-solver backend is considered only after the explicit CA GPU
    // switch is enabled.
    std::string backendName;
    gridpack::utility::Configuration::CursorPtr lscur =
      config->getCursor("Configuration.Powerflow.LinearSolver");
    if (lscur) lscur->get("Backend", &backendName);
    // Is the batched wave engine requested?  If so we keep the DEFAULT backend on
    // CPU sparse LU on purpose: the wave drives cuDSS directly (CuDSSBatchedSolver),
    // so the one-time base powerflow and the non-batchable per-contingency tail
    // should use fast CPU LU rather than a full cuDSS analyze+factor of the whole
    // system.  On a large network the base solve on cuDSS alone costs ~20 s (vs a
    // few seconds on KLU) -- pure fixed overhead that would sink the GPU wall.
    bool batchedReq = false;
    {
      std::string bt;
      if (gpucur && gpucur->get("batched", &bt)) {
        util.toLower(bt);
        batchedReq = (bt == "true");
      }
    }
    gridpack::math::LinearSolverBackend requested =
      gridpack::math::LinearSolverBackend::PETSc;
    if (gpuEnabled && batchedReq &&
        gridpack::math::cudssBackendAvailable()) {
      // wave uses cuDSS directly; base + fallback stay on CPU LU
      requested = gridpack::math::LinearSolverBackend::PETSc;
    } else if (gpuEnabled) {
      requested = gridpack::math::LinearSolverBackend::CuDSS;
    }
    gridpack::math::setDefaultLinearSolverBackend(requested);
    if (world.rank() == 0) {
      gridpack::math::LinearSolverBackend actual =
        gridpack::math::resolveLinearSolverBackend();
      if (requested == gridpack::math::LinearSolverBackend::CuDSS &&
          actual != gridpack::math::LinearSolverBackend::CuDSS) {
        printf("GPU linear solver requested but unavailable "
               "(binary not built with cuDSS or no CUDA device visible); "
               "falling back to CPU PETSc backend.\n");
      } else {
        printf("Linear solver backend: %s\n",
               gridpack::math::linearSolverBackendName(actual));
      }
      // The Phase-2 batched contingency engine (many contingencies solved in
      // one cuDSS batch, one shared symbolic analysis) is activated below, just
      // before the contingency loop, once qlim/groupSize are known (it requires
      // qlim=false and groupSize=1 so the per-bus structure is stable -- see
      // pf_batch_ca_assembler.hpp).
    }
  }
  // Set static flag for PFBus class BEFORE network creation.
  // This controls how Q values are reported in output functions:
  // - When check_Qlim = false: output uses calculated Q from p_Qinj
  // - When check_Qlim = true: output uses p_qg (set by chkQlim())
  gridpack::powerflow::PFBus::setQlim(check_Qlim);
  gridpack::powerflow::PFBus::setQlimDeadband(qlim_deadband);
  timer->stop(t_ca_config);
  timer->start(t_ca_model_setup);
  gridpack::parallel::Communicator task_comm = world.divide(grp_size);

  // Create powerflow applications on each task communicator
  boost::shared_ptr<gridpack::powerflow::PFNetwork>
    pf_network(new gridpack::powerflow::PFNetwork(task_comm));
  gridpack::powerflow::PFAppModule pf_app;
  pf_app.suppressOutput(!print_calcs);
  double _ts0 = MPI_Wtime();
  bool _tsp = (std::getenv("GRIDPACK_BATCH_PROFILE") != NULL);
  // Read in the network from an external file and partition it over the
  // processors in the task communicator. This will read in power flow
  // parameters from the Powerflow block in the input
  pf_app.readNetwork(pf_network,config);
  if (_tsp && world.rank()==0) fprintf(stderr,"[TS] readNetwork %.2fs\n", MPI_Wtime()-_ts0);
  // Finish initializing the network
  pf_app.initialize();
  if (_tsp && world.rank()==0) fprintf(stderr,"[TS] initialize %.2fs\n", MPI_Wtime()-_ts0);

  // Build (number -> name) lookup tables for area, zone, owner.
  // Keyed on the PSS/E-assigned number (not contiguous), used when
  // emitting per-(branch,contingency) CSV rows so each row carries
  // human-readable area/zone/owner names alongside the numbers.
  std::map<int, std::string> area_name_by_num;
  std::map<int, std::string> zone_name_by_num;
  std::map<int, std::string> owner_name_by_num;
  {
    boost::shared_ptr<gridpack::component::DataCollection> netdata =
      pf_network->getNetworkData();
    int aT = 0, zT = 0, oT = 0;
    netdata->getValue(AREA_TOTAL,  &aT);
    netdata->getValue(ZONE_TOTAL,  &zT);
    netdata->getValue(OWNER_TOTAL, &oT);
    for (int i = 0; i < aT; i++) {
      int n = 0; std::string s;
      netdata->getValue(AREAINTG_NUMBER, &n, i);
      netdata->getValue(AREAINTG_NAME,   &s, i);
      area_name_by_num[n] = s;
    }
    for (int i = 0; i < zT; i++) {
      int n = 0; std::string s;
      netdata->getValue(ZONE_NUMBER, &n, i);
      netdata->getValue(ZONE_NAME,   &s, i);
      zone_name_by_num[n] = s;
    }
    for (int i = 0; i < oT; i++) {
      int n = 0; std::string s;
      netdata->getValue(OWNER_NUMBER, &n, i);
      netdata->getValue(OWNER_NAME,   &s, i);
      owner_name_by_num[n] = s;
    }
  }

  // Per-rank bus metadata + wide/long branch-row outputs for csv_flat
  // (long-form one row per branch per case) and csv_delta (wide-form one
  // row per branch per case joining base and cont state). Both share the
  // bus_meta load, the buses sidecar, and the per-rank .part-file gather.
  bool wantBusSidecar = (outputFormat == "csv_flat" ||
                         outputFormat == "csv_delta");
  struct BusMeta {
    std::string name;
    double basekv;
    int    area, zone, owner;
  };
  std::map<int, BusMeta> bus_meta;
  if (wantBusSidecar) {
    int nBus = pf_network->numBuses();
    for (int i = 0; i < nBus; i++) {
      gridpack::powerflow::PFBus *bus =
        dynamic_cast<gridpack::powerflow::PFBus*>(pf_network->getBus(i).get());
      if (!bus) continue;
      int orig = pf_network->getOriginalBusIndex(i);
      BusMeta m;
      m.name   = bus->getBusName();
      m.basekv = bus->getBaseKV();
      m.area   = bus->getArea();
      m.zone   = bus->getZone();
      m.owner  = bus->getOwner();
      bus_meta[orig] = m;
    }
  }

  // Strip surrounding single quotes (PSS/E style) and outer whitespace.
  auto trim_quoted = [](const std::string &in) -> std::string {
    std::string s = in;
    size_t a = s.find_first_not_of(" \t");
    size_t b = s.find_last_not_of(" \t");
    if (a == std::string::npos) return std::string();
    s = s.substr(a, b - a + 1);
    if (s.size() >= 2 && s.front() == '\'' && s.back() == '\'') {
      s = s.substr(1, s.size() - 2);
    }
    a = s.find_first_not_of(" \t");
    b = s.find_last_not_of(" \t");
    return (a == std::string::npos) ? std::string() : s.substr(a, b - a + 1);
  };
  auto lookup_name = [&](const std::map<int, std::string> &m, int n) -> std::string {
    std::map<int, std::string>::const_iterator it = m.find(n);
    if (it == m.end()) return std::string();
    return trim_quoted(it->second);
  };

  // Per-rank bus metadata sidecar (deduped by world rank 0 after the loop).
  if (wantBusSidecar) {
    std::ostringstream oss;
    oss << outputFile << "_buses." << world.rank() << ".part";
    std::ofstream fbus(oss.str().c_str(),
                       std::ios::out | std::ios::trunc | std::ios::binary);
    fbus << std::fixed;
    for (std::map<int, BusMeta>::const_iterator it = bus_meta.begin();
         it != bus_meta.end(); ++it) {
      const BusMeta &m = it->second;
      fbus << it->first << ","
           << trim_quoted(m.name) << ","
           << std::setprecision(2) << m.basekv << ","
           << m.area << "," << m.zone << "," << m.owner << ","
           << lookup_name(area_name_by_num,  m.area)  << ","
           << lookup_name(zone_name_by_num,  m.zone)  << ","
           << lookup_name(owner_name_by_num, m.owner)
           << "\n";
    }
    fbus.close();
  }

  // Per-rank streaming output file. Opened on first row written so non-csv_flat
  // runs and ranks that produce no rows leave nothing behind.
  std::string flatPartPath;
  gridpack::contingency_analysis::AsyncRowWriter flatPart;
  const char* flatHeader =
    "event_idx,contingency,from_bus,to_bus,circuit_id,"
    "p_from_mw,q_from_mvar,mva_from,rate_mva,loading_percent,"
    "viol,v_from_pu,v_to_pu,ang_from_deg,ang_to_deg\n";
  if (outputFormat == "csv_flat") {
    if (bufferFlatOutput) {
      // Finalized with disjoint MPI-IO writes after the contingency loop.
    } else if (sharedFlatFile) {
      flatPartPath = outputFile + "_flat.csv";
      if (world.rank() == 0) {
        std::ofstream flatOutput(flatPartPath.c_str(),
            std::ios::out | std::ios::trunc | std::ios::binary);
        flatOutput << flatHeader;
      }
      world.sync();
      flatPart.open(flatPartPath, overlapIO, true);
    } else {
      std::ostringstream oss;
      oss << outputFile << "_flat." << world.rank() << ".part";
      flatPartPath = oss.str();
      flatPart.open(flatPartPath, overlapIO);
    }
  }
  size_t flatRowCount = 0;
  std::string flatMemory;
  auto writeFlatBlock = [&](const std::string& block) {
    if (bufferFlatOutput) {
      flatMemory.append(block);
    } else {
      flatPart.write(block);
    }
  };

  // (from, to, ckt) key shared by the monitor allowlist and base_cache.
  struct BranchKey {
    int from, to;
    std::string ckt;
    bool operator<(const BranchKey &o) const {
      if (from != o.from) return from < o.from;
      if (to   != o.to  ) return to   < o.to;
      return ckt < o.ckt;
    }
  };

  // Per-branch rate-A/B/C from the parsed network data. Keyed by (from,to,ckt)
  // so the csv_flat / csv_delta emit paths can pick the configured rating.
  // Built once before the contingency loop. Each rank only sees its own
  // active+ghost branches; that's fine -- the emit path is also rank-local.
  struct BranchRates {
    double rate_a, rate_b, rate_c;
  };
  std::map<BranchKey, BranchRates> branch_rates;
  if (outputFormat == "csv_flat" || outputFormat == "csv_delta") {
    int nBranch = pf_network->numBranches();
    for (int i = 0; i < nBranch; i++) {
      boost::shared_ptr<gridpack::component::DataCollection> bd =
        pf_network->getBranchData(i);
      if (!bd) continue;
      int from = 0, to = 0, nelems = 0;
      bd->getValue(BRANCH_FROMBUS, &from);
      bd->getValue(BRANCH_TOBUS,   &to);
      if (!bd->getValue(BRANCH_NUM_ELEMENTS, &nelems)) continue;
      for (int k = 0; k < nelems; k++) {
        std::string ckt;
        if (!bd->getValue(BRANCH_CKT, &ckt, k)) continue;
        // Trim leading/trailing whitespace and PSS/E surrounding quotes.
        size_t a = ckt.find_first_not_of(" \t");
        size_t b = ckt.find_last_not_of(" \t");
        ckt = (a == std::string::npos) ? std::string()
                                       : ckt.substr(a, b - a + 1);
        if (ckt.size() >= 2 && ckt.front() == '\'' && ckt.back() == '\'') {
          ckt = ckt.substr(1, ckt.size() - 2);
          a = ckt.find_first_not_of(" \t");
          b = ckt.find_last_not_of(" \t");
          ckt = (a == std::string::npos) ? std::string()
                                         : ckt.substr(a, b - a + 1);
        }
        BranchRates r;
        r.rate_a = 0.0; r.rate_b = 0.0; r.rate_c = 0.0;
        bd->getValue(BRANCH_RATING_A, &r.rate_a, k);
        bd->getValue(BRANCH_RATING_B, &r.rate_b, k);
        bd->getValue(BRANCH_RATING_C, &r.rate_c, k);
        BranchKey key;
        key.from = from; key.to = to; key.ckt = ckt;
        branch_rates[key] = r;
      }
    }
  }
  // Base case always uses rate-A (PSS/E "normal" rating). Contingency rows
  // use whichever the user picked, with B->A or C->B->A fallback.
  auto pickContRate = [&](const BranchRates &r) -> double {
    if (contingencyRating == "A") {
      return r.rate_a;
    }
    if (contingencyRating == "B") {
      return (r.rate_b > 0.0) ? r.rate_b : r.rate_a;
    }
    if (r.rate_c > 0.0) return r.rate_c;
    if (r.rate_b > 0.0) return r.rate_b;
    return r.rate_a;
  };

  // Monitor allowlist parsed from monitorBranchesFile. Empty -> emit all.
  std::set<BranchKey> monitorSet;
  if (!monitorBranchesFile.empty() &&
      (outputFormat == "csv_flat" || outputFormat == "csv_delta")) {
    std::ifstream fin(monitorBranchesFile.c_str());
    if (!fin.is_open()) {
      if (world.rank() == 0) {
        printf("WARNING: monitorBranchesFile '%s' not found; emitting all branches\n",
               monitorBranchesFile.c_str());
      }
    } else {
      std::string line;
      size_t lineNo = 0;
      while (std::getline(fin, line)) {
        lineNo++;
        // Strip trailing CR (Windows line endings).
        while (!line.empty() && (line[line.size()-1] == '\r' ||
                                 line[line.size()-1] == '\n')) {
          line.resize(line.size()-1);
        }
        // Skip blank lines and comments.
        size_t firstNon = line.find_first_not_of(" \t");
        if (firstNon == std::string::npos) continue;
        if (line[firstNon] == '#') continue;
        // Tokenize on commas.
        std::vector<std::string> tok;
        size_t pos = 0;
        while (pos <= line.size()) {
          size_t comma = line.find(',', pos);
          std::string t = (comma == std::string::npos)
                          ? line.substr(pos)
                          : line.substr(pos, comma - pos);
          size_t a = t.find_first_not_of(" \t");
          size_t b = t.find_last_not_of(" \t");
          tok.push_back((a == std::string::npos) ? std::string()
                                                 : t.substr(a, b - a + 1));
          if (comma == std::string::npos) break;
          pos = comma + 1;
        }
        if (tok.size() < 3) continue;
        // Skip header row: any non-numeric first field.
        if (tok[0].empty()) continue;
        bool numeric = true;
        for (size_t ci = 0; ci < tok[0].size(); ci++) {
          char c = tok[0][ci];
          if (!(c >= '0' && c <= '9') && c != '-' && c != '+') {
            numeric = false; break;
          }
        }
        if (!numeric) continue;
        BranchKey k;
        k.from = atoi(tok[0].c_str());
        k.to   = atoi(tok[1].c_str());
        k.ckt  = tok[2];
        monitorSet.insert(k);
      }
      if (world.rank() == 0) {
        printf("Monitor allowlist: %zu branches loaded from %s\n",
               monitorSet.size(), monitorBranchesFile.c_str());
      }
    }
  }
  auto isMonitored = [&](const BranchKey &k) {
    return monitorSet.empty() || monitorSet.find(k) != monitorSet.end();
  };
  // Area/kV gate. Either-endpoint match for areas (catches tie-lines).
  // kV is gated on max(kv_from, kv_to) so a 138/13.8 stepdown counts as 138.
  // Empty area set / zero kV bound = unrestricted on that dimension.
  auto passesAreaKv = [&](int area_from, int area_to,
                          double kv_from, double kv_to) {
    if (!monitorAreas.empty()) {
      if (monitorAreas.find(area_from) == monitorAreas.end() &&
          monitorAreas.find(area_to)   == monitorAreas.end()) {
        return false;
      }
    }
    double kv_max = (kv_from > kv_to) ? kv_from : kv_to;
    if (monitorKvMin > 0.0 && kv_max < monitorKvMin) return false;
    if (monitorKvMax > 0.0 && kv_max > monitorKvMax) return false;
    return true;
  };
  // When monitorBranchesFile presents, it overrides area/kV criteria.
  bool haveAreaKvFilter = !monitorAreas.empty() ||
                          monitorKvMin > 0.0 ||
                          monitorKvMax > 0.0;
  if (!monitorSet.empty() && haveAreaKvFilter) {
    if (world.rank() == 0) {
      printf("WARNING: monitorBranchesFile is set; ignoring "
             "monitorAreas/monitorKvMin/monitorKvMax\n");
    }
    monitorAreas.clear();
    monitorKvMin = 0.0;
    monitorKvMax = 0.0;
    haveAreaKvFilter = false;
  }
  if (world.rank() == 0) {
    if (!monitorAreas.empty()) {
      printf("Monitor areas filter: %zu areas\n", monitorAreas.size());
    }
    if (monitorKvMin > 0.0 || monitorKvMax > 0.0) {
      printf("Monitor kV filter: min=%.2f max=%.2f (0 means unbounded)\n",
             monitorKvMin, monitorKvMax);
    }
  }

  struct BaseFlow {
    double p_mw, q_mvar, mva, loading_pct;
    double base_rate, cont_rate;
    double v_from_pu, v_to_pu, ang_from_deg, ang_to_deg;
    double base_kv_from, base_kv_to;
    int    area_from, area_to;
  };
  std::map<BranchKey, BaseFlow> base_cache;
  std::string deltaPartPath;
  if (outputFormat == "csv_delta") {
    std::ostringstream oss;
    oss << outputFile << "_delta." << world.rank() << ".part";
    deltaPartPath = oss.str();
  }
  std::ofstream deltaPart;
  size_t deltaRowCount = 0;
  size_t deltaSkipCount = 0;

  // Convergence sidecar rows.
  struct ConvRow {
    int    event_idx;
    std::string name;
    std::string type;
    gridpack::utility::ConvergenceSummary cs;
    std::string status;
  };
  std::vector<ConvRow> localConvRows;
  // _convergence.csv: written for every outputFormat.
  bool emitConv = true;

  // Quote a string only when CSV requires it.  Contingency names are supplied
  // by the input deck and are not restricted to identifier characters.
  auto csvField = [](const std::string &value) -> std::string {
    if (value.find_first_of(",\"\r\n") == std::string::npos) return value;
    std::string quoted;
    quoted.reserve(value.size() + 2);
    quoted.push_back('"');
    for (size_t i = 0; i < value.size(); ++i) {
      if (value[i] == '"') quoted.push_back('"');
      quoted.push_back(value[i]);
    }
    quoted.push_back('"');
    return quoted;
  };

  // The event prefix is prepared once per case and retained at the beginning
  // of rowScratch.  Only the fixed-format branch suffix is regenerated for
  // each row.  In the uncommon event that the suffix exceeds its initial
  // capacity, resize and retry instead of emitting a truncated CSV row.
  auto appendFlatRow = [](std::string &block, std::vector<char> &rowScratch,
                          size_t prefixLen, int from, int to,
                          const char *ckt, double p, double q,
                          double flowMva, double rateSelected,
                          double loading, int viol, double vFrom,
                          double vTo, double aFrom, double aTo) {
    int suffixLen = -1;
    for (;;) {
      const size_t suffixCapacity = rowScratch.size() - prefixLen;
      suffixLen = std::snprintf(&rowScratch[prefixLen], suffixCapacity,
          "%d,%d,%s,%.4f,%.4f,%.4f,%.4f,%.2f,%d,"
          "%.6f,%.6f,%.4f,%.4f\n",
          from, to, ckt, p, q, flowMva, rateSelected, loading,
          viol, vFrom, vTo, aFrom, aTo);
      if (suffixLen < 0) {
        throw gridpack::Exception("Unable to format csv_flat row");
      }
      if (static_cast<size_t>(suffixLen) < suffixCapacity) break;
      rowScratch.resize(prefixLen + static_cast<size_t>(suffixLen) + 1);
    }
    block.append(&rowScratch[0], prefixLen + static_cast<size_t>(suffixLen));
  };

  // Lambda: parse current solved flow_str/vr_str and stream one CSV row
  // per branch into the rank's .part file. Called once per converged case
  // (base + each contingency) on every task communicator; non-rank-0
  // task_comm members short-circuit after the collective.
  auto captureFlatRows = [&](int event_idx, const std::string &name,
                             bool emit, bool is_base) {
    if (task_comm.size() == 1 && !std::getenv("GRIDPACK_FLAT_LEGACY")) {
      if (!emit) return;
      gridpack::utility::ScopedTimer formatTimer(timer, t_ca_flat_format);
      std::string frow;
      frow.reserve(static_cast<size_t>(pf_network->numBranches()) * 160);
      std::ostringstream prefixStream;
      prefixStream << event_idx << "," << csvField(name) << ",";
      const std::string rowPrefix = prefixStream.str();
      const size_t prefixLen = rowPrefix.size();
      std::vector<char> rowScratch(rowPrefix.begin(), rowPrefix.end());
      rowScratch.resize(prefixLen + 256);
      const double radiansToDegrees = 180.0 / (4.0 * std::atan(1.0));
      const int nbranch = pf_network->numBranches();
      for (int bi = 0; bi < nbranch; ++bi) {
        gridpack::powerflow::PFBranch* branch =
          dynamic_cast<gridpack::powerflow::PFBranch*>(
              pf_network->getBranch(bi).get());
        if (!branch) continue;
        int fromLocal = -1, toLocal = -1;
        pf_network->getBranchEndpoints(bi, &fromLocal, &toLocal);
        gridpack::powerflow::PFBus* fromBus =
          dynamic_cast<gridpack::powerflow::PFBus*>(
              pf_network->getBus(fromLocal).get());
        gridpack::powerflow::PFBus* toBus =
          dynamic_cast<gridpack::powerflow::PFBus*>(
              pf_network->getBus(toLocal).get());
        if (!fromBus || !toBus) continue;
        const int from = branch->getBus1OriginalIndex();
        const int to = branch->getBus2OriginalIndex();
        double vFrom = 0.0, aFrom = 0.0, vTo = 0.0, aTo = 0.0;
        fromBus->getVoltageState(vFrom, aFrom);
        toBus->getVoltageState(vTo, aTo);
        aFrom *= radiansToDegrees;
        aTo *= radiansToDegrees;
        const std::vector<std::string> tags = branch->getLineIDs();
        for (size_t li = 0; li < tags.size(); ++li) {
          std::string ckt = tags[li];
          while (!ckt.empty() && ckt[ckt.size() - 1] == ' ') ckt.resize(ckt.size() - 1);
          while (!ckt.empty() && ckt[0] == ' ') ckt.erase(0, 1);
          BranchKey mk;
          mk.from = from;
          mk.to = to;
          mk.ckt = ckt;
          if (!monitorSet.empty() && monitorSet.find(mk) == monitorSet.end()) continue;
          if (haveAreaKvFilter) {
            std::map<int, BusMeta>::const_iterator mf = bus_meta.find(from);
            std::map<int, BusMeta>::const_iterator mt = bus_meta.find(to);
            const int af = (mf != bus_meta.end()) ? mf->second.area : 0;
            const int at = (mt != bus_meta.end()) ? mt->second.area : 0;
            const double kf = (mf != bus_meta.end()) ? mf->second.basekv : 0.0;
            const double kt = (mt != bus_meta.end()) ? mt->second.basekv : 0.0;
            if (!passesAreaKv(af, at, kf, kt)) continue;
          }
          gridpack::ComplexType power = branch->getComplexPower(tags[li]);
          double p = real(power), q = imag(power);
          if (!branch->getBranchStatus(tags[li]) ||
              fromBus->isIsolated() || toBus->isIsolated()) {
            p = 0.0;
            q = 0.0;
          }
          const double rateA = branch->getBranchRatingA(tags[li]);
          double rateSelected = rateA;
          if (!is_base) {
            if (contingencyRating == "B") {
              rateSelected = branch->getBranchRatingB(tags[li]);
              if (rateSelected <= 0.0) rateSelected = rateA;
            } else if (contingencyRating == "C") {
              rateSelected = branch->getBranchRatingC(tags[li]);
              if (rateSelected <= 0.0) {
                rateSelected = branch->getBranchRatingB(tags[li]);
                if (rateSelected <= 0.0) rateSelected = rateA;
              }
            }
          }
          const double flowMva = std::sqrt(p * p + q * q);
          const double loading = rateSelected > 0.0 ?
            100.0 * flowMva / rateSelected : 0.0;
          const int viol =
            rateSelected > 0.0 && flowMva > rateSelected ? 1 : 0;
          appendFlatRow(frow, rowScratch, prefixLen, from, to, ckt.c_str(),
                        p, q, flowMva, rateSelected, loading, viol, vFrom,
                        vTo, aFrom, aTo);
          ++flatRowCount;
        }
      }
      formatTimer.stop();
      gridpack::utility::ScopedTimer submitTimer(timer, t_ca_flat_submit);
      writeFlatBlock(frow);
      return;
    }
    gridpack::utility::ScopedTimer formatTimer(timer, t_ca_flat_format);
    std::vector<std::string> v_strs = pf_app.writeBusString("vr_str");
    std::vector<std::string> b_strs = pf_app.writeBranchString("flow_str");
    if (!emit || task_comm.rank() != 0) return;
    // Buffer this contingency's rows into one block, then hand it to the writer
    // (Phase-5 I/O overlap).  Formatted with snprintf into a preallocated string
    // -- numerically identical to the old ostringstream(std::fixed) path but far
    // cheaper per row, which matters when a wave emits millions of branch rows
    // (the 24k-bus training network writes ~6M rows / >800 MB).
    std::string frow;
    frow.reserve(b_strs.size() * 100 + 64);
    std::ostringstream prefixStream;
    prefixStream << event_idx << "," << csvField(name) << ",";
    const std::string rowPrefix = prefixStream.str();
    const size_t prefixLen = rowPrefix.size();
    std::vector<char> rowScratch(rowPrefix.begin(), rowPrefix.end());
    rowScratch.resize(prefixLen + 256);
    std::map<int, std::pair<double,double> > vbymag_ang;
    for (size_t vi = 0; vi < v_strs.size(); vi++) {
      int    bus_id = 0, use_vmag = 0, changed = 0;
      double angle = 0.0, vmag = 0.0;
      if (sscanf(v_strs[vi].c_str(), "%d %lf %lf %d %d",
                 &bus_id, &angle, &vmag, &use_vmag, &changed) == 5) {
        vbymag_ang[bus_id] = std::make_pair(vmag, angle);
      }
    }
    for (size_t bi = 0; bi < b_strs.size(); bi++) {
      char ckt_buf[16] = {0};
      int rateAViol = 0;
      double p = 0.0, q = 0.0, perf = 0.0, ratea = 0.0;
      int from = 0, to = 0;
      if (sscanf(b_strs[bi].c_str(),
                 "%d %d %15s %lf %lf %lf %lf %d",
                 &from, &to, ckt_buf, &p, &q, &perf, &ratea,
                 &rateAViol) != 8) {
        continue;
      }
      char ckt[4];
      std::strncpy(ckt, ckt_buf, 3); ckt[3] = '\0';
      BranchKey mk;
      mk.from = from; mk.to = to; mk.ckt = ckt;
      while (!mk.ckt.empty() && mk.ckt[mk.ckt.size()-1] == ' ')
        mk.ckt.resize(mk.ckt.size()-1);
      if (!monitorSet.empty() && monitorSet.find(mk) == monitorSet.end()) continue;
      if (haveAreaKvFilter) {
        std::map<int, BusMeta>::const_iterator mf = bus_meta.find(from);
        std::map<int, BusMeta>::const_iterator mt = bus_meta.find(to);
        int af = (mf != bus_meta.end()) ? mf->second.area   : 0;
        int at = (mt != bus_meta.end()) ? mt->second.area   : 0;
        double kf = (mf != bus_meta.end()) ? mf->second.basekv : 0.0;
        double kt = (mt != bus_meta.end()) ? mt->second.basekv : 0.0;
        if (!passesAreaKv(af, at, kf, kt)) continue;
      }
      std::map<BranchKey, BranchRates>::const_iterator rIt = branch_rates.find(mk);
      double rate_sel = ratea;
      if (rIt != branch_rates.end()) {
        rate_sel = is_base ? rIt->second.rate_a : pickContRate(rIt->second);
      }
      double flow_mva    = std::sqrt(p*p + q*q);
      double loading_pct = (rate_sel > 0.0) ? (flow_mva / rate_sel) * 100.0 : 0.0;
      std::map<int, std::pair<double,double> >::const_iterator vf =
        vbymag_ang.find(from);
      std::map<int, std::pair<double,double> >::const_iterator vt =
        vbymag_ang.find(to);
      double v_from       = (vf != vbymag_ang.end()) ? vf->second.first  : 0.0;
      double ang_from_deg = (vf != vbymag_ang.end()) ? vf->second.second : 0.0;
      double v_to         = (vt != vbymag_ang.end()) ? vt->second.first  : 0.0;
      double ang_to_deg   = (vt != vbymag_ang.end()) ? vt->second.second : 0.0;
      const int viol =
        rate_sel > 0.0 && flow_mva > rate_sel ? 1 : 0;
      appendFlatRow(frow, rowScratch, prefixLen, from, to, ckt, p, q,
                    flow_mva, rate_sel, loading_pct, viol, v_from, v_to,
                    ang_from_deg, ang_to_deg);
      flatRowCount++;
    }
    formatTimer.stop();
    gridpack::utility::ScopedTimer submitTimer(timer, t_ca_flat_submit);
    writeFlatBlock(frow);
  };

  // Populate base_cache from current solved state. Called once after base
  // solve on every rank (csv_delta only); world.rank() == 0 is not special
  // here -- each rank caches the branches it sees on its task_comm so it
  // can join later in captureDeltaRows.
  auto populateBaseCache = [&]() {
    std::vector<std::string> v_strs = pf_app.writeBusString("vr_str");
    std::vector<std::string> b_strs = pf_app.writeBranchString("flow_str");
    if (task_comm.rank() != 0) return;
    std::map<int, std::pair<double,double> > vbymag_ang;
    for (size_t vi = 0; vi < v_strs.size(); vi++) {
      int    bus_id = 0, use_vmag = 0, changed = 0;
      double angle = 0.0, vmag = 0.0;
      if (sscanf(v_strs[vi].c_str(), "%d %lf %lf %d %d",
                 &bus_id, &angle, &vmag, &use_vmag, &changed) == 5) {
        vbymag_ang[bus_id] = std::make_pair(vmag, angle);
      }
    }
    for (size_t bi = 0; bi < b_strs.size(); bi++) {
      char ckt_buf[16] = {0};
      int viol = 0;
      double p = 0.0, q = 0.0, perf = 0.0, ratea = 0.0;
      int from = 0, to = 0;
      if (sscanf(b_strs[bi].c_str(),
                 "%d %d %15s %lf %lf %lf %lf %d",
                 &from, &to, ckt_buf, &p, &q, &perf, &ratea, &viol) != 8) {
        continue;
      }
      BranchKey k;
      k.from = from; k.to = to;
      k.ckt  = std::string(ckt_buf);
      // Strip trailing spaces from ckt so the key matches what flow_str
      // returns later (sscanf %15s already trims leading whitespace).
      while (!k.ckt.empty() && k.ckt[k.ckt.size()-1] == ' ') k.ckt.resize(k.ckt.size()-1);
      if (!isMonitored(k)) continue;
      if (haveAreaKvFilter) {
        std::map<int, BusMeta>::const_iterator mf = bus_meta.find(from);
        std::map<int, BusMeta>::const_iterator mt = bus_meta.find(to);
        int af = (mf != bus_meta.end()) ? mf->second.area   : 0;
        int at = (mt != bus_meta.end()) ? mt->second.area   : 0;
        double kf = (mf != bus_meta.end()) ? mf->second.basekv : 0.0;
        double kt = (mt != bus_meta.end()) ? mt->second.basekv : 0.0;
        if (!passesAreaKv(af, at, kf, kt)) continue;
      }
      double base_rate = ratea, cont_rate = ratea;
      std::map<BranchKey, BranchRates>::const_iterator rIt = branch_rates.find(k);
      if (rIt != branch_rates.end()) {
        base_rate = rIt->second.rate_a;
        cont_rate = pickContRate(rIt->second);
      }
      BaseFlow bf;
      bf.p_mw        = p;
      bf.q_mvar      = q;
      bf.mva         = std::sqrt(p*p + q*q);
      bf.base_rate   = base_rate;
      bf.cont_rate   = cont_rate;
      bf.loading_pct = (base_rate > 0.0) ? (bf.mva / base_rate) * 100.0 : 0.0;
      std::map<int, std::pair<double,double> >::const_iterator vf =
        vbymag_ang.find(from);
      std::map<int, std::pair<double,double> >::const_iterator vt =
        vbymag_ang.find(to);
      bf.v_from_pu     = (vf != vbymag_ang.end()) ? vf->second.first  : 0.0;
      bf.ang_from_deg  = (vf != vbymag_ang.end()) ? vf->second.second : 0.0;
      bf.v_to_pu       = (vt != vbymag_ang.end()) ? vt->second.first  : 0.0;
      bf.ang_to_deg    = (vt != vbymag_ang.end()) ? vt->second.second : 0.0;
      std::map<int, BusMeta>::const_iterator mf = bus_meta.find(from);
      std::map<int, BusMeta>::const_iterator mt = bus_meta.find(to);
      bf.base_kv_from = (mf != bus_meta.end()) ? mf->second.basekv : 0.0;
      bf.base_kv_to   = (mt != bus_meta.end()) ? mt->second.basekv : 0.0;
      bf.area_from    = (mf != bus_meta.end()) ? mf->second.area   : 0;
      bf.area_to      = (mt != bus_meta.end()) ? mt->second.area   : 0;
      base_cache[k] = bf;
    }
  };

  // Wide-form (base+cont on same row) capture for csv_delta. Mirrors
  // captureFlatRows but joins each branch with base_cache. Branches not
  // in base_cache are counted in deltaSkipCount and skipped silently.
  auto captureDeltaRows = [&](int event_idx,
                              const gridpack::powerflow::Contingency &evt,
                              bool emit) {
    std::vector<std::string> v_strs = pf_app.writeBusString("vr_str");
    std::vector<std::string> b_strs = pf_app.writeBranchString("flow_str");
    if (!emit || task_comm.rank() != 0) return;
    if (!deltaPart.is_open()) {
      deltaPart.open(deltaPartPath.c_str(), std::ios::out | std::ios::trunc);
      deltaPart << std::fixed;
    }
    // cont_event_facility: built once per contingency.
    std::string facility;
    if (evt.p_type == Branch && !evt.p_from.empty()) {
      int outFrom = evt.p_from[0];
      int area = 0;
      std::map<int, BusMeta>::const_iterator mf = bus_meta.find(outFrom);
      if (mf != bus_meta.end()) area = mf->second.area;
      char buf[64];
      snprintf(buf, sizeof(buf), "[%d] %d %d %s",
               area, outFrom, evt.p_to[0], evt.p_ckt[0].c_str());
      facility = buf;
      if (evt.p_from.size() > 1) {
        char suf[24];
        snprintf(suf, sizeof(suf), " (+%zu more)", evt.p_from.size() - 1);
        facility += suf;
      }
    } else if (evt.p_type == Generator && !evt.p_busid.empty()) {
      char buf[48];
      snprintf(buf, sizeof(buf), "gen %d %s",
               evt.p_busid[0], evt.p_genid[0].c_str());
      facility = buf;
      if (evt.p_busid.size() > 1) {
        char suf[24];
        snprintf(suf, sizeof(suf), " (+%zu more)", evt.p_busid.size() - 1);
        facility += suf;
      }
    }
    std::string ct_name = evt.p_name;
    while (!ct_name.empty() && ct_name[ct_name.size()-1] == ' ')
      ct_name.resize(ct_name.size()-1);
    const char *type_str = (evt.p_type == Branch) ? "branch" : "generator";
    std::map<int, std::pair<double,double> > vbymag_ang;
    for (size_t vi = 0; vi < v_strs.size(); vi++) {
      int    bus_id = 0, use_vmag = 0, changed = 0;
      double angle = 0.0, vmag = 0.0;
      if (sscanf(v_strs[vi].c_str(), "%d %lf %lf %d %d",
                 &bus_id, &angle, &vmag, &use_vmag, &changed) == 5) {
        vbymag_ang[bus_id] = std::make_pair(vmag, angle);
      }
    }
    for (size_t bi = 0; bi < b_strs.size(); bi++) {
      char ckt_buf[16] = {0};
      int viol = 0;
      double p = 0.0, q = 0.0, perf = 0.0, ratea = 0.0;
      int from = 0, to = 0;
      if (sscanf(b_strs[bi].c_str(),
                 "%d %d %15s %lf %lf %lf %lf %d",
                 &from, &to, ckt_buf, &p, &q, &perf, &ratea, &viol) != 8) {
        continue;
      }
      BranchKey k;
      k.from = from; k.to = to;
      k.ckt  = std::string(ckt_buf);
      while (!k.ckt.empty() && k.ckt[k.ckt.size()-1] == ' ')
        k.ckt.resize(k.ckt.size()-1);
      if (!isMonitored(k)) continue;
      std::map<BranchKey, BaseFlow>::const_iterator it = base_cache.find(k);
      if (it == base_cache.end()) { deltaSkipCount++; continue; }
      const BaseFlow &bf = it->second;
      double cont_mva     = std::sqrt(p*p + q*q);
      double cont_loading = (bf.cont_rate > 0.0) ? (cont_mva / bf.cont_rate) * 100.0 : 0.0;
      std::map<int, std::pair<double,double> >::const_iterator vf =
        vbymag_ang.find(from);
      std::map<int, std::pair<double,double> >::const_iterator vt =
        vbymag_ang.find(to);
      double v_from_c = (vf != vbymag_ang.end()) ? vf->second.first  : 0.0;
      double a_from_c = (vf != vbymag_ang.end()) ? vf->second.second : 0.0;
      double v_to_c   = (vt != vbymag_ang.end()) ? vt->second.first  : 0.0;
      double a_to_c   = (vt != vbymag_ang.end()) ? vt->second.second : 0.0;
      double d_ang_b  = bf.ang_from_deg - bf.ang_to_deg;
      double d_ang_c  = a_from_c - a_to_c;
      deltaPart << event_idx << "," << ct_name << "," << type_str << ","
                << from << "," << to << "," << k.ckt << ","
                << std::setprecision(2) << bf.base_kv_from << ","
                << std::setprecision(2) << bf.base_kv_to   << ","
                << bf.area_from << "," << bf.area_to << ","
                << std::setprecision(4) << bf.base_rate << ","
                << std::setprecision(4) << bf.cont_rate << ","
                << std::setprecision(4) << bf.p_mw   << ","
                << std::setprecision(4) << p         << ","
                << std::setprecision(4) << bf.q_mvar << ","
                << std::setprecision(4) << q         << ","
                << std::setprecision(4) << bf.mva    << ","
                << std::setprecision(4) << cont_mva  << ","
                << std::setprecision(2) << bf.loading_pct << ","
                << std::setprecision(2) << cont_loading  << ","
                << std::setprecision(6) << bf.v_from_pu << ","
                << std::setprecision(6) << v_from_c     << ","
                << std::setprecision(6) << bf.v_to_pu   << ","
                << std::setprecision(6) << v_to_c       << ","
                << std::setprecision(4) << bf.ang_from_deg << ","
                << std::setprecision(4) << a_from_c        << ","
                << std::setprecision(4) << bf.ang_to_deg   << ","
                << std::setprecision(4) << a_to_c          << ","
                << std::setprecision(4) << d_ang_b << ","
                << std::setprecision(4) << d_ang_c << ","
                << facility
                << "\n";
      deltaRowCount++;
    }
  };

  timer->stop(t_ca_model_setup);
  timer->start(t_ca_base);

  //  Set minimum and maximum voltage limits on all buses
  pf_app.setVoltageLimits(Vmin, Vmax);
  // Solve the base power flow on every task communicator. Abort if it fails.
  bool baseSolveOk = false;
  try {
    baseSolveOk = pf_app.solve();
    if (baseSolveOk && check_Qlim && !pf_app.checkQlimViolations()) {
      baseSolveOk = pf_app.solve();
    }
  } catch (const std::exception &e) {
    if (world.rank() == 0) {
      printf("ERROR: base-case solve threw exception: %s\n", e.what());
    }
    baseSolveOk = false;
  } catch (...) {
    if (world.rank() == 0) {
      printf("ERROR: base-case solve threw unknown exception\n");
    }
    baseSolveOk = false;
  }
  if (!baseSolveOk) {
    if (world.rank() == 0) {
      gridpack::utility::ConvergenceSummary cs = pf_app.getConvergence();
      printf("ERROR: base case did not converge "
             "(iterations=%d, final_tol=%.6e, "
             "max_p_bus=%d max_p_mismatch=%.4f, "
             "max_q_bus=%d max_q_mismatch=%.4f). "
             "Aborting contingency analysis.\n",
             cs.iterations, cs.finalTolerance,
             cs.finalMismatch.maxPBus, cs.finalMismatch.maxPMismatch,
             cs.finalMismatch.maxQBus, cs.finalMismatch.maxQMismatch);
    }
    world.barrier();
    MPI_Abort(static_cast<MPI_Comm>(world), 1);
  }
  if (_tsp && world.rank()==0) fprintf(stderr,"[TS] baseSolve %.2fs\n", MPI_Wtime()-_ts0);
  // Suppress voltage violations already present at base.
  pf_app.ignoreVoltageViolations();

  // Collect base case results for export. csv_flat captures rows directly
  // in the hot loop and skips the heavyweight collectResults() path.
  gridpack::utility::PowerFlowResults baseCaseResults;
  if (outputFormat == "json" || outputFormat == "csv") {
    baseCaseResults = pf_app.collectResults();
  }
  if (outputFormat == "csv_flat") {
    // The base case is replicated on every task communicator. captureFlatRows
    // calls writeBusString/writeBranchString which are task_comm collectives,
    // so every task_comm participates -- but only world rank 0 emits rows so
    // the base case isn't duplicated in the final file.
    captureFlatRows(0, std::string("base_case"), world.rank() == 0, true);
  }
  if (outputFormat == "csv_delta") {
    // Cache base-case branch state on every rank for the contingency join.
    populateBaseCache();
  }

  timer->stop(t_ca_base);
  timer->start(t_ca_contingency_setup);

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

  // Print contingency details (gated on printCalcFiles; noisy for large lists)
  if (print_calcs && world.rank() == 0) {
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
  gridpack::parallel::TaskManager taskmgr(world);
  int ntasks = events.size();
  taskmgr.set(ntasks);

  int nbus = pf_network->totalBuses();
  // Get bus voltage information for base case
  int i, j;
  // StatBlock objects and the per-case scratch vectors live across the
  // contingency loop so they are declared up here, regardless of whether
  // statistics output is enabled.
  boost::scoped_ptr<gridpack::analysis::StatBlock> vmag_stats;
  boost::scoped_ptr<gridpack::analysis::StatBlock> vang_stats;
  boost::scoped_ptr<gridpack::analysis::StatBlock> pgen_stats;
  boost::scoped_ptr<gridpack::analysis::StatBlock> qgen_stats;
  boost::scoped_ptr<gridpack::analysis::StatBlock> pflow_stats;
  boost::scoped_ptr<gridpack::analysis::StatBlock> qflow_stats;
  boost::scoped_ptr<gridpack::analysis::StatBlock> perf_stats;
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

  // ---------------------------------------------------------------------
  // Phase-2 GPU batched contingency engine: state + activation.
  // A wave of branch contingencies that leave every bus's PV/PQ/slack/isolated
  // status unchanged share ONE cuDSS symbolic analysis and are refactorized +
  // solved in batches on the GPU (pf_batch_ca.hpp / pf_batch_ca_assembler.hpp).
  // It requires qlim=false and groupSize=1 (so the whole network is serial on a
  // rank and the reduced-Jacobian structure is stable); everything else falls
  // back to the exact per-contingency path.  Output is captured through the same
  // code as the per-contingency loop, so all formats stay identical.
  // ---------------------------------------------------------------------
  std::map<int,int> batchIndexByTask;        // task_id -> index in the cuDSS batch
  std::vector<char> batchConverged;          // per batched case
  std::map<int, gridpack::utility::ConvergenceSummary> batchConvByTask;
#ifdef GRIDPACK_WITH_CUDSS
  boost::scoped_ptr<gridpack::powerflow::GridpackBatchAssembler> batchAsm;
  bool batchRankHealthy = true;
#endif
  bool gpu_batched = false;
  bool batchWarmStart = true;
  bool batchConnectivityScreen = true;
  int batchWaveSize = 8;
  {
    std::string t;
    gridpack::utility::Configuration::CursorPtr gcur =
      config->getCursor("Configuration.Contingency_analysis.GPU");
    if (gcur && gcur->get("batched", &t)) { util.toLower(t); gpu_batched = (t == "true"); }
    t.clear();
    if (gcur && gcur->get("warmStart", &t)) { util.toLower(t); batchWarmStart = (t != "false"); }
    t.clear();
    if (gcur && gcur->get("screen", &t)) {
      util.toLower(t);
      batchConnectivityScreen = (t != "false");
    }
    t.clear();
    if (gcur && gcur->get("waveSize", &t)) {
      util.toLower(t);
      if (t != "auto") {
        errno = 0;
        char *endptr = NULL;
        const long parsed = std::strtol(t.c_str(), &endptr, 10);
        while (endptr && *endptr &&
               std::isspace(static_cast<unsigned char>(*endptr))) {
          ++endptr;
        }
        if (errno == ERANGE || endptr == t.c_str() ||
            (endptr && *endptr != '\0') || parsed <= 0 ||
            parsed > static_cast<long>(INT_MAX)) {
          throw gridpack::Exception(
              std::string("Invalid GPU waveSize '") + t +
              "': use 'auto' or a positive integer");
        }
        batchWaveSize = static_cast<int>(parsed);
      }
    }
  }
  double batchTol = 1.0e-6;
  int batchMaxIter = 50;
  int batchChordCap = 0;
  int batchRefactorEvery = 1;
  bool batchConstantFactor = false;
  double batchDamping = 1.0;
  // Outer-loop controllers the batched inner Newton does NOT run; if any is on,
  // the batched result would diverge from the per-contingency solve, so the
  // batched engine must not engage (route to the correct per-contingency path).
  bool pf_switchedShunt = false, pf_ltc = false, pf_areaInterchange = false;
  {
    gridpack::utility::Configuration::CursorPtr pcur =
      config->getCursor("Configuration.Powerflow");
    if (pcur) {
      pcur->get("tolerance", &batchTol);
      pcur->get("maxIteration", &batchMaxIter);
      pcur->get("dampingFactor", &batchDamping);
      std::string t;
      if (pcur->get("SwitchedShunt", &t)) { util.toLower(t); pf_switchedShunt = (t == "true"); }
      t.clear();
      if (pcur->get("LTC", &t)) { util.toLower(t); pf_ltc = (t == "true"); }
      t.clear();
      if (pcur->get("AreaInterchange", &t)) { util.toLower(t); pf_areaInterchange = (t == "true"); }
    }
    gridpack::utility::Configuration::CursorPtr lcur =
      config->getCursor("Configuration.Powerflow.LinearSolver");
    if (lcur) {
      lcur->get("refactorEvery", &batchRefactorEvery);
      lcur->get("chordCap", &batchChordCap);
      std::string cf;
      if (lcur->get("constantFactor", &cf)) {
        util.toLower(cf);
        if (cf == "true") {
          batchRefactorEvery = batchMaxIter;   // factor once
          batchConstantFactor = true;          // engage the chord path in solveWave
        }
      }
    }
  }
  if (batchChordCap <= 0) batchChordCap = batchMaxIter;
  // The batched wave drives cuDSS DIRECTLY (CuDSSBatchedSolver), so it only needs
  // cuDSS to be built + a device present -- not the default backend to be cuDSS.
  // In fact for the batched path the default backend is deliberately left on CPU
  // LU (fast base solve + fallback); see the backend-selection block above.
  bool cudssActive = gridpack::math::cudssBackendAvailable();
  // The batched wave solves the inner Newton with the base PV/PQ status.  The
  // outer-loop controllers with a per-case check hook (qlim, switched shunt,
  // LTC) are handled AFTER the wave: any case a controller would act on is
  // routed to the per-contingency fallback (the exact solve() with the full
  // controller loop), so batched results match the CPU path within tolerance.
  // qlim in particular touches only the handful of cases that actually hit a
  // reactive limit.  Area interchange is an outermost slack-redistribution loop
  // with no per-case check hook, so with it enabled the batch is disabled and
  // every case uses the per-contingency GPU path (still correct, just not
  // wave-accelerated).
  bool useBatched = gpuEnabled && gpu_batched && cudssActive &&
                    (grp_size == 1) &&
                    !pf_areaInterchange;
  const bool batchOutputFullRefresh = pf_switchedShunt || pf_ltc;
  if (useBatched) {
    // The assembler keeps voltage magnitude and angle for every case in the
    // rank-local wave: 2 * waveSize * nbus doubles.  Bound that storage to
    // 256 MiB per rank and impose an absolute 256-case cap so a mistyped deck
    // cannot create a multi-gigabyte allocation.
    const unsigned long long stateBudget = 256ULL * 1024ULL * 1024ULL;
    const unsigned long long buses =
      static_cast<unsigned long long>(std::max(1, nbus));
    const unsigned long long perCaseBytes =
      2ULL * static_cast<unsigned long long>(sizeof(double)) * buses;
    unsigned long long memoryCap = stateBudget / perCaseBytes;
    if (memoryCap == 0) memoryCap = 1;
    const int waveCap = static_cast<int>(
        std::min<unsigned long long>(256ULL, memoryCap));
    if (batchWaveSize > waveCap) {
      if (world.rank() == 0) {
        printf("NOTE: GPU waveSize=%d exceeds the per-rank limit for this "
               "%d-bus case; capped at %d (256 MiB/256-case maximum).\n",
               batchWaveSize, nbus, waveCap);
      }
      batchWaveSize = waveCap;
    }
  }
  if (gpu_batched && world.rank() == 0) {
    if (!gpuEnabled)
      printf("NOTE: <GPU><batched> is ignored because <GPU><enabled> is not "
             "true; using the CPU path.\n");
    else if (!cudssActive)
      printf("NOTE: <GPU><batched> requested but cuDSS is unavailable (binary not "
             "built with cuDSS or no CUDA device visible); using the CPU path.\n");
    else if (grp_size != 1)
      printf("NOTE: <GPU><batched> requested but groupSize=%d; the batched engine "
             "requires groupSize=1. Using the per-contingency GPU path.\n", grp_size);
    else if (pf_areaInterchange)
      printf("NOTE: <GPU><batched> requested but AreaInterchange is enabled; the "
             "batched wave has no per-case area-slack check, so every case uses "
             "the per-contingency GPU path.\n");
    else
      printf("GPU batched contingency engine ENABLED on all %d rank(s), "
             "waveSize=%d "
             "(tol=%.1e, maxIter=%d, chordCap=%d, solve=%s, warmStart=%s, "
             "damping=%.2f, "
             "qlim=%s, shunt=%s, ltc=%s via post-wave fallback).\n",
             world.size(), batchWaveSize,
             batchTol, batchMaxIter, batchChordCap,
             batchConstantFactor ? "constant-factor/chord" : "exact-Newton",
             batchWarmStart ? "true" : "false", batchDamping,
             check_Qlim ? "on" : "off", pf_switchedShunt ? "on" : "off",
             pf_ltc ? "on" : "off");
  }

  // Convergence row recorder; indexes events[task_id].
  auto recordConv = [&](int task_id, const char *status,
                        const std::string &) {
    if (!emitConv) return;
    if (task_comm.rank() != 0) return;
    ConvRow r;
    r.event_idx = task_id + 1;
    r.name      = events[task_id].p_name;
    r.type      = (events[task_id].p_type == Branch) ? "branch" : "generator";
    // Batched cases skip pf_app.solve(), so pf_app.getConvergence() would be
    // stale -- use the summary injected from the batched solve when present.
    std::map<int, gridpack::utility::ConvergenceSummary>::const_iterator bci =
      batchConvByTask.find(task_id);
    r.cs        = (bci != batchConvByTask.end()) ? bci->second
                                                 : pf_app.getConvergence();
    r.status    = status;
    localConvRows.push_back(r);
  };

  // Evaluate contingencies using the task manager
  int task_id;
  char sbuf[128];
  // The per-contingency work (reset, set contingency, solve or overlay a batched
  // result, capture, restore) is one lambda so the CPU per-contingency loop and
  // the GPU batched path drive IDENTICAL setup/capture/teardown.
  auto runOneCase = [&](int task_id) {
    gridpack::utility::ScopedTimer caseSetupTimer(timer, t_ca_case_setup);
    if (print_calcs) printf("Executing task %d on process %d\n",task_id,world.rank());
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
        if (print_calcs) printf("p[%d] Line: (from) %d (to) %d (line) \'%s\'\n",
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
        if (print_calcs) printf("p[%d] Generator: (bus) %d (generator ID) \'%s\'\n",
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
    // Skip power flow if contingency setup failed (no valid slack) or islanding detected
    bool slackCapacityOk = true;  // Will be checked after solve
    bool solveOk = false;
    bool usedBatchOverlay = false;
    int batchOverlayIndex = -1;
    caseSetupTimer.stop();
#ifdef GRIDPACK_WITH_CUDSS
    std::map<int,int>::iterator _bit = batchIndexByTask.find(task_id);
    if (_bit != batchIndexByTask.end()) {
      gridpack::utility::ScopedTimer overlayTimer(timer, t_ca_gpu_overlay);
      try {
        // Pre-solved by the GPU batch: overlay its converged state (the branch
        // is already out of service from setContingency above) rather than
        // re-solving, so the capture below sees the batched solution.
        batchOverlayIndex = _bit->second;
        if (batchOverlayIndex < 0 ||
            batchOverlayIndex >= static_cast<int>(batchConverged.size())) {
          throw gridpack::Exception("Invalid batched output case index");
        }
        batchAsm->applyCaseForOutput(batchOverlayIndex,
                                     batchOutputFullRefresh);
        solveOk = (batchConverged[batchOverlayIndex] != 0);
        usedBatchOverlay = true;
        // solve() was skipped, so inject the batched convergence summary;
        // otherwise collectResults() would copy a stale p_convergence.
        std::map<int, gridpack::utility::ConvergenceSummary>::iterator _ci =
          batchConvByTask.find(task_id);
        if (_ci != batchConvByTask.end()) pf_app.setConvergence(_ci->second);
      } catch (const std::exception& e) {
        printf("p[%d] batched output overlay failed for task %d: %s; "
               "using the exact CPU path\n", world.rank(), task_id, e.what());
        batchRankHealthy = false;
        batchIndexByTask.clear();
        batchConvByTask.clear();
        pf_app.unSetContingency(events[task_id]);
        try {
          batchAsm->restoreBaseState();
        } catch (...) {
          // The exact solve below rebuilds its network matrices.
        }
        pf_app.resetVoltages();
        pf_network->updateBuses();
        contingencyFound = pf_app.setContingency(events[task_id]);
        islandCount = pf_app.getIslandCount();
        hasLoneBus = pf_app.hasLoneBus();
        islandDetected = (islandCount > 1);
      } catch (...) {
        printf("p[%d] batched output overlay failed for task %d; "
               "using the exact CPU path\n", world.rank(), task_id);
        batchRankHealthy = false;
        batchIndexByTask.clear();
        batchConvByTask.clear();
        pf_app.unSetContingency(events[task_id]);
        try {
          batchAsm->restoreBaseState();
        } catch (...) {
        }
        pf_app.resetVoltages();
        pf_network->updateBuses();
        contingencyFound = pf_app.setContingency(events[task_id]);
        islandCount = pf_app.getIslandCount();
        hasLoneBus = pf_app.hasLoneBus();
        islandDetected = (islandCount > 1);
      }
    }
#endif
    if (!usedBatchOverlay && contingencyFound && !islandDetected) {
      gridpack::utility::ScopedTimer exactSolveTimer(timer, t_ca_exact_solve);
      try {
        solveOk = pf_app.solve();
        if (solveOk && check_Qlim && !pf_app.checkQlimViolations()) {
          pf_app.solve();
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
    gridpack::utility::ScopedTimer caseOutputTimer(timer, t_ca_case_output);
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
        // Slack generator exceeds Pmax - insufficient generation capacity
        // This is treated as a failure, similar to divergence
        if (outputFormat == "json" || outputFormat == "csv") {
          gridpack::utility::ContingencyResult ctResult;
          ctResult.name = events[task_id].p_name;
          ctResult.type = (events[task_id].p_type == Branch) ? "branch" : "generator";
          ctResult.hasVoltageViolation = false;
          ctResult.hasBranchViolation = false;
          ctResult.solution.convergence = pf_app.getConvergence();
          ctResult.solution.convergence.converged = false;
          localContingencies.push_back(ctResult);
        }
        recordConv(task_id, "SLACK_OVERLOAD", std::string());
        sprintf(sbuf,"\nInsufficient generation capacity for contingency %s\n",
            events[task_id].p_name.c_str());
        if (print_calcs) pf_app.print(sbuf);
      } else {
        // Power flow solved and slack within capacity
        // If power flow solution is successful, write out voltages and currents
        if (print_calcs) pf_app.write();
        // Check for violations
        bool ok1 = pf_app.checkVoltageViolations();
        bool ok2 = pf_app.checkLineOverloadViolations();
        bool ok = ok1 && ok2;
        // Collect results for JSON/CSV export
        if (outputFormat == "json" || outputFormat == "csv") {
          gridpack::utility::ContingencyResult ctResult;
          ctResult.name = events[task_id].p_name;
          ctResult.type = (events[task_id].p_type == Branch) ? "branch" : "generator";
          ctResult.hasVoltageViolation = !ok1;
          ctResult.hasBranchViolation = !ok2;
          ctResult.solution = pf_app.collectResults();
          localContingencies.push_back(ctResult);
        }
        if (outputFormat == "csv_flat") {
          captureFlatRows(task_id + 1, events[task_id].p_name, true, false);
        }
        if (outputFormat == "csv_delta") {
          captureDeltaRows(task_id + 1, events[task_id], true);
        }
        recordConv(task_id, "OK", std::string());
      // Include results of violation checks in output
      if (ok) {
        sprintf(sbuf,"\nNo violation for contingency %s\n",
            events[task_id].p_name.c_str());
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
      if (outputFormat == "json" || outputFormat == "csv") {
        gridpack::utility::ContingencyResult ctResult;
        ctResult.name = events[task_id].p_name;
        ctResult.type = (events[task_id].p_type == Branch) ? "branch" : "generator";
        ctResult.hasVoltageViolation = false;
        ctResult.hasBranchViolation = false;
        ctResult.solution.convergence = pf_app.getConvergence();
        ctResult.solution.convergence.converged = false;
        localContingencies.push_back(ctResult);
      }
      {
        const char *st;
        if (islandDetected) {
          st = "ISLANDED";
        } else if (!contingencyFound) {
          st = "NO_SLACK";
        } else {
          st = "DIVERGED";
        }
        recordConv(task_id, st, std::string());
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
    caseOutputTimer.stop();
    gridpack::utility::ScopedTimer caseRestoreTimer(timer, t_ca_case_restore);
    // Return network to its original base case state
    pf_app.unSetContingency(events[task_id]);
#ifdef GRIDPACK_WITH_CUDSS
    if (usedBatchOverlay) {
      try {
        batchAsm->clearCaseForOutput(batchOverlayIndex,
                                     batchOutputFullRefresh);
      } catch (const std::exception& e) {
        printf("p[%d] batched output cleanup failed for task %d: %s\n",
               world.rank(), task_id, e.what());
        batchRankHealthy = false;
        batchIndexByTask.clear();
        batchConvByTask.clear();
        try {
          batchAsm->restoreBaseState();
        } catch (...) {
        }
      } catch (...) {
        printf("p[%d] batched output cleanup failed for task %d\n",
               world.rank(), task_id);
        batchRankHealthy = false;
        batchIndexByTask.clear();
        batchConvByTask.clear();
        try {
          batchAsm->restoreBaseState();
        } catch (...) {
        }
      }
    }
#endif
    // Clear Q limit violations AFTER unSetContingency so generators are restored first.
    // This ensures clearQlim() sees the correct generator status when deciding
    // whether to restore p_isPV (PV bus status).
    if (check_Qlim) pf_app.clearQlimViolations();
    // Clear Q limit warnings for next contingency
    gridpack::powerflow::PFBus::clearQlimWarnings();
    // Close output file for this contingency
    if (print_calcs) pf_app.close();
  };  // end runOneCase

  timer->stop(t_ca_contingency_setup);
  timer->start(t_ca_processing);
#ifdef GRIDPACK_WITH_CUDSS
  if (_tsp && world.rank()==0) fprintf(stderr,"[TS] preContingency %.2fs\n", MPI_Wtime()-_ts0);
  if (useBatched) {
    unsigned long long localGpuWaves = 0;
    unsigned long long localInspectedTasks = 0;
    unsigned long long localEligibleTasks = 0;
    unsigned long long localDirectFallbackTasks = 0;
    unsigned long long localNonConvergedTasks = 0;
    unsigned long long localControllerFallbackTasks = 0;
    unsigned long long localAdaptiveRefactors = 0;
    unsigned long long localScreenedChecks = 0;
    unsigned long long localBridgeCount = 0;
    // Construct the expensive base mapper/CSR/scatter/connectivity invariants
    // once per rank. beginWave() repairs mutable network caches and reuses those
    // invariants for every subsequent task reservation.
    {
      gridpack::utility::ScopedTimer invariantTimer(timer, t_ca_gpu_invariants);
      try {
        const std::vector<int> noTasks;
        batchAsm.reset(new gridpack::powerflow::GridpackBatchAssembler(
            pf_app, events, noTasks, batchWarmStart, batchDamping,
            batchConnectivityScreen));
        localBridgeCount =
          static_cast<unsigned long long>(batchAsm->bridgeCount());
      } catch (const std::exception& e) {
        printf("p[%d] batched GPU initialization failed: %s; this rank will use "
               "the exact CPU path\n", world.rank(), e.what());
        batchRankHealthy = false;
      } catch (...) {
        printf("p[%d] batched GPU initialization failed; this rank will use "
               "the exact CPU path\n", world.rank());
        batchRankHealthy = false;
      }
    }
    for (;;) {
    // Collect this task communicator's share of the contingencies, then run the
    // branch-eligible subset as cuDSS batches (one shared symbolic analysis) and
    // fall back to the per-contingency path for the rest.  Every case is finally
    // driven through runOneCase so the captured output is identical.
    std::vector<int> myTasks;
    bool noMoreTasks = false;
    while (batchWaveSize <= 0 ||
           static_cast<int>(myTasks.size()) < batchWaveSize) {
      bool haveTask = false;
      {
        gridpack::utility::ScopedTimer dispatchTimer(timer, t_ca_task_dispatch);
        haveTask = taskmgr.nextTask(task_comm, &task_id);
      }
      if (haveTask) {
        myTasks.push_back(task_id);
      } else {
        noMoreTasks = true;
        break;
      }
    }
    if (myTasks.empty()) break;
    ++localGpuWaves;
    localInspectedTasks += static_cast<unsigned long long>(myTasks.size());
    batchIndexByTask.clear();
    batchConvByTask.clear();
    batchConverged.clear();

    bool _bprof = (std::getenv("GRIDPACK_BATCH_PROFILE") != NULL);
    double _bt0 = _bprof ? MPI_Wtime() : 0.0;
    int Wbatch = 0;
    int waveDirectFallback = static_cast<int>(myTasks.size());
    int waveNonConverged = 0;
    int waveControllerFallback = 0;
    unsigned long long waveAdaptiveRefactors = 0;
    bool waveBatchSucceeded = false;
    if (batchRankHealthy) {
      try {
        gridpack::utility::ScopedTimer wavePrepareTimer(
            timer, t_ca_gpu_wave_prepare);
        batchAsm->beginWave(myTasks);
        double _bt1 = _bprof ? MPI_Wtime() : 0.0;
        batchAsm->prepare();
        wavePrepareTimer.stop();
        if (_bprof && world.rank() == 0) {
          double _bt2 = MPI_Wtime();
          fprintf(stderr, "[BATCH_SETUP] beginWave=%.2fs prepare=%.2fs "
                  "(cases=%d)\n", _bt1 - _bt0, _bt2 - _bt1,
                  batchAsm->caseCount());
        }

        if (_bprof && world.rank() == 0 && batchConnectivityScreen) {
          printf("[connectivity screen] linear-time bridge pass: %d islanding "
                 "line outage(s), %d per-case topology check(s) avoided\n",
                 batchAsm->bridgeCount(), batchAsm->screenSkipped());
        }

        Wbatch = batchAsm->caseCount();
        waveDirectFallback =
          static_cast<int>(batchAsm->nonBatchTaskIds().size());
        if (Wbatch > 0) {
          gridpack::utility::ScopedTimer batchNewtonTimer(
              timer, t_ca_gpu_batch_newton);
          gridpack::powerflow::PFBatchNR nr(
              *batchAsm, batchTol, batchMaxIter, batchRefactorEvery,
              batchConstantFactor, batchChordCap, batchDamping);
          const std::vector<gridpack::powerflow::BatchCaseStatus>& st =
            nr.solveWave();
          batchConverged.assign(Wbatch, 0);
          for (int k = 0; k < Wbatch; k++) {
            int tid = batchAsm->batchTaskId(k);
            batchIndexByTask[tid] = k;
            batchConverged[k] = st[k].converged ? 1 : 0;
            gridpack::utility::ConvergenceSummary cs;
            cs.converged = st[k].converged;
            cs.iterations = st[k].iterations;
            cs.finalTolerance = st[k].mismatch;
            cs.finalMismatch.maxPBus = st[k].maxPBus;
            cs.finalMismatch.maxPMismatch = st[k].maxPMismatch;
            cs.finalMismatch.maxQBus = st[k].maxQBus;
            cs.finalMismatch.maxQMismatch = st[k].maxQMismatch;
            batchConvByTask[tid] = cs;
            waveAdaptiveRefactors += static_cast<unsigned long long>(
                std::max(0, st[k].refactorizations));
          }
        }
        if (_bprof && world.rank() == 0) {
          printf("[batched GPU] %d branch contingencies solved in the cuDSS "
                 "batch (1 shared symbolic analysis); %d routed to the "
                 "per-contingency path\n", Wbatch, waveDirectFallback);
        }

        // A non-converged modified-Newton case is always re-solved by the exact
        // CPU Newton path; no approximate result is emitted.
        for (int k = 0; k < Wbatch; k++) {
          int tid = batchAsm->batchTaskId(k);
          if (!batchConverged[k] && batchIndexByTask.count(tid)) {
            batchIndexByTask.erase(tid);
            batchConvByTask.erase(tid);
            ++waveNonConverged;
          }
        }
        if (_bprof && waveNonConverged && world.rank() == 0) {
          printf("[batched GPU] %d non-converged case(s) -> exact CPU "
                 "fallback\n", waveNonConverged);
        }

        // A local branch/Ybus overlay is sufficient for qlim. Switched shunts
        // and LTC checks mutate admittance state, so those cases request a full
        // refresh after the controller state is restored.
        if (Wbatch > 0 && (check_Qlim || pf_switchedShunt || pf_ltc)) {
          gridpack::utility::ScopedTimer controllerTimer(
              timer, t_ca_gpu_controllers);
          for (int k = 0; k < Wbatch; k++) {
            int tid = batchAsm->batchTaskId(k);
            if (batchIndexByTask.find(tid) == batchIndexByTask.end()) continue;
            batchAsm->applyCaseForOutput(k, batchOutputFullRefresh);
            bool ok = true;
            if (check_Qlim && !pf_app.checkQlimViolations()) ok = false;
            if (ok && pf_switchedShunt &&
                !pf_app.checkSwitchedShuntViolations()) ok = false;
            if (ok && pf_ltc && !pf_app.checkLTCViolations()) ok = false;
            // Restore controller mutations before rebuilding the base cache.
            if (check_Qlim) pf_app.clearQlimViolations();
            if (pf_switchedShunt) pf_app.clearSwitchedShunts();
            if (pf_ltc) pf_app.clearLTCControls();
            batchAsm->clearCaseForOutput(k, batchOutputFullRefresh);
            if (!ok) {
              batchIndexByTask.erase(tid);
              batchConvByTask.erase(tid);
              ++waveControllerFallback;
            }
          }
          if (_bprof && waveControllerFallback && world.rank() == 0) {
            printf("[batched GPU controllers] %d case(s) hit a qlim/shunt/LTC "
                   "limit -> exact CPU fallback\n", waveControllerFallback);
          }
        }
        waveBatchSucceeded = true;
      } catch (const std::exception& e) {
        printf("p[%d] batched GPU wave failed: %s; all %d tasks will use the "
               "exact CPU path\n", world.rank(), e.what(),
               static_cast<int>(myTasks.size()));
        batchRankHealthy = false;
        batchIndexByTask.clear();
        batchConvByTask.clear();
        batchConverged.clear();
        try {
          if (check_Qlim) pf_app.clearQlimViolations();
          if (pf_switchedShunt) pf_app.clearSwitchedShunts();
          if (pf_ltc) pf_app.clearLTCControls();
        } catch (...) {
        }
        try {
          batchAsm->restoreBaseState();
        } catch (...) {
        }
      } catch (...) {
        printf("p[%d] batched GPU wave failed; all %d tasks will use the exact "
               "CPU path\n", world.rank(), static_cast<int>(myTasks.size()));
        batchRankHealthy = false;
        batchIndexByTask.clear();
        batchConvByTask.clear();
        batchConverged.clear();
        try {
          if (check_Qlim) pf_app.clearQlimViolations();
          if (pf_switchedShunt) pf_app.clearSwitchedShunts();
          if (pf_ltc) pf_app.clearLTCControls();
        } catch (...) {
        }
        try {
          batchAsm->restoreBaseState();
        } catch (...) {
        }
      }
    }

    if (waveBatchSucceeded) {
      localEligibleTasks += static_cast<unsigned long long>(Wbatch);
      localDirectFallbackTasks +=
        static_cast<unsigned long long>(waveDirectFallback);
      localNonConvergedTasks +=
        static_cast<unsigned long long>(waveNonConverged);
      localControllerFallbackTasks +=
        static_cast<unsigned long long>(waveControllerFallback);
      localAdaptiveRefactors += waveAdaptiveRefactors;
      localScreenedChecks +=
        static_cast<unsigned long long>(batchAsm->screenSkipped());
    } else {
      localDirectFallbackTasks +=
        static_cast<unsigned long long>(myTasks.size());
    }

    // Capture every case (batched cases overlay their converged state inside
    // runOneCase; the rest solve normally there).
    //
    // The batched wave ran on cuDSS directly (CuDSSBatchedSolver), independent of
    // the default backend.  The cases runOneCase() still has to *solve* here are
    // exactly the ones the batch could NOT take -- structure-changed / islanded /
    // slack-transfer, non-converged, and qlim/shunt/LTC violators.  A single
    // per-contingency cuDSS solve is ~3x slower than sparse CPU LU at this scale
    // (fresh symbolic analysis + host<->device copies per controller pass), so on
    // radial networks with a high non-batchable fraction that tail can make the
    // GPU path slower than CPU.  Route those residual solves to the CPU LU
    // backend; the batched cases below don't solve at all (they overlay their
    // converged batch state), so this only touches the fallback tail.  Results are
    // identical -- same Newton, different linear solver.
    gridpack::math::setDefaultLinearSolverBackend(
        gridpack::math::LinearSolverBackend::PETSc);
    // Emit all retained GPU states while the assembler's wave-local state is
    // still current. CPU fallbacks can rebuild global Ybus/Sbus caches, so they
    // run only after every retained overlay has been captured.
    std::vector<int> retainedTasks;
    std::vector<int> fallbackTasks;
    retainedTasks.reserve(myTasks.size());
    fallbackTasks.reserve(myTasks.size());
    for (size_t ti = 0; ti < myTasks.size(); ti++) {
      if (batchIndexByTask.find(myTasks[ti]) != batchIndexByTask.end())
        retainedTasks.push_back(myTasks[ti]);
      else
        fallbackTasks.push_back(myTasks[ti]);
    }
    for (size_t ti = 0; ti < retainedTasks.size(); ti++)
      runOneCase(retainedTasks[ti]);
    if (batchAsm) {
      gridpack::utility::ScopedTimer restoreTimer(timer, t_ca_gpu_restore);
      try {
        batchAsm->restoreBaseState();
      } catch (const std::exception& e) {
        printf("p[%d] failed to restore the batched base state before CPU "
               "fallbacks: %s\n", world.rank(), e.what());
        batchRankHealthy = false;
      } catch (...) {
        printf("p[%d] failed to restore the batched base state before CPU "
               "fallbacks\n", world.rank());
        batchRankHealthy = false;
      }
    }
    for (size_t ti = 0; ti < fallbackTasks.size(); ti++)
      runOneCase(fallbackTasks[ti]);
    if (noMoreTasks) break;
    }

    unsigned long long localGpuCounts[8] = {
      localGpuWaves,
      localInspectedTasks,
      localEligibleTasks,
      localDirectFallbackTasks,
      localNonConvergedTasks,
      localControllerFallbackTasks,
      localAdaptiveRefactors,
      localScreenedChecks
    };
    unsigned long long globalGpuCounts[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    unsigned long long globalBridgeCount = 0;
    MPI_Reduce(localGpuCounts, globalGpuCounts, 8, MPI_UNSIGNED_LONG_LONG,
               MPI_SUM, 0, static_cast<MPI_Comm>(world));
    MPI_Reduce(&localBridgeCount, &globalBridgeCount, 1,
               MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0,
               static_cast<MPI_Comm>(world));
    if (world.rank() == 0) {
      const unsigned long long retained =
        globalGpuCounts[2] - globalGpuCounts[4] - globalGpuCounts[5];
      printf("[GPU all-rank summary] ranks=%d waveSize=%d waves=%llu "
             "inspected=%llu eligible=%llu direct_fallback=%llu "
             "nonconverged_fallback=%llu controller_fallback=%llu "
             "retained_gpu=%llu adaptive_refactors=%llu "
             "screen_checks_avoided=%llu islanding_bridges=%llu\n",
             world.size(), batchWaveSize, globalGpuCounts[0],
             globalGpuCounts[1], globalGpuCounts[2], globalGpuCounts[3],
             globalGpuCounts[4], globalGpuCounts[5], retained,
             globalGpuCounts[6], globalGpuCounts[7], globalBridgeCount);
    }
  } else
#endif
  {
    for (;;) {
      bool haveTask = false;
      {
        gridpack::utility::ScopedTimer dispatchTimer(timer, t_ca_task_dispatch);
        haveTask = taskmgr.nextTask(task_comm, &task_id);
      }
      if (!haveTask) break;
      runOneCase(task_id);
    }
  }
  if (_tsp && world.rank()==0) fprintf(stderr,"[TS] contingencyDone %.2fs\n", MPI_Wtime()-_ts0);
  timer->stop(t_ca_processing);
  timer->start(t_ca_finalization);

  gridpack::utility::ScopedTimer flatFinalizeTimer(
      timer, t_ca_flat_finalize);
  // csv_flat / csv_delta: each rank streamed rows to its .part file during
  // the loop. Close, sync, then world rank 0 writes header + concatenates.
  if (outputFormat == "csv_flat") {
    if (bufferFlatOutput) {
      const unsigned long long localBytes =
        static_cast<unsigned long long>(flatMemory.size());
      std::vector<unsigned long long> bytes(world.size(), 0);
      MPI_Allgather(&localBytes, 1, MPI_UNSIGNED_LONG_LONG, &bytes[0], 1,
                    MPI_UNSIGNED_LONG_LONG, static_cast<MPI_Comm>(world));
      const MPI_Offset headerBytes =
        static_cast<MPI_Offset>(std::strlen(flatHeader));
      MPI_Offset offset = headerBytes;
      MPI_Offset totalBytes = headerBytes;
      for (int rank = 0; rank < world.size(); ++rank) {
        if (rank < world.rank()) offset += static_cast<MPI_Offset>(bytes[rank]);
        totalBytes += static_cast<MPI_Offset>(bytes[rank]);
      }
      const std::string flatOutputPath = outputFile + "_flat.csv";
      MPI_File file = MPI_FILE_NULL;
      MPI_File_open(static_cast<MPI_Comm>(world),
                    const_cast<char*>(flatOutputPath.c_str()),
                    MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
      MPI_File_set_size(file, totalBytes);
      if (world.rank() == 0) {
        MPI_File_write_at(file, 0, const_cast<char*>(flatHeader),
                          static_cast<int>(headerBytes), MPI_CHAR,
                          MPI_STATUS_IGNORE);
      }
      MPI_Barrier(static_cast<MPI_Comm>(world));
      size_t written = 0;
      while (written < flatMemory.size()) {
        const size_t remaining = flatMemory.size() - written;
        const int chunk = static_cast<int>(
            std::min(remaining, static_cast<size_t>(INT_MAX)));
        MPI_File_write_at(file, offset + static_cast<MPI_Offset>(written),
                          &flatMemory[written], chunk, MPI_CHAR,
                          MPI_STATUS_IGNORE);
        written += static_cast<size_t>(chunk);
      }
      MPI_File_close(&file);
      flatMemory.clear();
    } else {
      flatPart.close();
    }
  }
  long totalFlatRows = static_cast<long>(flatRowCount);
  if (outputFormat == "csv_flat") {
    world.sum(&totalFlatRows, 1);
  }
  if (outputFormat == "csv_delta") {
    if (deltaPart.is_open()) deltaPart.close();
  }
  if (wantBusSidecar) {
    world.sync();
    if (world.rank() == 0) {
      const size_t BUFSZ = 1 << 20;
      std::vector<char> buf(BUFSZ);

      auto concatParts = [&](const char *suffix, const char *header,
                             const char *tag, const char *outName) {
        std::string outFile = outputFile + outName;
        std::ofstream fout(outFile.c_str(),
                           std::ios::out | std::ios::trunc | std::ios::binary);
        fout << header;
        size_t rows = 0;
        for (int p = 0; p < world.size(); p++) {
          std::ostringstream oss;
          oss << outputFile << suffix << p << ".part";
          std::string part = oss.str();
          std::ifstream fin(part.c_str(), std::ios::in | std::ios::binary);
          if (!fin) continue;
          while (fin) {
            fin.read(&buf[0], BUFSZ);
            std::streamsize got = fin.gcount();
            if (got > 0) {
              fout.write(&buf[0], got);
              for (std::streamsize k = 0; k < got; k++) {
                if (buf[k] == '\n') rows++;
              }
            }
          }
          fin.close();
          std::remove(part.c_str());
        }
        fout.close();
        printf("[%s] wrote %zu rows to %s\n", tag, rows, outFile.c_str());
      };

      if (outputFormat == "csv_flat" && !sharedFlatFile &&
          !bufferFlatOutput) {
        concatParts("_flat.",
                    flatHeader,
                    "csv_flat",
                    "_flat.csv");
      } else if (outputFormat == "csv_flat") {
        printf("[csv_flat] wrote %ld rows to %s_flat.csv\n",
               totalFlatRows, outputFile.c_str());
      }
      if (outputFormat == "csv_delta") {
        concatParts("_delta.",
                    "event_idx,contingency,type,from_bus,to_bus,ckt,"
                    "base_kv_from,base_kv_to,area_from,area_to,base_rate_mva,cont_rate_mva,"
                    "base_p_mw,cont_p_mw,base_q_mvar,cont_q_mvar,"
                    "base_mva,cont_mva,base_loading_pct,cont_loading_pct,"
                    "v_from_base,v_from_cont,v_to_base,v_to_cont,"
                    "ang_from_base,ang_from_cont,ang_to_base,ang_to_cont,"
                    "d_angle_base,d_angle_cont,cont_event_facility\n",
                    "csv_delta",
                    "_delta.csv");
      }

      // Bus metadata sidecar (deduped by bus_id, first writer wins).
      std::string busFile = outputFile + "_buses.csv";
      std::ofstream bout(busFile.c_str(),
                         std::ios::out | std::ios::trunc | std::ios::binary);
      bout << "bus_id,bus_name,base_kv,area,zone,owner,"
              "area_name,zone_name,owner_name\n";
      std::set<int> seen_bus;
      size_t bus_rows = 0;
      for (int p = 0; p < world.size(); p++) {
        std::ostringstream oss;
        oss << outputFile << "_buses." << p << ".part";
        std::string part = oss.str();
        std::ifstream fin(part.c_str());
        if (!fin) continue;
        std::string line;
        while (std::getline(fin, line)) {
          if (line.empty()) continue;
          size_t comma = line.find(',');
          if (comma == std::string::npos) continue;
          int bus_id = std::atoi(line.substr(0, comma).c_str());
          if (seen_bus.insert(bus_id).second) {
            bout << line << "\n";
            bus_rows++;
          }
        }
        fin.close();
        std::remove(part.c_str());
      }
      bout.close();
      printf("[buses] wrote %zu rows to %s\n", bus_rows, busFile.c_str());
    }
  }
  // Aggregate skip count across ranks for diagnostics.
  if (outputFormat == "csv_delta") {
    long localSkip = static_cast<long>(deltaSkipCount);
    long totalSkip = localSkip;
    world.sum(&totalSkip, 1);
    if (world.rank() == 0 && totalSkip > 0) {
      printf("[csv_delta] %ld branch rows had no base-cache match\n",
             totalSkip);
    }
  }
  flatFinalizeTimer.stop();

  // Print statistics from task manager describing the number of tasks performed
  // per processor
  taskmgr.printStats();

  // Sync GA before MPI collectives to flush any pending one-sided operations
  world.sync();

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
    // Convergence is emitted by the universal sidecar block below.
    std::ostringstream localBus, localBranch, localGen;
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
    }

    // Gather all CSV fragments on rank 0 using point-to-point send/recv
    MPI_Comm mpi_comm = static_cast<MPI_Comm>(world);
    std::vector<std::string> allBus(world.size()), allBranch(world.size());
    std::vector<std::string> allGen(world.size());
    allBus[0] = localBus.str();
    allBranch[0] = localBranch.str();
    allGen[0] = localGen.str();
    if (world.rank() == 0) {
      for (int p = 1; p < world.size(); p++) {
        int lens[3];
        MPI_Recv(lens, 3, MPI_INT, p, 0, mpi_comm, MPI_STATUS_IGNORE);
        allBus[p].resize(lens[0]);
        allBranch[p].resize(lens[1]);
        allGen[p].resize(lens[2]);
        if (lens[0] > 0)
          MPI_Recv(&allBus[p][0], lens[0], MPI_CHAR, p, 1, mpi_comm,
                   MPI_STATUS_IGNORE);
        if (lens[1] > 0)
          MPI_Recv(&allBranch[p][0], lens[1], MPI_CHAR, p, 2, mpi_comm,
                   MPI_STATUS_IGNORE);
        if (lens[2] > 0)
          MPI_Recv(&allGen[p][0], lens[2], MPI_CHAR, p, 3, mpi_comm,
                   MPI_STATUS_IGNORE);
      }
    } else {
      std::string sBus = localBus.str(), sBranch = localBranch.str();
      std::string sGen = localGen.str();
      int lens[3] = {(int)sBus.size(), (int)sBranch.size(), (int)sGen.size()};
      MPI_Send(lens, 3, MPI_INT, 0, 0, mpi_comm);
      if (lens[0] > 0)
        MPI_Send(const_cast<char*>(sBus.c_str()), lens[0], MPI_CHAR, 0, 1,
                 mpi_comm);
      if (lens[1] > 0)
        MPI_Send(const_cast<char*>(sBranch.c_str()), lens[1], MPI_CHAR, 0, 2,
                 mpi_comm);
      if (lens[2] > 0)
        MPI_Send(const_cast<char*>(sGen.c_str()), lens[2], MPI_CHAR, 0, 3,
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
    }
  }

  // Universal convergence sidecar: gather, sort by event_idx, write.
  if (emitConv) {
    gridpack::utility::ScopedTimer convergenceTimer(
        timer, t_ca_convergence_output);
    auto formatRow = [](std::ostringstream &os, const ConvRow &r) {
      os << r.event_idx << ","
         << r.name << ","
         << r.type << ","
         << (r.cs.converged ? "true" : "false") << ","
         << r.cs.iterations << ","
         << std::scientific << r.cs.finalTolerance << ","
         << std::fixed
         << r.cs.finalMismatch.maxPBus << ","
         << std::setprecision(4) << r.cs.finalMismatch.maxPMismatch << ","
         << r.cs.finalMismatch.maxQBus << ","
         << std::setprecision(4) << r.cs.finalMismatch.maxQMismatch << ","
         << r.status << "\n";
    };

    std::vector<int> idx;
    std::ostringstream localStream;
    localStream << std::fixed;
    std::vector<int> localOffsets;
    localOffsets.reserve(localConvRows.size() + 1);
    for (size_t i = 0; i < localConvRows.size(); i++) {
      localOffsets.push_back(static_cast<int>(localStream.tellp()));
      formatRow(localStream, localConvRows[i]);
      idx.push_back(localConvRows[i].event_idx);
    }
    localOffsets.push_back(static_cast<int>(localStream.tellp()));
    std::string localStr = localStream.str();

    MPI_Comm conv_comm = static_cast<MPI_Comm>(world);
    if (world.rank() == 0) {
      std::vector<std::pair<int, std::string> > all;
      for (size_t i = 0; i < idx.size(); i++) {
        std::string row = localStr.substr(localOffsets[i],
                                          localOffsets[i+1] - localOffsets[i]);
        all.push_back(std::make_pair(idx[i], row));
      }
      for (int p = 1; p < world.size(); p++) {
        int n = 0;
        MPI_Recv(&n, 1, MPI_INT, p, 10, conv_comm, MPI_STATUS_IGNORE);
        if (n <= 0) continue;
        std::vector<int> remIdx(n), remOff(n + 1);
        MPI_Recv(&remIdx[0], n, MPI_INT, p, 11, conv_comm, MPI_STATUS_IGNORE);
        MPI_Recv(&remOff[0], n + 1, MPI_INT, p, 12, conv_comm,
                 MPI_STATUS_IGNORE);
        int total = remOff[n];
        std::string buf(total, '\0');
        if (total > 0) {
          MPI_Recv(&buf[0], total, MPI_CHAR, p, 13, conv_comm,
                   MPI_STATUS_IGNORE);
        }
        for (int i = 0; i < n; i++) {
          all.push_back(std::make_pair(
              remIdx[i],
              buf.substr(remOff[i], remOff[i+1] - remOff[i])));
        }
      }
      std::sort(all.begin(), all.end());
      std::string convFile = outputFile + "_convergence.csv";
      std::ofstream cout(convFile.c_str(),
                         std::ios::out | std::ios::trunc);
      cout << "event_idx,contingency,type,converged,iterations,"
              "final_tolerance,max_p_bus,max_p_mismatch,max_q_bus,"
              "max_q_mismatch,status_code\n";
      for (size_t i = 0; i < all.size(); i++) cout << all[i].second;
      cout.close();
      printf("[convergence] wrote %zu rows to %s\n",
             all.size(), convFile.c_str());
    } else {
      int n = static_cast<int>(idx.size());
      MPI_Send(&n, 1, MPI_INT, 0, 10, conv_comm);
      if (n > 0) {
        MPI_Send(&idx[0], n, MPI_INT, 0, 11, conv_comm);
        MPI_Send(&localOffsets[0], n + 1, MPI_INT, 0, 12, conv_comm);
        int total = localOffsets[n];
        if (total > 0) {
          MPI_Send(const_cast<char*>(localStr.c_str()), total, MPI_CHAR, 0, 13,
                   conv_comm);
        }
      }
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
  timer->stop(t_ca_finalization);
  timer->stop(t_total);
  if (world.rank() == 0) {
    printf("[profiling] schema=ca-v2 common_phases=6 "
           "legacy_categories=preserved\n");
  }
  if (_tsp && world.rank()==0) fprintf(stderr,"[TS] end %.2fs\n", MPI_Wtime()-_ts0);
  // If all processors executed at least one task, then print out timing
  // statistics (this printout does not work if some processors do not define
  // all timing variables)
  if (events.size()*grp_size >= world.size()) {
    timer->dump();
  }
}
