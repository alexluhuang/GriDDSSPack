// Emacs Mode Line: -*- Mode:c++;-*-
// -------------------------------------------------------------
/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
// -------------------------------------------------------------
/**
 * @file   pf_screen.hpp
 * @brief  Phase-3 batched connectivity pre-pass (Union-Set) for N-1 screening.
 *
 * For each branch outage the network is tested for connectivity with a
 * Union-Find over all OTHER branches; a case that splits the grid into >1
 * component (or isolates a bus) is flagged so the expensive AC solve can be
 * skipped or routed to the existing island-handling logic.  This is Chen's
 * Union-Set idea (one independent test per branch outage, O(K + K*alpha(N)));
 * it replaces the per-contingency CPU getIslandCount()/hasLoneBus() with a
 * single up-front pass over all outages.
 *
 * The core is a self-contained, dependency-free struct so it is trivially
 * unit-tested against a reference and against GridPACK's own island detection.
 * The tests are independent, so the whole pre-pass parallelizes across the
 * Grace cores (OpenMP if available) or, unchanged in structure, one CUDA block
 * per outage (root[] in shared memory) as in the reference architecture.
 */
// -------------------------------------------------------------

#ifndef _pf_screen_hpp_
#define _pf_screen_hpp_

#include <vector>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gridpack {
namespace powerflow {

// -------------------------------------------------------------
//  class N1ConnectivityScreen
// -------------------------------------------------------------
/// Union-Set N-1 branch-outage connectivity screen over a bus/branch graph.
/**
 * Buses are 0..nbus-1 (compact local indexing); each branch is an unordered
 * pair (from,to).  A single "in-service" mask lets already-out branches (from a
 * prior N-k base) be excluded from every test.
 */
class N1ConnectivityScreen
{
public:

  N1ConnectivityScreen(int nbus,
                       const std::vector<int>& from,
                       const std::vector<int>& to)
    : p_nbus(nbus), p_from(from), p_to(to)
  {}

  /// Number of connected components with branch @c outage removed.
  /**
   * Isolated buses count as their own components, so a result of 1 means the
   * post-outage network is fully connected; >1 means islanding.
   *
   * @param outage index of the branch removed for this test (-1 = base case)
   * @return number of connected components
   */
  int componentsWithout(int outage) const
  {
    std::vector<int> root(p_nbus);
    std::iota(root.begin(), root.end(), 0);
    const int K = static_cast<int>(p_from.size());
    for (int e = 0; e < K; ++e) {
      if (e == outage) continue;
      p_merge(root, p_from[e], p_to[e]);
    }
    int comps = 0;
    for (int v = 0; v < p_nbus; ++v) {
      if (p_find(root, v) == v) ++comps;
    }
    return comps;
  }

  /// True iff removing branch @c outage islands the grid (>1 component).
  bool islandsWithout(int outage) const
  {
    return componentsWithout(outage) > 1;
  }

  /// Screen every branch outage; result[e] = component count with e removed.
  /**
   * Independent per outage, so this is embarrassingly parallel; parallelized
   * across the Grace cores with OpenMP when available.
   */
  std::vector<int> screenAllBranchOutages(void) const
  {
    const int K = static_cast<int>(p_from.size());
    std::vector<int> comps(K, 1);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 64)
#endif
    for (int e = 0; e < K; ++e) {
      comps[e] = componentsWithout(e);
    }
    return comps;
  }

  int numBuses(void) const    { return p_nbus; }
  int numBranches(void) const { return static_cast<int>(p_from.size()); }

private:

  int p_nbus;
  std::vector<int> p_from, p_to;

  static int p_find(std::vector<int>& root, int x)
  {
    while (root[x] != x) {
      root[x] = root[root[x]];   // path halving
      x = root[x];
    }
    return x;
  }

  static void p_merge(std::vector<int>& root, int a, int b)
  {
    int ra = p_find(root, a), rb = p_find(root, b);
    if (ra != rb) root[ra] = rb;
  }
};

} // namespace powerflow
} // namespace gridpack

#endif
