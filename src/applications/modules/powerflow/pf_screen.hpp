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
#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>

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
  {
    if (p_nbus < 0) {
      throw std::invalid_argument(
          "N1ConnectivityScreen: bus count must be nonnegative");
    }
    if (p_from.size() != p_to.size()) {
      throw std::invalid_argument(
          "N1ConnectivityScreen: branch endpoint arrays have different sizes");
    }
    if (p_from.size() >
        static_cast<size_t>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument(
          "N1ConnectivityScreen: branch count exceeds supported index range");
    }
    for (size_t e = 0; e < p_from.size(); ++e) {
      if (p_from[e] < 0 || p_from[e] >= p_nbus ||
          p_to[e] < 0 || p_to[e] >= p_nbus) {
        throw std::out_of_range(
            "N1ConnectivityScreen: branch endpoint is outside the bus range");
      }
    }
  }

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
    const int K = static_cast<int>(p_from.size());
    if (outage < -1 || outage >= K) {
      throw std::out_of_range(
          "N1ConnectivityScreen: outage index is outside the branch range");
    }
    std::vector<int> root(p_nbus);
    std::iota(root.begin(), root.end(), 0);
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
   * A single Tarjan bridge traversal classifies every N-1 outage in O(V+E).
   * Edge IDs, rather than parent vertices, are used when walking the graph so
   * parallel circuits are handled correctly: neither parallel edge is a bridge.
   */
  std::vector<int> screenAllBranchOutages(void) const
  {
    const int K = static_cast<int>(p_from.size());
    std::vector<int> comps(K, 0);
    std::vector<std::vector<Edge> > graph(p_nbus);
    for (int e = 0; e < K; ++e) {
      const int u = p_from[e], v = p_to[e];
      if (u == v) continue;
      graph[u].push_back(Edge(v, e));
      graph[v].push_back(Edge(u, e));
    }

    std::vector<int> discovery(p_nbus, -1), low(p_nbus, -1);
    std::vector<int> parentVertex(p_nbus, -1), parentEdge(p_nbus, -1);
    std::vector<char> bridge(K, 0);
    int clock = 0;
    int baseComponents = 0;
    std::vector<Frame> stack;
    stack.reserve(p_nbus);
    for (int root = 0; root < p_nbus; ++root) {
      if (discovery[root] >= 0) continue;
      ++baseComponents;
      discovery[root] = low[root] = clock++;
      stack.push_back(Frame(root));
      while (!stack.empty()) {
        Frame& frame = stack.back();
        const int u = frame.vertex;
        if (frame.nextEdge < graph[u].size()) {
          const Edge edge = graph[u][frame.nextEdge++];
          if (edge.id == parentEdge[u]) continue;
          if (discovery[edge.to] < 0) {
            parentVertex[edge.to] = u;
            parentEdge[edge.to] = edge.id;
            discovery[edge.to] = low[edge.to] = clock++;
            stack.push_back(Frame(edge.to));
          } else {
            low[u] = std::min(low[u], discovery[edge.to]);
          }
          continue;
        }
        stack.pop_back();
        const int parent = parentVertex[u];
        if (parent >= 0) {
          low[parent] = std::min(low[parent], low[u]);
          if (low[u] > discovery[parent]) bridge[parentEdge[u]] = 1;
        }
      }
    }
    for (int e = 0; e < K; ++e) {
      comps[e] = baseComponents + (bridge[e] ? 1 : 0);
    }
    return comps;
  }

  int numBuses(void) const    { return p_nbus; }
  int numBranches(void) const { return static_cast<int>(p_from.size()); }

private:

  struct Edge {
    int to;
    int id;
    Edge(int to_, int id_) : to(to_), id(id_) {}
  };

  struct Frame {
    int vertex;
    size_t nextEdge;
    explicit Frame(int vertex_) : vertex(vertex_), nextEdge(0) {}
  };

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
