/*
 * Copyright (c) 2026 Battelle Memorial Institute
 * Licensed under the modified BSD License.
 */

#include "ca_pattern_scheduler.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <utility>

#include "gridpack/applications/components/pf_matrix/pf_components.hpp"
#include "gridpack/utilities/exception.hpp"

namespace {

struct BranchElement
{
  int firstBus;
  int secondBus;
  std::string circuit;

  bool operator<(const BranchElement& other) const
  {
    if (firstBus != other.firstBus) return firstBus < other.firstBus;
    if (secondBus != other.secondBus) return secondBus < other.secondBus;
    return circuit < other.circuit;
  }
};

BranchElement branchElement(int firstBus, int secondBus,
    const std::string& circuit)
{
  BranchElement result;
  result.firstBus = std::min(firstBus, secondBus);
  result.secondBus = std::max(firstBus, secondBus);
  result.circuit = circuit;
  return result;
}

std::string busListClass(const char *prefix, std::vector<int> buses)
{
  std::sort(buses.begin(), buses.end());
  buses.erase(std::unique(buses.begin(), buses.end()), buses.end());
  std::ostringstream result;
  result << "v1:" << prefix;
  for (std::size_t i = 0; i < buses.size(); ++i) {
    result << ':' << buses[i];
  }
  return result.str();
}

std::string generatorPatternClass(
    const boost::shared_ptr<gridpack::powerflow::PFNetwork>& network,
    const gridpack::powerflow::Contingency& contingency)
{
  std::map<int, std::set<std::string> > outagesByBus;
  for (std::size_t i = 0; i < contingency.p_busid.size(); ++i) {
    outagesByBus[contingency.p_busid[i]].insert(contingency.p_genid[i]);
  }

  std::vector<int> pqBuses;
  std::vector<int> referenceBuses;
  for (std::map<int, std::set<std::string> >::const_iterator outage =
       outagesByBus.begin(); outage != outagesByBus.end(); ++outage) {
    const std::vector<int> localIndices =
      network->getLocalBusIndices(outage->first);
    if (localIndices.empty()) return "v1:no-solve:unknown-generator-bus";

    gridpack::powerflow::PFBus *bus = NULL;
    for (std::size_t i = 0; i < localIndices.size(); ++i) {
      gridpack::powerflow::PFBus *candidate =
        dynamic_cast<gridpack::powerflow::PFBus*>(
            network->getBus(localIndices[i]).get());
      if (candidate != NULL && network->getActiveBus(localIndices[i])) {
        bus = candidate;
        break;
      }
    }
    if (bus == NULL) return "v1:no-solve:unknown-generator-bus";

    int onlineGenerators = 0;
    int onlineOutages = 0;
    std::set<std::string> matched;
    const std::vector<std::string> generators = bus->getGenerators();
    for (std::size_t i = 0; i < generators.size(); ++i) {
      if (outage->second.find(generators[i]) != outage->second.end()) {
        matched.insert(generators[i]);
      }
      if (!bus->getGenStatus(generators[i])) continue;
      ++onlineGenerators;
      if (outage->second.find(generators[i]) != outage->second.end()) {
        ++onlineOutages;
      }
    }
    if (matched.size() != outage->second.size()) {
      return "v1:no-solve:unknown-generator";
    }

    if (onlineGenerators > 0 && onlineGenerators == onlineOutages) {
      if (bus->getReferenceBus()) {
        referenceBuses.push_back(outage->first);
      } else if (bus->isPV()) {
        pqBuses.push_back(outage->first);
      }
    }
  }

  if (!referenceBuses.empty()) {
    // Other generator outages in an N-k event can change which bus becomes
    // the replacement reference. Include every affected generator
    // conservatively because units at one bus can have different capacities.
    std::ostringstream result;
    result << "v1:layout:reference-transfer";
    for (std::map<int, std::set<std::string> >::const_iterator outage =
         outagesByBus.begin(); outage != outagesByBus.end(); ++outage) {
      for (std::set<std::string>::const_iterator generator =
           outage->second.begin(); generator != outage->second.end();
           ++generator) {
        result << ':' << outage->first << '[' << *generator << ']';
      }
    }
    return result.str();
  }
  if (!pqBuses.empty()) return busListClass("layout:pq", pqBuses);
  return "v1:layout:base";
}

} // anonymous namespace

struct gridpack::contingency_analysis::ExpectedJacobianPatternClassifier::Impl
{
  struct Edge
  {
    std::size_t first;
    std::size_t second;
    int activeCircuits;
  };

  struct Circuit
  {
    std::size_t edge;
    bool online;
  };

  explicit Impl(
      const boost::shared_ptr<gridpack::powerflow::PFNetwork>& inputNetwork)
    : network(inputNetwork)
  {
    if (!network || network->communicator().size() != 1) {
      throw gridpack::Exception(
          "expected-pattern scheduling requires size-one task communicators");
    }

    for (int i = 0; i < network->numBuses(); ++i) {
      if (!network->getActiveBus(i)) continue;
      gridpack::powerflow::PFBus *bus =
        dynamic_cast<gridpack::powerflow::PFBus*>(network->getBus(i).get());
      if (bus == NULL || bus->isIsolated()) continue;
      const int originalBus = bus->getOriginalIndex();
      vertexByOriginalBus[originalBus] = originalBuses.size();
      originalBuses.push_back(originalBus);
    }
    baseDegree.assign(originalBuses.size(), 0);

    const std::size_t noEdge = std::numeric_limits<std::size_t>::max();
    for (int i = 0; i < network->numBranches(); ++i) {
      if (!network->getActiveBranch(i)) continue;
      gridpack::powerflow::PFBranch *branch =
        dynamic_cast<gridpack::powerflow::PFBranch*>(
            network->getBranch(i).get());
      if (branch == NULL) continue;
      const int firstBus = branch->getBus1OriginalIndex();
      const int secondBus = branch->getBus2OriginalIndex();
      std::map<int, std::size_t>::const_iterator first =
        vertexByOriginalBus.find(firstBus);
      std::map<int, std::size_t>::const_iterator second =
        vertexByOriginalBus.find(secondBus);
      const std::vector<std::string> lineIds = branch->getLineIDs();

      int activeCircuits = 0;
      for (std::size_t line = 0; line < lineIds.size(); ++line) {
        if (branch->getBranchStatus(lineIds[line])) ++activeCircuits;
      }
      std::size_t edge = noEdge;
      if (activeCircuits > 0 && first != vertexByOriginalBus.end() &&
          second != vertexByOriginalBus.end()) {
        Edge value;
        value.first = first->second;
        value.second = second->second;
        value.activeCircuits = activeCircuits;
        edges.push_back(value);
        edge = edges.size() - 1;
        ++baseDegree[value.first];
        ++baseDegree[value.second];
      }
      for (std::size_t line = 0; line < lineIds.size(); ++line) {
        Circuit value;
        value.edge = edge;
        value.online = branch->getBranchStatus(lineIds[line]);
        circuits[branchElement(firstBus, secondBus, lineIds[line])] = value;
      }
    }
  }

  std::string classifyBranch(
      const gridpack::powerflow::Contingency& contingency) const
  {
    if (contingency.p_from.size() != contingency.p_to.size() ||
        contingency.p_from.size() != contingency.p_ckt.size()) {
      return "v1:no-solve:invalid-branch";
    }

    std::set<BranchElement> outages;
    for (std::size_t i = 0; i < contingency.p_ckt.size(); ++i) {
      outages.insert(branchElement(contingency.p_from[i], contingency.p_to[i],
          contingency.p_ckt[i]));
    }

    std::map<std::size_t, int> onlineOutagesByEdge;
    const std::size_t noEdge = std::numeric_limits<std::size_t>::max();
    for (std::set<BranchElement>::const_iterator outage = outages.begin();
         outage != outages.end(); ++outage) {
      std::map<BranchElement, Circuit>::const_iterator circuit =
        circuits.find(*outage);
      if (circuit == circuits.end()) return "v1:no-solve:unknown-branch";
      if (circuit->second.online && circuit->second.edge != noEdge) {
        ++onlineOutagesByEdge[circuit->second.edge];
      }
    }

    std::set<std::size_t> removedEdges;
    for (std::map<std::size_t, int>::const_iterator edge =
         onlineOutagesByEdge.begin(); edge != onlineOutagesByEdge.end();
         ++edge) {
      if (edge->second >= edges[edge->first].activeCircuits) {
        removedEdges.insert(edge->first);
      }
    }
    if (removedEdges.empty()) return "v1:layout:base";

    if (removedEdges.size() == 1 && outages.size() == 1) {
      const Edge& edge = edges[*removedEdges.begin()];
      std::vector<int> loneBuses;
      if (baseDegree[edge.first] == 1) {
        loneBuses.push_back(originalBuses[edge.first]);
      }
      if (baseDegree[edge.second] == 1) {
        loneBuses.push_back(originalBuses[edge.second]);
      }
      if (!loneBuses.empty()) return busListClass("layout:lone", loneBuses);
      // A non-lone bridge produces no linear solve. Keeping it in the base
      // class is safe and avoids an O(V+E) bridge search for every N-1 event.
      return "v1:layout:base";
    }

    std::vector<int> degree(baseDegree);
    for (std::set<std::size_t>::const_iterator removed =
         removedEdges.begin(); removed != removedEdges.end(); ++removed) {
      --degree[edges[*removed].first];
      --degree[edges[*removed].second];
    }
    std::vector<int> loneBuses;
    std::vector<bool> included(originalBuses.size(), true);
    for (std::size_t vertex = 0; vertex < degree.size(); ++vertex) {
      if (degree[vertex] == 0) {
        included[vertex] = false;
        loneBuses.push_back(originalBuses[vertex]);
      }
    }

    std::vector<std::vector<std::size_t> > adjacency(originalBuses.size());
    for (std::size_t edge = 0; edge < edges.size(); ++edge) {
      if (removedEdges.find(edge) != removedEdges.end()) continue;
      adjacency[edges[edge].first].push_back(edges[edge].second);
      adjacency[edges[edge].second].push_back(edges[edge].first);
    }
    int components = 0;
    std::vector<bool> visited(originalBuses.size(), false);
    for (std::size_t start = 0; start < originalBuses.size(); ++start) {
      if (!included[start] || visited[start]) continue;
      ++components;
      std::queue<std::size_t> pending;
      pending.push(start);
      visited[start] = true;
      while (!pending.empty()) {
        const std::size_t current = pending.front();
        pending.pop();
        for (std::size_t i = 0; i < adjacency[current].size(); ++i) {
          const std::size_t next = adjacency[current][i];
          if (included[next] && !visited[next]) {
            visited[next] = true;
            pending.push(next);
          }
        }
      }
    }
    if (components != 1) return "v1:no-solve:island";
    if (!loneBuses.empty()) return busListClass("layout:lone", loneBuses);
    return "v1:layout:base";
  }

  boost::shared_ptr<gridpack::powerflow::PFNetwork> network;
  std::vector<int> originalBuses;
  std::map<int, std::size_t> vertexByOriginalBus;
  std::vector<Edge> edges;
  std::vector<int> baseDegree;
  std::map<BranchElement, Circuit> circuits;
};

gridpack::contingency_analysis::ExpectedJacobianPatternClassifier::
ExpectedJacobianPatternClassifier(
    const boost::shared_ptr<gridpack::powerflow::PFNetwork>& network)
  : p_impl(new Impl(network))
{}

std::string gridpack::contingency_analysis::
ExpectedJacobianPatternClassifier::classify(
    const gridpack::powerflow::Contingency& contingency) const
{
  if (contingency.p_type == gridpack::powerflow::Branch) {
    return p_impl->classifyBranch(contingency);
  }
  if (contingency.p_type == gridpack::powerflow::Generator) {
    if (contingency.p_busid.size() != contingency.p_genid.size()) {
      return "v1:no-solve:invalid-generator";
    }
    return generatorPatternClass(p_impl->network, contingency);
  }
  return "v1:no-solve:invalid-type";
}

std::string gridpack::contingency_analysis::expectedJacobianPatternClass(
    const boost::shared_ptr<gridpack::powerflow::PFNetwork>& network,
    const gridpack::powerflow::Contingency& contingency)
{
  ExpectedJacobianPatternClassifier classifier(network);
  return classifier.classify(contingency);
}

std::vector<gridpack::contingency_analysis::PatternScheduleEpoch>
gridpack::contingency_analysis::buildPatternSchedule(
    const std::vector<std::string>& expectedPatternClasses,
    std::size_t epochSize)
{
  if (epochSize == 0) {
    throw gridpack::Exception("expected-pattern schedule epoch size is zero");
  }

  struct Group
  {
    std::string key;
    std::vector<int> taskIds;
    std::size_t firstTask;
  };

  std::vector<Group> groups;
  std::map<std::string, std::size_t> groupByKey;
  for (std::size_t task = 0; task < expectedPatternClasses.size(); ++task) {
    const std::string& key = expectedPatternClasses[task];
    std::map<std::string, std::size_t>::iterator existing =
      groupByKey.find(key);
    if (existing == groupByKey.end()) {
      Group group;
      group.key = key;
      group.firstTask = task;
      groups.push_back(group);
      const std::size_t index = groups.size() - 1;
      groupByKey[key] = index;
      existing = groupByKey.find(key);
    }
    groups[existing->second].taskIds.push_back(static_cast<int>(task));
  }

  // Large solvable groups go first so full GPU epochs are not delayed by
  // predicted no-solve work. Ties retain original input order.
  std::stable_sort(groups.begin(), groups.end(),
      [](const Group& first, const Group& second) {
        const bool firstNoSolve = first.key.find("v1:no-solve:") == 0;
        const bool secondNoSolve = second.key.find("v1:no-solve:") == 0;
        if (firstNoSolve != secondNoSolve) return !firstNoSolve;
        if (first.taskIds.size() != second.taskIds.size()) {
          return first.taskIds.size() > second.taskIds.size();
        }
        return first.firstTask < second.firstTask;
      });

  std::vector<PatternScheduleEpoch> result;
  for (std::size_t group = 0; group < groups.size(); ++group) {
    for (std::size_t begin = 0; begin < groups[group].taskIds.size();
         begin += epochSize) {
      PatternScheduleEpoch epoch;
      epoch.expectedPatternClass = groups[group].key;
      const std::size_t end =
        std::min(begin + epochSize, groups[group].taskIds.size());
      epoch.taskIds.assign(groups[group].taskIds.begin() + begin,
                           groups[group].taskIds.begin() + end);
      result.push_back(epoch);
    }
  }
  return result;
}
