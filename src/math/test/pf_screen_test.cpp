// -------------------------------------------------------------
/*
 *     Copyright (c) 2013 Battelle Memorial Institute
 *     Licensed under modified BSD License. A copy of this license can be found
 *     in the LICENSE file in the top level directory of this distribution.
 */
// -------------------------------------------------------------

#include <cstdio>
#include <vector>

#include "gridpack/applications/modules/powerflow/pf_screen.hpp"

namespace {

bool expect(const std::vector<int>& actual, const std::vector<int>& expected,
            const char* name)
{
  if (actual == expected) return true;
  std::fprintf(stderr, "%s failed:", name);
  for (size_t i = 0; i < actual.size(); ++i) {
    std::fprintf(stderr, " %d", actual[i]);
  }
  std::fprintf(stderr, "\n");
  return false;
}

} // namespace

int main()
{
  bool ok = true;

  {
    const int from[] = {0, 1, 2, 2};
    const int to[] =   {1, 2, 0, 3};
    gridpack::powerflow::N1ConnectivityScreen screen(
        4, std::vector<int>(from, from + 4), std::vector<int>(to, to + 4));
    const int expected[] = {1, 1, 1, 2};
    ok &= expect(screen.screenAllBranchOutages(),
                 std::vector<int>(expected, expected + 4), "cycle-with-tail");
  }

  {
    const int from[] = {0, 0, 1};
    const int to[] =   {1, 1, 2};
    gridpack::powerflow::N1ConnectivityScreen screen(
        3, std::vector<int>(from, from + 3), std::vector<int>(to, to + 3));
    const int expected[] = {1, 1, 2};
    ok &= expect(screen.screenAllBranchOutages(),
                 std::vector<int>(expected, expected + 3), "parallel-circuits");
  }

  {
    const int nbus = 100000;
    std::vector<int> from(nbus - 1), to(nbus - 1);
    for (int i = 0; i < nbus - 1; ++i) {
      from[i] = i;
      to[i] = i + 1;
    }
    gridpack::powerflow::N1ConnectivityScreen screen(nbus, from, to);
    const std::vector<int> result = screen.screenAllBranchOutages();
    ok &= result.size() == from.size() && result.front() == 2 &&
          result.back() == 2;
  }

  std::printf("pf_screen_test: %s\n", ok ? "PASSED" : "FAILED");
  return ok ? 0 : 1;
}
