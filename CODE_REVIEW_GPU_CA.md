# Adversarial code review — GPU contingency-analysis work

**Scope:** commits `10e5cf26` → `570bedb8` (range `cff541f4..HEAD`), i.e. everything from
"GPU build and runtime prerequisites, a cuDSS direct-solver backend, batched contingency
engine" forward. `cff541f4` and earlier is treated as base GridPACK.

**Diff:** 45 files, +97,915 / −59 (≈91k of the insertions are the checked-in comparison report).

**Method:** every claim below was checked against the source at HEAD. The output-path finding
includes an empirical `strace` of libstdc++ append behaviour on this machine. Findings marked
**[codex]** overlap the external review in `Research GridPACK speedups.md`; a section at the end
records where that report is stale or wrong about HEAD.

Findings are ranked by severity. Each gives the location, the failure mode, why it matters, and a
concrete fix.

---

## Table of contents

- [Tier 1 — Correctness and verification integrity](#tier-1--correctness-and-verification-integrity)
- [Tier 2 — Reporting semantics](#tier-2--reporting-semantics)
- [Tier 3 — Performance defects](#tier-3--performance-defects)
- [Tier 4 — Dead code and misleading documentation](#tier-4--dead-code-and-misleading-documentation)
- [Where the external review is wrong about HEAD](#where-the-external-review-is-wrong-about-head)
- [Suggested order of work](#suggested-order-of-work)

---

## Tier 1 — Correctness and verification integrity

### 1. Both correctness oracles for the GA-free assembler are dead code

**This is the most serious finding.**

`PFBatchNR::solveWave` calls exactly five assembler methods
([pf_batch_ca.hpp:179-246](src/applications/modules/powerflow/pf_batch_ca.hpp#L179-L246)):
`assembleLive`, `updateLive`, `updateLiveRhs`, `finishLive`, plus the pattern getters. It **never**
calls `assemble()` or `update()`.

But both validation hooks live only inside `assemble()` / `update()`:

```cpp
// pf_batch_ca_assembler.hpp:256-259
if (p_useFast) {
  static bool val = (std::getenv("GRIDPACK_BATCH_VALIDATE") != NULL);
  if (val && p_nAsm <= 1) p_validateAssemble(k);
  return p_assembleFast(k, jac, rhs);
}
```

Consequences:

- `GRIDPACK_BATCH_VALIDATE=1` is **inert** on the shipped path. The "validated byte-identical
  (ΔJ = ΔRHS = 0) against the GA mapper" claim in
  [GPU_CA_IMPLEMENTATION.md:117](GPU_CA_IMPLEMENTATION.md#L117) **cannot be reproduced** with the
  code as shipped.
- `GRIDPACK_BATCH_NOFAST=1` is **inert** too. `p_useFast` gates only `assemble` / `update` (and
  `p_buildScatterMap`). `assembleLive` / `updateLive` / `updateLiveRhs` use `p_fastJac` / `p_fastRhs`
  unconditionally. Setting `NOFAST=1` does not fall back to the GA mapper — it skips building the
  scatter map, leaving `p_diag` / `p_off` empty, which produces an **all-zero Jacobian**: a silent
  wrong answer, not a fallback.

Why this matters: the GA-free assembler is the most invasive change in the range, and the
`baseStatus` bug (fixed in `81d9015a`) proves this path *does* harbour silent state corruption that
only appears from roughly case 3 of a wave onward — exactly the regime a `p_nAsm <= 1` gate cannot
see.

**Fix**

```cpp
// in assembleLive(), after p_fastJac(jac):
#ifndef NDEBUG
  static const bool val = (std::getenv("GRIDPACK_BATCH_VALIDATE") != NULL);
  if (val) p_validateAssembleAt(k, jac, rhs);   // every case, every iteration
#endif
```

Add the same to `updateLive` / `updateLiveRhs` and drop the `p_nAsm <= 1` gate. Then either make
`p_useFast == false` actually route `assembleLive` through the GA mapper, or delete `p_useFast`,
`assemble()`, `update()` and the `NOFAST` env var so nobody trusts a switch that does nothing.

---

### 2. Shared-append output depends on unguaranteed atomicity **[codex, extended]**

`sharedFlatFile=true` is the **default**
([ca_driver.cpp:431](src/applications/contingency_analysis/ca_driver.cpp#L431)): all ranks open the
same `_flat.csv` in append mode and each writes ~780 KB blocks
([ca_driver.cpp:692](src/applications/contingency_analysis/ca_driver.cpp#L692)).

Measured syscall behaviour for a 780 KB `ofstream << std::string` in append mode on this machine
(aarch64, glibc/libstdc++):

```
openat(AT_FDCWD, "wtest.txt", O_WRONLY|O_CREAT|O_APPEND, 0666) = 3
writev(3, [{NULL,0}, {"xxx...", 780000}], 2) = 780000
writev(3, [{"",0},   {"xxx...", 780000}], 2) = 780000
```

One `writev()` per block, full-count return, `O_APPEND` set. On Linux with a local filesystem that
append is atomic — which is why the full 79 M-row and 2.0 B-row field scans found zero malformed
rows. **The design works today.** It works, however, by accident of three properties that nothing
in the codebase guarantees:

1. **Short writes are unhandled at the atomicity level.** libstdc++ loops on a short `writev()`;
   each retry is a *separate* syscall, so a concurrent rank's block can land between the halves and
   tear a row. Nothing prevents or detects this.
2. **`O_APPEND` atomicity does not hold on NFS**, and is weak on some parallel filesystems. Running
   this on a cluster scratch mount yields silent CSV corruption with no error and no warning.
3. **Block size scales with the network.** Texas7k ≈ 780 KB/block; `training.raw` ≈ 2.7 MB/block.
   Larger networks approach single-write size limits and make short writes more likely.

Separately: with `sharedFlatFile=true` **and** `overlapIO=true`, cross-rank interleaving is
nondeterministic, so the file is *not* byte-identical to the synchronous run. The byte-identical
verification in [GPU_CA_IMPLEMENTATION.md:423](GPU_CA_IMPLEMENTATION.md#L423) predates `d939666f`
and was performed against per-rank `.part` files.

The already-implemented `bufferFlatOutput` MPI-IO path is the *correct* primitive but is unusable at
scale: it holds each rank's entire output in RAM (5.7 GB/rank × 20 = 114 GB on `training.raw`, on a
128 GB box).

**Fix**

Add a third mode: incremental collective I/O. Keep a per-rank memory cap (e.g. 256 MB); when the
buffer crosses it, atomically claim an extent and write it at that offset:

```cpp
// reserve a byte range with an atomic fetch-and-add on a shared offset
MPI_Fetch_and_op(&nbytes, &myOffset, MPI_OFFSET, 0, 0, MPI_SUM, offsetWin);
MPI_File_write_at(file, headerBytes + myOffset, buf.data(), n, MPI_CHAR, MPI_STATUS_IGNORE);
```

This is O(1) memory, requires no `O_APPEND` guarantee, and works on any MPI-IO-capable filesystem.
Failing that, at minimum: detect non-local filesystems (`statfs` → `NFS_SUPER_MAGIC`,
`LL_SUPER_MAGIC`, GPFS) and refuse `sharedFlatFile=true`, and add `MPI_File_open` / `ofstream`
error checks — there are currently none anywhere in the flat path.

---

### 3. `waveSize` is parsed unsafely, has no bound, and its documented meanings contradict each other

```cpp
// ca_driver.cpp:1668-1671
if (gcur && gcur->get("waveSize", &t)) {
  util.toLower(t);
  if (t != "auto") batchWaveSize = std::max(0, std::atoi(t.c_str()));
}
...
if (useBatched && batchWaveSize == 0) batchWaveSize = 8;   // :1728
```

Four problems:

- **`std::atoi` on garbage returns 0** → silently becomes 8. `<waveSize>eight</waveSize>` "works"
  and nobody is told.
- **`"auto"` is documented as cuDSS memory prediction** in
  [input_14.xml](src/applications/data_sets/input/ca/input_14.xml)
  (`<!-- 'auto' => cuDSS memory-prediction -->`). There is no memory prediction; `auto` means
  literally 8.
- **`0` means opposite things in two places.**
  [pf_batch_ca.hpp:63](src/applications/modules/powerflow/pf_batch_ca.hpp#L63) documents
  `waveSize: 0 => all at once`; the driver maps 0 → 8.
- **No upper bound, and memory is O(W × nbus).** `prepare()` ends with
  `p_caseV.assign(W, p_startV); p_caseA.assign(W, p_startA);`
  ([:239-241](src/applications/modules/powerflow/pf_batch_ca_assembler.hpp#L239-L241)) —
  `2 × W × nbus × 8` bytes. `<waveSize>100000</waveSize>` on Texas7k allocates 10.7 GB per rank and
  reproduces the measured 113.70 s straggler regression.

**Fix**

Parse with `strtol` plus validation, reject unparseable values loudly, cap at something defensible
(e.g. `min(user, 256)`) with a printed notice, and either implement `auto` via
`cudssDataGet(CUDSS_DATA_MEMORY_ESTIMATES)` or rename it `default` and correct the comments.

---

### 4. The screened fast path skips three of four eligibility checks — safe today, for undocumented reasons

```cpp
// pf_batch_ca_assembler.hpp:208-229
if (brs.size() == 1 && evt.p_to.size() == 1) {
  ... if (pos != p_lineIsBridge.end()) { screened = true; eligible = !pos->second; }
}
if (!screened) {
  p_restoreStart();
  bool found = p_app.setContingency(evt);
  if (found && p_app.getIslandCount() <= 1 && !p_app.hasLoneBus() &&
      p_structureSignature() == p_baseSig) eligible = true;
  p_app.unSetContingency(evt);
}
```

A screened case is declared eligible on the bridge test **alone**. The probe it replaces checked
four things. Whether the other three are implied:

| Skipped check | Implied by "not a bridge"? | Why |
|---|---|---|
| `hasLoneBus()` | **Yes** | A bus whose only in-service branch is removed goes to degree 0, which means that branch separated it — so it *was* a bridge. |
| `checkAndTransferSlack()` | **Yes** | [pf_factory_module.cpp](src/applications/modules/powerflow/pf_factory_module.cpp) transfers slack only when `!bus->hasOnlineGenerator()`. A branch outage never changes generator status, so it returns early with `slackTransferred=false`. |
| `p_structureSignature() == p_baseSig` | **Yes** | `matrixDiagSize` depends on `isIsolated()`, `getReferenceBus()`, `p_isPV`, `p_isIREG_PV`. None is mutated by `setContingency` for a branch event (`p_isIREG_PV` is written only at load time). |

So it is correct — but all three conclusions are non-obvious, non-local invariants with **zero**
comment, assertion or test. If anyone later makes `checkAndTransferSlack` topology-aware, or runs
`chkQlim` inside the inner Newton, the screened path silently starts batching structurally
incompatible cases and the shared symbolic analysis becomes invalid, producing wrong numbers rather
than a crash.

**Fix**

Document the three implications at
[:208](src/applications/modules/powerflow/pf_batch_ca_assembler.hpp#L208), and add a debug-only
assertion that revalidates a sampled screened case against the full probe:

```cpp
#ifndef NDEBUG
if (screened && eligible && (tid % 97 == 0)) {
  p_restoreStart(); p_app.setContingency(evt);
  assert(p_app.getIslandCount() <= 1 && !p_app.hasLoneBus() &&
         p_structureSignature() == p_baseSig);
  p_app.unSetContingency(evt);
}
#endif
```

---

### 5. `p_findSlot` returning −1 silently drops Jacobian entries

```cpp
// pf_batch_ca_assembler.hpp:592-597
int p_findSlot(int row, int col) const {
  for (int k = p_rowptr[row]; k < p_rowptr[row+1]; k++)
    if (p_colind[k] == col) return k;
  return -1;
}
// :722, :728, :731
if (r.slot[t] >= 0) jac[r.slot[t]] += vals[t];
```

Every scatter is guarded by `>= 0`, so a missing slot is discarded with no error, no counter and no
log. This should never happen (the pattern is extracted from the same base topology the map is built
from), but if it ever does the failure mode is uniquely hard to spot: a *chord* iteration with a
slightly wrong Jacobian **still converges to the correct root** (the fixed point is `F(x)=0`
regardless of `J`), so the only symptom is a mildly elevated fallback rate — indistinguishable from
ordinary numerical variation. The run-to-run retained-GPU wobble
(6,388 / 6,411 / 6,458 / 6,462 / 6,497) would hide it completely.

**Fix**

Count −1 slots during `p_buildScatterMap` and throw if nonzero:

```cpp
if (p_missingSlots) throw gridpack::Exception(
  "pf_batch_ca: scatter map has " + std::to_string(p_missingSlots) +
  " component entries outside the base CSR pattern");
```

---

### 6. `p_fastRhs` leaves skipped rows stale (latent)

```cpp
// pf_batch_ca_assembler.hpp:695-709
double p_fastRhs(double* rhs) const {
  double vals[2]; double inf = 0.0;
  for (size_t i = 0; i < p_diag.size(); i++) {
    const DiagRec& r = p_diag[i];
    if (!r.bus->vectorValues(vals)) continue;   // <-- rhs[r.row0..] never written
    ...
  }
  return inf;
}
```

`p_fastJac` correctly `std::fill`s to zero first
([:716](src/applications/modules/powerflow/pf_batch_ca_assembler.hpp#L716)); `p_fastRhs` does not. A
skipped bus leaves the previous case's or previous iteration's value in `rhs` (the buffer is reused
across the whole wave in `solveWave`), **and** its rows do not contribute to the inf-norm — so a
case could be declared converged with unsatisfied equations.

Currently unreachable: both predicates are identical — `matrixDiagSize`
([pf_components.cpp:175-203](src/applications/components/pf_matrix/pf_components.cpp#L175-L203)) and
`vectorSize` ([:247-267](src/applications/components/pf_matrix/pf_components.cpp#L247-L267)) both
return false for isolated and reference buses, 1 for `p_isPV && !p_isIREG_PV`, else 2. So `p_diag`
membership guarantees `vectorValues` succeeds. But that is a coincidence of two separate functions
in a file this range does not own.

**Fix**

One line — `if (rhs) std::fill(rhs, rhs + p_n, 0.0);` at the top, and make the skip branch an
assertion rather than a `continue`.

---

## Tier 2 — Reporting semantics

### 7. `viol` is computed against rate A while `loading_percent` uses rate C **[codex]**

```cpp
// ca_driver.cpp:992-994
const double loading = rateSelected > 0.0 ? 100.0 * flowMva / rateSelected : 0.0;
const int viol = rateA > 0.0 && flowMva / rateA > 1.0 ? 1 : 0;
```

With the default `contingencyRating=C`, a row can read `loading_percent = 55.0` and `viol = 1`. The
direct formatter faithfully reproduces the legacy `flow_str` semantics
([pf_components.cpp:3643-3651](src/applications/components/pf_matrix/pf_components.cpp#L3643-L3651)
also uses `p_rateA`), so this is a *preserved* pre-existing inconsistency rather than a new one — but
it defeats the purpose of `contingencyRating`, and `PFFactoryModule::checkLineOverloadViolations` can
only use rate A or B, never C, so `hasBranchViolation` in the JSON/CSV summary disagrees too.

The harness's **3,361 violation-flag mismatches (0.0045 %)** are precisely this class: borderline
rate-A crossings amplified by the ~1e-4 flow differences the chord path produces.

**Fix**

Compute `viol` against `rateSelected` and add an explicit `viol_rating` column, or add
`<violationRating>A|B|C|selected</violationRating>` defaulting to `A` for backward compatibility.
Either way, document it — a silent A/C split within one row is worse than either choice.

---

### 8. Batched convergence rows zero out worst-bus diagnostics, which corrupts the accuracy report

```cpp
// ca_driver.cpp:2230-2233
cs.finalMismatch.maxPBus = 0;  cs.finalMismatch.maxPMismatch = 0.0;
cs.finalMismatch.maxQBus = 0;  cs.finalMismatch.maxQMismatch = 0.0;
```

Honest and documented in the code. But `numerical_differences.csv` then reports `max_p_mismatch`
MAPD = 79.0 % and `max_q_mismatch` = 71.6 %, with GPU means an order of magnitude smaller than CPU
(6.7e-5 vs 6.8e-4) and 4,063 / 3,757 zero-reference exclusions. A reader concludes "the GPU is more
accurate." It is not — roughly 72 % of the rows are zeros written by that block.

**Fix**

Emit an empty field (not `0`) for batched cases so the harness classifies them as missing rather
than as a value, or have `finishLive` return the argmax over `p_fastRhs` — it already scans every row
to compute the inf-norm, so capturing the index and the P/Q split is free.

---

### 9. `snprintf` clamp can emit a row without a newline; contingency names silently truncate at 23 chars

```cpp
// ca_driver.cpp:1000-1004
if (ln > 0) frow.append(lbuf, ln < (int)sizeof(lbuf) ? ln : (int)sizeof(lbuf) - 1);
```

If `snprintf` ever truncates (`ln >= 256`), this appends 255 bytes with the `\n` chopped off — a
malformed row instead of a hard failure. And `char ct_name[24]` with `strncpy(..., 23)`
([:926](src/applications/contingency_analysis/ca_driver.cpp#L926)) truncates longer names, so two
distinct multi-element contingencies sharing a 23-character prefix become indistinguishable in the
`contingency` column. (`event_idx` still disambiguates, which is why the harness reported
`contingency_name_mismatches: 0`.)

**Fix**

`if (ln <= 0 || ln >= (int)sizeof(lbuf)) throw gridpack::Exception("csv_flat row overflow");` and
size `ct_name` from the actual maximum name length, or emit the full `std::string`.

---

### 10. cuDSS pattern cache keys on `(n, nnz)` only **[codex]**

```cpp
// cudss_linear_solver_implementation.hpp:202
if (p_analyzed && n == p_nCached && nnz == p_nnzCached) return;  // reuse analysis
```

Two different sparsity patterns with the same dimension and nonzero count reuse a stale symbolic
analysis, producing garbage. Safe in CA (a fresh `LinearSolver` per contingency resets
`p_analyzed`), but this is a public math-layer class.

**Fix**

Cache a cheap 64-bit FNV hash of `rowptr` + `colind` alongside `n` / `nnz` and compare it.

---

## Tier 3 — Performance defects

### 11. The assembler — and the whole Tarjan pass — is rebuilt once per wave

`batchAsm.reset(new GridpackBatchAssembler(...))` at
[ca_driver.cpp:2192](src/applications/contingency_analysis/ca_driver.cpp#L2192) runs **1,112 times**
on Texas7k. Each construction performs all of:

- a full `setYBus()` + `setSBus()`, a fresh `BusVectorMap` and `FullMatrixMap`, and a full GA
  `mapToRealVector` / `mapToRealMatrix`
  ([:128-137](src/applications/modules/powerflow/pf_batch_ca_assembler.hpp#L128-L137));
- `p_buildScatterMap()` — roughly (6,717 × 4 + 8,646 × 8) `p_findSlot` linear scans;
- `p_buildConnectivityScreen()` — a whole-network Tarjan pass **plus ~8,646 inserts into a
  `std::map<std::pair<int,std::string>, bool>`**, i.e. ~9.6 M red-black-tree inserts with string keys
  across the run;
- snapshotting `startV` / `startA` and the base structure signature over all buses.

All of it is wave-invariant. This is very likely why `Powerflow: Create Mappers` is still 7.69 s in
the GPU run.

**Caveat that makes this non-trivial:** the per-wave rebuild is *accidentally load-bearing*. The
controller-fallback loop mutates `p_isPV` via `checkQlimViolations()` and restores via
`clearQlimViolations()`; if that restore is ever imperfect, the stale `p_diag` / `p_baseSig` would be
wrong — but the next wave rebuilds them, so the leak self-heals. A previous attempt to hoist the
constructor measured no gain (92.74 s) and was reverted; that result is consistent with the cost
being real but masked by MPI load-balance noise at wave 8. Hoisting it without addressing the
state-leak invariant would be unsafe.

**Fix, in order**

1. Hoist `p_lineIsBridge` and the Tarjan pass into a shared object built once per rank and passed by
   const reference; replace the `map<pair<int,string>>` with a flat `vector<char>` indexed by a
   precomputed dense circuit id.
2. Hoist the scatter map and base pattern similarly.
3. Keep the per-wave *state* reset (`p_caseV` / `p_caseA` / `p_baseSig` re-snapshot) so the leak
   protection survives.
4. Re-measure with `GRIDPACK_BATCH_PROFILE=1` at waves 8, 16 and 32 once the fixed cost is gone —
   the optimal wave size will move.

---

### 12. `applyCaseForOutput` does a full-network `setYBus` plus GA vector map, twice per retained case

```cpp
// pf_batch_ca_assembler.hpp:458-473
p_restoreCase(k); p_toggleBranches(k, false);
p_factory->setYBus();          // FULL network
p_factory->setMode(YBus); p_factory->setSBus();
p_factory->setMode(RHS); p_vMap->mapToRealVector(p_PQ);   // FULL GA map
```

Called once per case in the controller loop
([ca_driver.cpp:2277](src/applications/contingency_analysis/ca_driver.cpp#L2277)) and again in
`runOneCase` ([:1848](src/applications/contingency_analysis/ca_driver.cpp#L1848)) — roughly **13,500
full YBus rebuilds plus GA maps** on Texas7k, in a design that went to considerable lengths to avoid
exactly this.

Also: `clearCaseForOutput` only calls `p_toggleBranches(k, true)` and does **not** refresh YBus, so
it leaves the cached admittances describing the outaged topology. It self-heals because the next
`applyCaseForOutput` does a full `setYBus`, and because `PFAppModule::solve()` does its own — but
that is an undocumented dependency, and it breaks the moment someone removes the redundant full
`setYBus`.

**Fix**

Replace with `p_localSetYBus(k, false)` + `p_fastRhs(NULL)` (the assembler already holds the
converged iterate and has a fast RHS pass that refreshes `Pinj` / `Qinj`); cache the qlim decision
computed at `finishLive` so the controller loop needs no reload at all; and make
`clearCaseForOutput` symmetric with `p_localSetYBus(k, true)`.

---

### 13. The chord cap ignores `<maxIteration>`, and `refactorEvery` is dead in the wave engine **[codex]**

`PFBatchNR`'s `chordCap` defaults to 25
([pf_batch_ca.hpp:156](src/applications/modules/powerflow/pf_batch_ca.hpp#L156)) and the driver
**never passes it** — [ca_driver.cpp:2214-2215](src/applications/contingency_analysis/ca_driver.cpp#L2214-L2215)
passes only `refactorEvery` and `constantFactor`. So `<maxIteration>50</maxIteration>` has no effect
on the chord path, and `p_refactorEvery` is stored and never read: the driver's
`batchRefactorEvery = batchMaxIter` assignment at
[:1704](src/applications/contingency_analysis/ca_driver.cpp#L1704) is a no-op.

A user raising `maxIteration` to fix convergence gets more fallbacks than expected, with no
indication why.

**Fix**

Pass `batchMaxIter` as `chordCap` (or add `<chordCap>`), and implement adaptive refactorization —
track the residual ratio `‖F(x_{k+1})‖ / ‖F(x_k)‖` and refactorize when it exceeds ~0.5 instead of
falling back to CPU. That directly attacks the 279 non-converged fallbacks, each of which pays full
GPU setup *and then* a full CPU solve.

---

## Tier 4 — Dead code and misleading documentation

Not runtime bugs, but several actively assert the **rejected** design, which is worse than no
comment at all.

| # | Location | Problem |
|---|---|---|
| 14 | [pf_batch_ca.hpp:57-71](src/applications/modules/powerflow/pf_batch_ca.hpp#L57-L71) | **`PFBatchCAConfig` is entirely dead.** Zero references outside its own definition — the driver reads XML into local variables. `fastDecoupled`, `screen`, `waveSize`, `maxIterations`, `tolerance`, `enabled`, `batched`, `warmStart` all unused. The external review's `fastDecoupled` finding generalises to the whole struct. |
| 15 | [cudss_batched_solver.hpp:169-179](src/math/cudss/cudss_batched_solver.hpp#L169-L179) | Comment states "the base Jacobian is factorized **ONCE** here and every case's chord iterations reuse it… Only ONE factorization is done per wave." The shipped `solveWave` factorizes **per case**. This describes the design that was tried and rejected for 66 % non-convergence. |
| 16 | [pf_batch_ca_assembler.hpp:389-397](src/applications/modules/powerflow/pf_batch_ca_assembler.hpp#L389-L397) | Same false claim: "The base Jacobian is assembled+factorized ONCE by the engine; each case then iterates with the base factors." |
| 17 | [cudss_batched_solver.hpp:122-141](src/math/cudss/cudss_batched_solver.hpp#L122-L141) | `reset()` carries a detailed "cuDSS factor drift after a few dozen refactorizations" theory. That hypothesis was **wrong** — the real cause was the `baseStatus` bug. `reset()`, `factorize()`, `setValues()`, `setAllValues()`, `solve()`, `p_analyzed`, `p_W`, `p_h_rowptr`, `p_h_colind` are all dead on the production path. Delete them, or the next reader will "fix" a nonexistent problem. |
| 18 | [pf_batch_ca.hpp:30-32](src/applications/modules/powerflow/pf_batch_ca.hpp#L30-L32) | "Wiring that concrete assembler onto PFFactoryModule/FullMatrixMap is the remaining integration step." It has been wired since `31a98d56`. |
| 19 | [pf_screen.hpp:11-26](src/applications/modules/powerflow/pf_screen.hpp#L11-L26) | Header says "Union-Set", "O(K + K·α(N))", and "parallelizes across the Grace cores (OpenMP if available)". The implementation is Tarjan, and the OpenMP pragma was deleted in `81d9015a` — `#include <omp.h>` is now dead. This stale header is what led the external review to report a nonexistent O(E²) bug. |
| 20 | [pf_screen.hpp:70-84](src/applications/modules/powerflow/pf_screen.hpp#L70-L84) | `componentsWithout()` indexes `root[p_from[e]]` with **no bounds check**, while its sibling `screenAllBranchOutages()` does check (`if (u < 0 \|\| u >= p_nbus ...) continue`). Out-of-range endpoints are an out-of-bounds vector write in one and skipped in the other; the two also disagree on which edges count toward `baseComponents`, and `p_buildConnectivityScreen` compares results from both. |
| 21 | [GPU_CA_IMPLEMENTATION.md:18](GPU_CA_IMPLEMENTATION.md#L18), [:222-238](GPU_CA_IMPLEMENTATION.md#L222-L238) | "**Identical outputs** — the three CSVs keep their exact schema" and "ALL PARITY CHECKS PASSED / byte-identical". Not true at HEAD: the flat file has 4,268,160 more rows than the pre-change formatter, convergence metadata differs by construction (chord vs exact Newton), and the strict IEEE-14 parity script now fails (68 cells ~1e-4, 48 metadata). The schema is preserved; the bytes are not. |
| 22 | [GPU_CA_IMPLEMENTATION.md:508-511](GPU_CA_IMPLEMENTATION.md#L508-L511) | "On the GB10's unified memory these arrays are directly GPU-addressable." `p_prepare` always does `cudaMalloc` + `cudaMemcpy` ([:212-224](src/math/cudss/cudss_linear_solver_implementation.hpp#L212-L224)). There is no zero-copy path. |
| 23 | [Dockerfile](Dockerfile), [src/CMakeLists.txt:405-410](src/CMakeLists.txt#L405-L410) | `CMAKE_CUDA_ARCHITECTURES=121` is **inert** — `enable_language(CUDA)` is never called and there are no `.cu` files. The comment "ship PTX as a hedge" describes nothing. |
| 24 | [input_14_gpu.xml](src/applications/data_sets/input/ca/input_14_gpu.xml) | Advertises `<hybridMemory>`, which is read and then ignored ([:372-375](src/math/cudss/cudss_linear_solver_implementation.hpp#L372-L375)), and `<deterministic>` / `<iterativeRefinement>`, which apply only to the per-case drop-in backend and **not** to `CuDSSBatchedSolver` — so with `batched=true` they are silently no-ops. The file's header also claims byte-for-byte CSV parity. |
| 25 | [ca_async_writer.hpp:104](src/applications/contingency_analysis/ca_async_writer.hpp#L104) | `wroteAnything()` has zero callers and reads `p_wrote` (written by the writer thread) without synchronization — a formal data race on dead API. Delete it. |
| 26 | [cudss_batched_solver.hpp:303-306](src/math/cudss/cudss_batched_solver.hpp#L303-L306) | Hard-codes `CUDSS_R_32I` / `int` indices while the drop-in backend correctly derives the type from `sizeof(PetscInt)` ([cudss_csr_extractor.hpp:69-72](src/math/cudss/cudss_csr_extractor.hpp#L69-L72)). On a `--with-64-bit-indices` PETSc build the assembler's `static_cast<int>` truncates silently. Add a `nnz > INT_MAX` guard at minimum. |
| 27 | [cudss_linear_solver_implementation.hpp:212-247](src/math/cudss/cudss_linear_solver_implementation.hpp#L212-L247) | If a later `cudaMalloc` / `cudssMatrixCreate` throws inside `p_prepare`, earlier allocations leak — the constructor has a try/catch but `p_prepare` does not, and the PETSc fallback in `petsc_linear_solver.cpp` only catches at construction time. Wrap the body in try/catch → `p_freeDeviceAndMatrices()` → rethrow. |
| 28 | [ca_driver.cpp:2203-2206](src/applications/contingency_analysis/ca_driver.cpp#L2203-L2206), [:2294-2297](src/applications/contingency_analysis/ca_driver.cpp#L2294-L2297) | The `[connectivity screen]` line prints `screenSkipped`, which is **reset every wave**, so it reads like a global count but is always ≤ waveSize; `bridgeCount` (1,061) is reprinted identically every wave. `[batched GPU controllers]` prints even when `nredo == 0`. Rank 0 emits ~170 near-duplicate lines. Aggregate these into the existing `MPI_Reduce` summary. |

---

## Where the external review is wrong about HEAD

Four of the eight "important gaps" in `Research GridPACK speedups.md` describe code that no longer
exists — its own later commits fixed them, but the summary was never corrected, and one was caused
by reading the stale `pf_screen.hpp` header (finding 19 above).

| External claim | Verdict |
|---|---|
| "The connectivity algorithm is quadratic — approximately O(E²α(V))" | **Wrong at HEAD.** `screenAllBranchOutages()` is a single iterative Tarjan bridge pass, O(V+E), since `81d9015a`. The union-find `componentsWithout()` survives only as a reference helper. |
| "`waveSize` and `GPU.screen` are effectively inert" | **Wrong at HEAD.** `waveSize` is read at [:1668](src/applications/contingency_analysis/ca_driver.cpp#L1668) and bounds the reservation loop at [:2175](src/applications/contingency_analysis/ca_driver.cpp#L2175); `screen` is read at [:1663](src/applications/contingency_analysis/ca_driver.cpp#L1663) and drives `p_screenEnabled`. Both are live. (`waveSize` *is* badly validated — finding 3.) |
| "The connectivity screen is diagnostic, not a pre-screen" | **Wrong at HEAD.** `prepare()` consults `p_lineIsBridge` **before** the expensive probe and skips it on a hit ([:210-220](src/applications/modules/powerflow/pf_batch_ca_assembler.hpp#L210-L220)). |
| "`prepare()` restores all 6,717 bus states once per event before consulting the screen" | **Wrong at HEAD.** `p_restoreStart()` moved inside the `if (!screened)` block. |
| "The batch is not concurrent; `batchCount=1`" | **Correct.** `solveWave` constructs `CuDSSBatchedSolver(n, nnz, ..., 1)` and loops cases sequentially. |
| "Every Newton step performs synchronous transfers" | **Correct.** No pinned memory, no streams, no async copies; `cudaMemcpy` on the default stream per RHS and per solution. |
| "cuDSS knobs don't affect the batched engine; `hybridMemory` is a no-op" | **Correct** — finding 24. |
| "The fallback decision is static, not a cost model" | **Correct.** |
| "Output remains an Amdahl bottleneck" | **Correct**, and the right strategic read. |

The external review's architectural recommendations (GPU-resident waves with device-side assembly,
CUDA graphs, active-case compaction, DC/LODF pre-screening, portfolio solver selection, columnar
output) are sound and consistent with the code. One caution on its "one controlling rank per GPU"
topology suggestion: that is precisely the rank-0-only design `d939666f` replaced, and the measured
evidence (86.26 s all-rank vs 92.28 s rank-0-only) says all-rank wins on this hardware.

---

## Suggested order of work

1. **Findings 1, 5, 6** — restore a working correctness oracle and fail loudly on the two
   silent-wrong-answer paths. Everything else is unsafe to change until assembly correctness can be
   proven on demand.
2. **Finding 2** — incremental MPI-IO with reserved extents. The current default is correct only on
   a local filesystem with a libstdc++ implementation detail holding.
3. **Findings 3, 13** — input validation and a real chord policy (adaptive refactorization plus
   `chordCap` from config). Cheap, removes footguns, attacks the 279 non-converged fallbacks.
4. **Findings 11, 12** — hoist the wave-invariant setup and eliminate the double
   `applyCaseForOutput`. This is the largest remaining measurable win in the GPU path, and it must be
   done *with* the state-leak invariant from finding 11 made explicit.
5. **Findings 7, 8** — fix the rating/violation semantics and stop writing zeros into the
   convergence sidecar, so the accuracy report means what it says.
6. **Findings 14-28** — delete the dead code and correct the comments that assert the rejected
   design. Do this before the next reader reaches `cudss_batched_solver.hpp:169` and concludes the
   wave shares one factorization.
