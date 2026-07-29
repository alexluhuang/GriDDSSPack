# From GridPACK `feature/ca-scalability` to GriDSSPack

## Design history, conceptual foundations, accuracy safeguards, and performance evidence

| Document property | Value |
|---|---|
| Document type | Explanation, followed by reference appendices |
| Audience | Readers with intermediate programming, calculus, linear algebra, and introductory physics |
| Base boundary | GridPACK through commit `cff541f4a0fff89ddd71f254833efad50419cd68`, inclusive |
| Current revision reviewed | GriDSSPack commit `f0a97125ee7af7071e0f444937640a2424fb546c` |
| Main validation workload | Full Texas7k branch-and-generator N-1 run, 8,891 contingencies, 20 MPI ranks, exhaustive `csv_flat` output |
| Report date | 2026-07-29 |

> **Naming note:** the repository and image use **GriDSSPack**. This report uses
> that spelling for the current project, even where earlier discussion called it
> “GriDDSS.”

## How this document is organized

The [Divio documentation system](https://docs.divio.com/documentation-system/)
separates tutorials, how-to guides, reference material, and explanations because
they answer different questions. This report is primarily an
[explanation](https://docs.divio.com/documentation-system/explanation/): it
provides context, explains why the design changed, and discusses alternatives
that failed. Exact commit, file, setting, and measurement details are collected
in reference-oriented tables and appendices, following Divio's recommendation
that [reference material remain code-determined and precise](https://docs.divio.com/documentation-system/reference/).

Related documents keep task instructions separate:

- Use [DOCKER_CA.md](DOCKER_CA.md) to build and run the image.
- Use [gpucputest/README.md](gpucputest/README.md) to run the large-output
  comparison.
- Use this report to understand what changed, why it changed, and how the
  changes affect speed and accuracy.

The report deliberately explains the electrical and numerical ideas before the
commit history. That makes the history understandable rather than reducing it to
a list of C++ edits.

### Contents

- [Executive summary](#executive-summary)
- [Scope, method, and evidence](#1-scope-method-and-evidence)
- [Validated workload](#2-the-workload-used-for-performance-and-accuracy-claims)
- [Conceptual foundations](#3-conceptual-foundations)
- [Chronological implementation history](#4-chronological-implementation-history)
- [Current end-to-end system](#5-how-the-current-system-works-end-to-end)
- [Speed attribution](#6-why-the-complete-speedup-has-several-sources)
- [Speed and accuracy ledger](#7-speed-and-accuracy-effect-of-each-optimization-class)
- [Rejected approaches](#8-consolidated-record-of-approaches-that-did-not-work)
- [Limitations](#9-current-limitations-and-documentation-debt)
- [Conclusions](#10-conclusions)
- [Reference appendices](#reference-appendices)

## Executive summary

GriDSSPack did not become faster through one GPU replacement. It became faster
by removing different kinds of repeated work from the complete application:

1. The Docker build changed from GridPACK `Debug` to `Release`, accelerating
   CPU code throughout the program.
2. The hot contingency path stopped rebuilding and transporting sparse data
   through general-purpose mapping machinery when the relevant structure was
   already known.
3. A linear-time graph pass replaced repeated connectivity tests for simple
   branch outages.
4. cuDSS reuses sparse structural analysis, and the modified-Newton path reuses
   each contingency's numerical factors while the nonlinear mismatch continues
   to improve.
5. All 20 MPI ranks claim bounded waves of eight tasks. No one rank is reserved
   as a GPU worker or broker.
6. Cases that do not fit the fast assumptions, fail to converge, or need a
   controller action return to the established PETSc/KLU path.
7. CSV production now reads branch objects directly and all ranks append
   complete event blocks to the final file, avoiding both text round trips and
   a large serial concatenation step.
8. Later hardening made the fast assembler verifiable, made sparse-pattern
   assumptions fail closed, added adaptive refactorization, repaired diagnostic
   fields, and strengthened CSV correctness.

The distinction between **whole-application speedup** and **GPU contribution**
is essential:

| Comparison on the same full Texas workload | Time | Relative result |
|---|---:|---:|
| Stock `pnnl/gridpack:ca-scalability-v2` CPU | 214.136 s | Reference |
| Current GriDSSPack, optimized CPU mode | 89.922 s | 2.38× stock |
| Current GriDSSPack, opt-in GPU mode | 68.260 s | 3.14× stock; 1.32× optimized CPU |

These current-session measurements mean:

- The complete current product was about 3.14 times as fast as the stock image.
- Enabling the current GPU path reduced the optimized-CPU time by about 24.1%.
- It would be incorrect to call the full 3.14× result a “GPU speedup.” Most of
  the stock-to-current gain also includes the release build, faster assembly,
  output work, scheduling, and other CPU-side changes.

An earlier implementation showed only a 0.84% median improvement over the
optimized CPU path. That result came from an intermediate rank-owner experiment
whose logged 446 eligible contingencies represented one rank's local share, not
the global stream. The later bounded all-rank design inspected all 8,891 tasks
and found 7,099 initially eligible cases.

Accuracy also needs a precise statement. GriDSSPack does not promise that CPU
and GPU floating-point files are byte-for-byte identical. It preserves:

- the same AC equations;
- the same nonlinear convergence tolerance;
- exact CPU fallback for cases that do not satisfy the fast-path conditions;
- full key and schema comparison;
- outcome-level comparison and numerical-error measurement.

The historical committed comparison found identical convergence outcomes for
all 8,891 cases but also exposed output and diagnostic differences. The later
hardening addressed those findings. The current full comparison found equal
row coverage and no unmatched convergence, status, or violation records between
the optimized CPU and GPU modes.

## 1. Scope, method, and evidence

### 1.1 Exact version boundary

The user-defined base includes commit
[`cff541f4`](https://github.com/alexluhuang/GriDSSPack/commit/cff541f4a0fff89ddd71f254833efad50419cd68)
and every ancestor. The changes attributed to GriDSSPack begin with
[`10e5cf26`](https://github.com/alexluhuang/GriDSSPack/commit/10e5cf26a9572c117050302f1d40d0d95762e1a0)
and end, for this report, at
[`f0a97125`](https://github.com/alexluhuang/GriDSSPack/commit/f0a97125ee7af7071e0f444937640a2424fb546c).

The range contains 9 commits, 46 changed files, 99,297 inserted lines, and 68
deleted lines. About 91,000 inserted lines are stored comparison data, not
runtime code.

This branch is not “the newest upstream GridPACK plus GPU.” After the base
boundary, upstream `feature/ca-scalability` continued on a separate line of
development. Its later phantom-flow, rating, violation-reporting, ranking, and
summary changes are not ancestors of current GriDSSPack. This report therefore
compares exactly the requested base with the current fork, not with the later
upstream branch tip.

### 1.2 Evidence hierarchy

Claims in this report use four levels of evidence:

| Evidence label | Meaning |
|---|---|
| Source-verified | Present in the Git diff or current implementation |
| Stored result | Reproducible from committed files under `gpucputest/results/` |
| Session result | Produced during the development and validation work described in this conversation, but not committed as a result artifact |
| Engineering inference | A causal explanation consistent with the code and measurements; not an isolated benchmark of that one change |

This distinction matters because several changes were introduced together. A
complete run can measure their combined result, but cannot honestly assign an
exact number of seconds to every change without an ablation build for each one.

### 1.3 What “faster” and “more accurate” mean here

“Faster” can refer to different costs:

- shorter Docker build time;
- shorter one-time application setup;
- less CPU assembly work;
- fewer sparse factorizations;
- less host-to-device traffic;
- better load balance among MPI ranks;
- less output formatting work;
- removal of the final serial file merge.

“More accurate” can also refer to different properties:

- solving the same physical equations to the requested tolerance;
- rejecting an unsafe fast-path assumption;
- falling back rather than emitting a nonconverged approximation;
- reporting the true worst active- and reactive-power mismatches;
- emitting every physical circuit;
- using the selected equipment rating consistently;
- preserving complete CSV fields and valid schema.

A change can improve reporting accuracy without changing the voltage solution.
Likewise, a scheduler can improve speed without changing any numerical value.
Each chronological section identifies the category explicitly.

### 1.4 Research ideas versus implemented features

The earlier literature review contributed candidate ideas: fixed-matrix
iteration from GPU N-1 fast-decoupled work, Union-Set connectivity screening,
same-pattern sparse batches, transfer minimization, and bounded processing under
GPU memory limits. This report credits an idea only when the Git history and
source implement it.

In particular, current GriDSSPack uses a full-Jacobian chord method with
adaptive refresh; it is not the paper's true \(B'/B''\) fast-decoupled method.
It also does not yet have device-resident mismatch/Jacobian assembly, active-case
compaction, mixed precision, or a successful native cuDSS uniform batch. Those
remain research directions, not sources of the measured speedup.

## 2. The workload used for performance and accuracy claims

Performance changed substantially with system size, output volume, rank count,
and the fraction of cases eligible for factor reuse. The primary benchmark is
therefore the full scenario, not IEEE-14 or a 64/256-case sample.

The attached Texas input uses:

| Setting | Value | Consequence |
|---|---|---|
| Network file | `Texas7k_20210804.raw` | 7k-class transmission model |
| `FullBranchN1` | `true` | Generate every supported single branch/circuit outage |
| `FullGeneratorN1` | `true` | Generate every supported single generator outage |
| Total contingencies | 8,891 | Complete task stream inspected by all ranks collectively |
| MPI ranks | 20 | One rank-local network and GPU context per process |
| `groupSize` | `1` | Each MPI rank owns a complete serial network for its claimed tasks |
| Q-limit checking | `true` | Cases needing reactive-limit control return to the full controller loop |
| LTC / switched shunt / area interchange | `false` / `false` / `false` | No transformer-tap, shunt, or area-slack outer loop in this deck |
| Newton maximum / tolerance | 50 / `1.0e-4` | Same nonlinear acceptance target in compared modes |
| Output | `csv_flat` | Full event-by-circuit output is included in wall time |
| Rating | `C` | Contingency `csv_flat` rating, loading, and row-level `viol` use C→B→A; base rows use A, and the separate legacy overload checker has its own A/B policy |
| Shared output | `sharedFlatFile=true`, `bufferFlatOutput=false` | Direct shared append; no final flat-file concatenation |
| GPU wave | enabled, batched, size 8, warm start, screen | Current all-rank fast path |

The required flat-file schema is:

```text
event_idx,contingency,from_bus,to_bus,circuit_id,p_from_mw,q_from_mvar,mva_from,rate_mva,loading_percent,viol,v_from_pu,v_to_pu,ang_from_deg,ang_to_deg
```

The convergence and bus sidecars have their own stable headers documented in
[DOCKER_CA.md](DOCKER_CA.md). A fair speed comparison must use the same network,
contingency generation, rank count, controller settings, tolerance, and output
burden. A run that selects `text` output instead of the multi-gigabyte
`csv_flat` file is not comparable.

## 3. Conceptual foundations

### 3.1 The network, voltage, current, and admittance

A power network is represented as:

- **buses**, the connection points at which voltage is defined;
- **circuits**, the transmission lines or transformers joining buses;
- generators and loads attached to buses.

Alternating-current voltage has both a size and a timing angle. It is therefore
represented by a complex number, or **phasor**:

\[
V_i = |V_i|e^{j\theta_i}.
\]

**Admittance** describes how readily current flows when a voltage difference is
applied. It is the reciprocal of impedance:

\[
Y=\frac{1}{Z}=G+jB.
\]

This is more than a renamed resistance. The real part \(G\), conductance,
describes in-phase current and real-energy loss. The imaginary part \(B\),
susceptance, describes the shifted current associated with electric and magnetic
field storage. A large admittance means a small voltage difference can drive a
large current. A small admittance means the circuit strongly opposes current.

Every in-service circuit contributes a small set of terms to a network matrix
called **Y-bus**. The assembled relationship is:

\[
I=Y_{\text{bus}}V.
\]

With the injection convention used here, each row gives the net current
injected from that bus into the network for the stated bus voltages. A line
mainly touches its two endpoint rows and columns. Most bus pairs are not
directly connected, so most Y-bus entries are zero. That physical locality is
the reason the matrix is sparse.

Taking one circuit out changes its own contribution and the endpoint diagonal
totals. GriDSSPack's local Y-bus update exploits this locality; it does not
invent an approximate electrical model.

### 3.2 Why AC power flow is nonlinear

The current relationship \(I=YV\) is linear in voltage. Power is not:

\[
S=P+jQ=VI^*.
\]

Voltage is multiplied by the complex conjugate of current, and current itself
depends on voltage. The calculated real power \(P\) and reactive power \(Q\)
therefore contain products of voltage magnitudes and trigonometric functions of
angle differences.

The solver knows what power each bus is supposed to inject or consume from the
network data. For a trial set of voltages, it can calculate what the network
would actually inject. Their difference is the **power mismatch**:

\[
F(x)=
\begin{bmatrix}
P_{\text{specified}}-P_{\text{calculated}}(x)\\
Q_{\text{specified}}-Q_{\text{calculated}}(x)
\end{bmatrix}.
\]

This is a reduced vector rather than two entries for every physical bus. It
normally contains one real-power entry for each non-slack active bus and one
reactive-power entry for each active PQ bus.

The equation is useful, but the concept is simpler:

> A mismatch is the amount by which the proposed voltages fail the physical
> power bookkeeping at the buses.

If the specified net injection is +100 MW and the trial voltages produce
+97 MW, this convention gives a +3 MW real-power mismatch. The trial voltages
are not yet a valid steady-state solution. Newton iteration changes the voltages
until every required balance error is small enough.

In this nonlinear context, **residual** and **mismatch** refer to the same
remaining equation error, viewed from two perspectives:

- “Mismatch” emphasizes its physical meaning: missing or excess \(P\) and \(Q\).
- “Residual” emphasizes its numerical meaning: how far \(F(x)\) remains from
  zero.

There is also a different, temporary **linear-solve residual**:
\(r_{\text{linear}}=b-A\Delta x\). A sparse solver can solve its linear system
very accurately while the outer nonlinear power mismatch remains large. That is
why GriDSSPack recomputes the true AC mismatch after every chord step. Passing
the linear solve alone does not prove the power flow converged.

### 3.3 Bus types and why structure can change

The reduced power-flow state does not assign the same unknowns to every bus:

| Bus type | Known quantities | Main unknowns/equations in the reduced solve |
|---|---|---|
| Slack/reference | Voltage magnitude and angle | No ordinary \(P/Q\) correction pair; it balances the system |
| PV/generator | Real power and voltage magnitude | Voltage angle, using a real-power equation |
| PQ/load | Real and reactive power | Voltage angle and magnitude, using real- and reactive-power equations |
| Isolated | Electrically disconnected | Removed from the active reduced system |

A reactive-power limit can cause a PV bus to behave temporarily as a PQ bus.
Islanding can change which bus acts as reference. Those changes alter the number
or arrangement of reduced equations. Reusing a sparse structure is safe only
when those structural assumptions remain true. This is why controller-changing,
islanding, generator, and other irregular cases use the general CPU path.

### 3.4 Newton's method: a local linear model

Let \(x\) contain the unknown voltage angles and selected voltage magnitudes.
Define the conventional power-flow Jacobian from the **calculated** powers:

\[
J(x)=
\frac{\partial[P_{\text{calculated}},Q_{\text{calculated}}]}{\partial x}.
\]

Because this report defined mismatch as specified minus calculated, its
first-order change has the opposite sign:

\[
F(x_k+\Delta x)\approx F(x_k)-J(x_k)\Delta x.
\]

Setting that approximation to zero gives:

\[
J(x_k)\Delta x=F(x_k), \qquad x_{k+1}=x_k+\Delta x.
\]

GridPACK internally uses the algebraically equivalent opposite convention:
calculated minus specified, an opposite-signed correction, and a subtractive
state update. The physical result is unchanged as long as the residual,
right-hand side, solve, and update signs are used consistently.

The Jacobian is not Y-bus:

- Y-bus maps voltages to currents and represents network admittance.
- The conventional Jacobian maps small voltage-state changes to changes in
  calculated \(P/Q\). Since mismatch here is specified minus calculated, its
  change has the opposite sign.
- Y-bus helps compute the Jacobian, but the two matrices have different meanings.

“Exact Newton” in this report means the Jacobian is reevaluated and numerically
refactored at the current state for every step. It does **not** mean exact
arithmetic, exact equality between CPU and GPU bits, or guaranteed one-step
convergence.

Near a well-behaved solution, exact Newton often has **quadratic convergence**:
once close enough, the number of correct digits can grow very quickly. That
fast local convergence is purchased by rebuilding and refactoring the Jacobian
repeatedly.

### 3.5 The three stages of a sparse direct solve

Under this report's sign convention, each Newton step contains a sparse linear
system \(A\Delta x=b\), where \(A=J(x_k)\) and \(b=F(x_k)\). GridPACK's
opposite residual and subtractive-update convention is algebraically
equivalent. A sparse direct solver divides its work into three conceptually
different stages.

#### Stage 1: structural or symbolic analysis

This stage examines **which row/column positions belong to the stored sparse
pattern**, not their present numeric values. A structurally stored position can
currently contain zero. The solver chooses an equation ordering intended to
keep the factors sparse, predicts fill according to its ordering and pivoting
policy, allocates internal storage, and establishes dependencies.

Why ordering matters: eliminating one unknown can connect equations that were
not previously connected. Those new factor entries are called **fill-in**.
Different orderings solve the same equations but can create dramatically
different amounts of work and memory.

Structural analysis can be reused while all of these remain unchanged:

- matrix dimensions;
- row offsets;
- column positions;
- the solver's relevant structural assumptions.

Equal dimensions and equal nonzero counts are not sufficient. Two matrices can
have the same size and number of entries but place those entries differently.
Current GriDSSPack therefore compares the actual CSR row and column arrays
before reusing a cached analysis.

#### Stage 2: numeric factorization

The solver now uses the current numbers to decompose the ordered matrix into
easier triangular factors, conceptually:

\[
PAQ=LU,
\]

where \(P\) and \(Q\) represent ordering/permutation choices, \(L\) is lower
triangular, and \(U\) is upper triangular.

**Factoring** means computing those \(L\) and \(U\) numbers. It is not the same
as “speedup factor.” Sparse numeric factorization is usually much more
expensive than solving with factors that already exist.

If matrix values change but their positions do not, structural analysis can
usually be reused, but the numeric factors normally need to be recomputed.

#### Stage 3: triangular solve

Once factors exist, the solver applies permutations and performs:

\[
z=Q^{-1}\Delta x,\qquad Ly=Pb,\qquad Uz=y,\qquad \Delta x=Qz.
\]

The first equation is solved from top to bottom (**forward substitution**); the
second is solved from bottom to top (**back substitution**). This is the
**triangular solve**. A new right-hand side can reuse both structural analysis
and numeric factors, making this stage much cheaper than refactorization.

The reuse rule is:

| What changed? | Structural analysis | Numeric factorization | Triangular solve |
|---|---|---|---|
| Only the RHS, while the matrix is deliberately held fixed (the chord case) | Reuse | Reuse | Run again |
| Matrix values, same CSR pattern | Reuse | Recompute | Run |
| CSR pattern or dimension | Recompute | Recompute | Run |

cuDSS names essentially these phases analysis, factorization, and solve.

### 3.6 CSR: the sparse storage contract

**Compressed Sparse Row (CSR)** stores only selected row/column positions in
the sparse pattern. A stored value can temporarily be numerically zero. CSR
uses three arrays:

- `row_ptr`: where each row starts in the other arrays;
- `col_idx`: the column of each stored entry;
- `values`: the current numeric value at that position.

For example:

\[
A=
\begin{bmatrix}
10&0&-2\\
0&5&1\\
-2&1&7
\end{bmatrix}
\]

has:

```text
row_ptr = [0, 2, 4, 7]
col_idx = [0, 2, 1, 2, 0, 1, 2]
values  = [10, -2, 5, 1, -2, 1, 7]
```

`row_ptr[1]=2` and `row_ptr[2]=4` mean zero-based row 1—the second row—occupies
positions 2 and 3 in `col_idx`/`values`. CSR is a storage layout, not a solver
and not a physical approximation.

GridPACK components naturally compute small blocks of the Jacobian. The stock
general GridPACK/GA/PETSc matrix-assembly path discovers and transports those
contributions. The direct component-to-CSR path precomputes where each small
block belongs, zeroes the numeric array for the new assembly, and adds each
contribution into its cached CSR slot.

This change is analogous to replacing repeated address lookups with a verified
address book:

- the electrical formulas do not change;
- for the same live network and voltage state, the component formulas and
  resulting entries are intended to match;
- only the route by which they reach the sparse array changes.

The current validation mode checks the fast result against canonical GridPACK
assembly and fails closed on a difference. Exact arrays, tolerance, and failure
behavior are recorded in Section 4.10 and Appendix B.

### 3.7 Chord, or modified Newton, and adaptive factor reuse

Exact Newton updates the local tangent matrix every step. A **chord method**
holds a Jacobian fixed between refactor points:

\[
B\Delta x_k=F(x_k), \qquad x_{k+1}=x_k+\Delta x_k,
\]

where \(B\) is a previously computed Jacobian.

In this report and code, chord means simplified or modified Newton with a
Jacobian frozen between refresh points. Some numerical-analysis sources use the
word for related fixed-slope or secant variants. It is unrelated to a graph
chord.

The current GriDSSPack constant-factor path does **not** use one base-network
factorization for every outage. It:

1. applies one contingency at the warm-start voltages;
2. assembles that contingency's first Jacobian and mismatch;
3. factors that case-specific Jacobian;
4. reuses its factors for later mismatch right-hand sides;
5. recomputes the true nonlinear \(P/Q\) mismatch after every correction;
6. refreshes the current case Jacobian when progress becomes poor;
7. abandons the fast result and performs exact CPU Newton if convergence is not
   acceptable.

Why it is faster: repeated triangular solves are cheaper than repeated numeric
factorizations.

Why it can take more iterations: the fixed matrix becomes a less accurate local
description as the trial voltage moves. When the frozen Jacobian is sufficiently
good and the iteration converges locally, convergence is generally linear
rather than quadratic. A poor frozen Jacobian can stagnate or diverge.

Why convergence accuracy can remain the same: the acceptance test is still the
actual \(F(x)\), not the fixed linear model. If \(F(x)\) reaches the same
tolerance, it meets the same residual stopping test for the same equations.
This does not promise bitwise equality or the same root when a nonlinear system
has multiple roots. If it does not reach tolerance, the case is not emitted as
a successful approximation; it is re-solved through the exact path.

### 3.8 Connectivity screening: repeated Union-Find versus Tarjan bridges

Assume the active base graph is connected and one contingency removes one
circuit edge. Treat active buses as graph vertices and in-service circuits as
graph edges.

Removing one edge can:

- leave an alternate route between its endpoints, so the graph remains
  connected; or
- remove the only route, splitting the graph into islands.

An edge of the second kind is a **bridge**.

The first screen used this exact but repeated procedure for every proposed
outage:

1. omit that one edge;
2. Union-Find all other edge endpoints into connected sets;
3. count the resulting components.

After initialization, one candidate costs \(O(E\alpha(V))\) with Union-Find.
Repeating it for \(O(E)\) edge candidates costs approximately
\(O(E^2\alpha(V))\).

Tarjan's bridge algorithm traverses the graph once. During depth-first search it
records:

- when each vertex was discovered;
- the smallest discovery index reachable from a vertex's depth-first-search
  subtree using tree edges and at most one non-parent back edge.

For a tree edge from parent \(u\) to child \(v\), it is a bridge when:

\[
\operatorname{low}(v)>\operatorname{discovery}(u).
\]

The condition means no edge path from \(v\)'s subtree reaches \(u\) or an
ancestor without using the tested tree edge. All bridges are found in
\(O(V+E)\) time.

GriDSSPack's implementation uses an explicit stack instead of the program call
stack, avoiding stack overflow on a long chain. It identifies the parent by
**edge ID**, not just by the neighboring bus. That detail is essential for
parallel circuits: if two circuits join the same buses, removing one still
leaves the other route, so neither one is a bridge.

For a connected active base graph and a single-edge outage, repeated
connectivity testing reports islanding exactly when Tarjan classifies that edge
as a bridge. Tarjan improves cost, not physics. Current hardening builds the
graph only from active, non-isolated buses and in-service circuits, validates
endpoints, and trusts the shortcut only when the base active graph is connected.
Unknown or complex cases use the established topology check.

## 4. Chronological implementation history

### 4.1 Base GridPACK through `cff541f4` — the starting point

The base was already a capable parallel contingency-analysis application. It is
important not to credit existing behavior to GriDSSPack.

#### What the base already did

In the base
[`ca_driver.cpp`](https://github.com/alexluhuang/GriDSSPack/blob/cff541f4a0fff89ddd71f254833efad50419cd68/src/applications/contingency_analysis/ca_driver.cpp):

- `TaskManager::nextTask` dynamically assigned one contingency at a time.
- Each task restored the base state, applied its contingency, checked topology,
  ran the ordinary power flow, captured output, and restored the network.
- Full branch and generator N-1 generation already existed.
- Q-limit handling, deadband settings, rating selection, monitoring filters,
  convergence sidecars, `csv_flat`, `csv_delta`, and `writeStats` already
  existed.
- `csv_flat` was not the source-code default—it defaulted to `text`—but the
  attached Texas workload explicitly selected `csv_flat`.
- `csv_flat` rows were streamed to one `.part` file per rank. After all solving,
  rank 0 serially read every part and rewrote the final file.

In
[`petsc_linear_solver.cpp`](https://github.com/alexluhuang/GriDSSPack/blob/cff541f4a0fff89ddd71f254833efad50419cd68/src/math/petsc/petsc_linear_solver.cpp),
GridPACK's linear-solver factory always constructed the PETSc implementation.
The attached benchmark's `PETScOptions` then selected CPU KLU for the serial
reduced Jacobian on each rank; KLU was an XML/PETSc choice, not a separate
factory branch.

The base Dockerfile:

- started from `ubuntu:questing`;
- built GridPACK with `CMAKE_BUILD_TYPE=Debug`;
- had no CUDA/cuDSS build;
- used the same user-facing `mpirun -n 20 ca.x input.xml` command.

#### Baseline computational shape

MPI already parallelized independent contingencies. Within each contingency,
however, every Newton step followed the general CPU assembly and direct-solve
path. The final flat-file merge was a serial tail: once all ranks finished, only
rank 0 was doing useful work.

Accuracy was the reference behavior for this project. The optimization goal was
not to change the AC model, but to remove repeated setup, transport,
factorization, and output work while retaining the base path as a fallback.

---

### 4.2 `10e5cf26` — GPU prerequisites, a generic cuDSS backend, and prototypes

Commit:
[`10e5cf26`](https://github.com/alexluhuang/GriDSSPack/commit/10e5cf26a9572c117050302f1d40d0d95762e1a0),
2026-07-17.

This first post-base commit added 2,756 lines across 23 files. It established
infrastructure and several prototypes, but only the scalar cuDSS solver was
connected to real contingency analysis.

#### Change 1: CUDA/cuDSS Docker build and Release mode

**Where**

- [`Dockerfile`](Dockerfile)
- [`src/CMakeLists.txt`](src/CMakeLists.txt)
- [`src/math/CMakeLists.txt`](src/math/CMakeLists.txt)

**What changed**

- The base Ubuntu image was replaced by a CUDA 13 development image for SBSA
  ARM64.
- cuDSS 0.8.0.10 was pinned and installed.
- `GRIDPACK_WITH_CUDSS` was added as a CMake option, defaulting to `OFF` for
  ordinary builds.
- CMake discovers `CUDAToolkit` and cuDSS and links `cudss` plus the CUDA
  runtime when enabled.
- GridPACK changed from `Debug` to `Release`.
- The Docker configuration specified architecture 121 for the GB10 target.

**Reasoning**

The same `ca.x` needed to contain CPU and GPU implementations so users could
select the path in XML without learning a different executable or command.
Release mode was also necessary for meaningful performance work.

**Speed effect**

Release compilation is a broad CPU optimization. It accelerates parsing,
component calculations, KLU calls around the solver, task handling, and output.
It is part of the stock-to-GriDSSPack gain but not evidence that the GPU itself
is faster.

The architecture value does not compile project-owned GPU kernels because this
repository does not enable the CMake CUDA language or contain a `.cu` kernel in
this path. cuDSS supplies its own precompiled GPU code. The setting is therefore
mostly build metadata for the current implementation.

**Accuracy effect**

The physical equations and precision remain double precision. Release mode can
change low-level floating-point ordering relative to a debug build, so equality
should be judged by tolerance and results, not bytes.

#### Change 2: runtime-selectable linear-solver backend

**Where**

- [`src/math/linear_solver_backend.hpp`](src/math/linear_solver_backend.hpp)
- [`src/math/linear_solver_backend.cpp`](src/math/linear_solver_backend.cpp)
- [`src/math/petsc/petsc_linear_solver.cpp`](src/math/petsc/petsc_linear_solver.cpp)
- [`src/applications/contingency_analysis/ca_driver.cpp`](src/applications/contingency_analysis/ca_driver.cpp)

**What changed**

A process-wide selector can request PETSc or cuDSS. The solver factory creates
the cuDSS implementation when compiled, requested, and able to see a CUDA
device; otherwise it creates the PETSc implementation. The CA driver initially
accepted either `<GPU><enabled>true</enabled>` or
`<LinearSolver><Backend>cudss</Backend>` as a request.

**Reasoning**

This preserves the ordinary GridPACK math interfaces. Power-flow code still
asks for a linear solver; the factory decides which implementation satisfies
that request.

**Speed effect**

Selection alone saves no time. It makes experiments and graceful CPU operation
possible.

**Accuracy and reliability effect**

The fallback handled unavailable devices and construction failures. At this
stage it did not catch every failure that might occur later during
factorization or solve.

#### Change 3: PETSc-to-CSR extraction and scalar cuDSS direct solve

**Where**

- [`src/math/cudss/cudss_csr_extractor.hpp`](src/math/cudss/cudss_csr_extractor.hpp)
- [`src/math/cudss/cudss_exception.hpp`](src/math/cudss/cudss_exception.hpp)
- [`src/math/cudss/cudss_linear_solver_implementation.hpp`](src/math/cudss/cudss_linear_solver_implementation.hpp)

**What changed**

`PetscSeqCSRView` exposes a serial PETSc AIJ matrix as CSR. The cuDSS
implementation maps GridPACK's solver lifecycle onto:

1. CSR allocation/upload and cuDSS analysis;
2. value upload and numeric factorization;
3. right-hand-side upload, solve, and correction download.

**Reasoning and theory**

Newton normally retains the same Jacobian positions while its derivative values
change. Reusing analysis should avoid repeated ordering and sparse-structure
work.

**Speed effect**

The theory did not overcome end-to-end overhead in this first real path. Later
Texas-256 measurements showed:

| Mode | 1 rank | 20 ranks sharing one GPU |
|---|---:|---:|
| Per-contingency cuDSS | 42.10 s | 13.22 s |
| CPU KLU | 36.36 s | 4.86 s |

Each contingency still paid for a cold solver/analysis, synchronous copies, and
GPU launch overhead. Twenty processes also contended for one device. This
negative result motivated reuse across cases rather than simple solver
replacement.

**Accuracy effect and early gaps**

IEEE-14 CPU/GPU parity was tested, but full Texas N-1 was not yet validated.
cuDSS and KLU solve the same FP64 linear system but need not produce identical
bits.

The initial analysis cache considered only the matrix dimensions and number of
stored entries. It did not compare the actual CSR pattern, so a same-sized but
differently arranged matrix could incorrectly reuse analysis. The advertised
`hybridMemory` option was reserved but did nothing, and data moved through
`cudaMemcpy`; it was not a zero-copy implementation.

#### Change 4: synthetic multi-case Newton engine

**Where**

- [`src/applications/modules/powerflow/pf_batch_ca.hpp`](src/applications/modules/powerflow/pf_batch_ca.hpp)
- [`src/math/cudss/cudss_batched_solver.hpp`](src/math/cudss/cudss_batched_solver.hpp)
- [`src/math/test/pf_batch_ca_test.cpp`](src/math/test/pf_batch_ca_test.cpp)
- [`src/math/test/cudss_batched_test.cpp`](src/math/test/cudss_batched_test.cpp)

**What changed**

The first `PFBatchNR` advanced a wave in lockstep: assemble every active case,
make one native cuDSS multi-matrix call, apply all corrections, and retire
converged cases. `CuDSSBatchedSolver` used cuDSS batch CSR/dense matrix objects.

**Reasoning**

Independent systems with the same pattern appear ideal for parallel GPU work.
One analysis and a larger batch should improve arithmetic intensity.

**Actual effect**

This was a tested numerical prototype behind an abstract `BatchAssembler`. The
CA driver explicitly reported that `<GPU><batched>true</batched>` was not yet
connected. It produced no full-Texas runtime improvement in this commit.

#### Change 5: first connectivity-screen prototype

**Where**

- [`src/applications/modules/powerflow/pf_screen.hpp`](src/applications/modules/powerflow/pf_screen.hpp)

**What changed**

`N1ConnectivityScreen` used Union-Find once per candidate line outage.

**Reasoning**

An islanding case cannot safely share the ordinary connected-system structure.
Classifying it before an AC solve could avoid expensive stateful topology work.

**Actual effect**

The complete screen was approximately \(O(E(V+E))\), usually close to
\(O(E^2)\) for these sparse networks, and it was not connected to production
routing. It was an idea and a unit-testable component, not yet a speedup.

#### Result of the first milestone

This commit proved that GridPACK could build and run with an opt-in cuDSS
backend while retaining PETSc. It also disproved the simplest performance
hypothesis: replacing KLU with a cold per-contingency GPU solve was slower.

---

### 4.3 `31a98d56` — real wave integration, direct assembly, and per-case factor reuse

Commit:
[`31a98d56`](https://github.com/alexluhuang/GriDSSPack/commit/31a98d56d40f9e02bb16cdeba371f78d21ac7d0d),
2026-07-18.

This commit added the core ideas that made the GPU path useful. It also records
several designs that failed and were replaced before commit.

#### Change 1: expose and preserve per-case voltage state

**Where**

- [`src/applications/components/pf_matrix/pf_components.hpp`](src/applications/components/pf_matrix/pf_components.hpp)
  — `PFBus::setVoltageState` and `getVoltageState`
- [`src/applications/modules/powerflow/pf_app_module.hpp`](src/applications/modules/powerflow/pf_app_module.hpp)
  — network/factory access and convergence-state support

**What changed**

The wave engine can save the voltage magnitude and angle for one case, restore
another case, and later overlay a converged case for normal output.

**Reasoning**

One rank owns one mutable `PFNetwork`, but a wave represents several independent
outages. Their iterates must never be confused. Explicit snapshots provide the
minimum state isolation without cloning the full network once per contingency.

**Speed effect**

Saving compact voltage arrays is cheaper than copying or reconstructing an
entire component graph.

**Accuracy effect**

Correct state ownership prevents one contingency's iterate or branch status
from contaminating another.

#### Change 2: concrete `GridpackBatchAssembler`

**Where**

- New
  [`src/applications/modules/powerflow/pf_batch_ca_assembler.hpp`](src/applications/modules/powerflow/pf_batch_ca_assembler.hpp)

**What changed**

The assembler:

- extracts the base reduced Jacobian pattern;
- records each bus's reduced equation size;
- stores the base voltage state;
- resolves each outage to circuit objects;
- captures circuit status **before** applying the outage;
- classifies structurally compatible branch contingencies;
- precomputes component-block-to-CSR destinations;
- assembles Jacobian and mismatch arrays directly;
- updates only the outaged circuit's Y-bus contribution and endpoint terms;
- saves each case's iterate for later output.

**Reasoning**

The general GridPACK mapper is designed for flexible distributed assembly. With
`groupSize=1`, a rank has the complete network and repeated N-1 cases use the
same reduced pattern. Rediscovering locations and transporting small blocks
through Global Arrays and PETSc adds cost without adding information.

**Speed effect**

Direct scatter removes repeated general mapping. Local Y-bus repair avoids a
complete network rebuild for a one-circuit change. Keeping one case live avoids
restore/reapply work between every assembly and update.

**Accuracy effect**

The component formulas remain GridPACK's `PFBus` and `PFBranch` formulas.
The optimization changes placement and cache maintenance, not equations.
Capturing `baseStatus` before `setContingency` is crucial: an earlier integration
attempt captured the already-open status, so “restoration” left prior outages
open and failures began around the third case.

#### Change 3: abandon lockstep and native cuDSS batch

**Where**

- [`src/applications/modules/powerflow/pf_batch_ca.hpp`](src/applications/modules/powerflow/pf_batch_ca.hpp)
- [`src/math/cudss/cudss_batched_solver.hpp`](src/math/cudss/cudss_batched_solver.hpp)

**What changed**

The engine stopped advancing all cases in lockstep. It now keeps one case live
through its nonlinear iterations, then advances the next. The native cuDSS batch
matrices were replaced with one ordinary CSR workspace reused sequentially.

**Why the first design failed**

The shared `PFNetwork` was mutated by other cases between one case's assembly
and update. Restoring and toggling many cases around a lockstep loop corrupted
the state. Independently, the native multi-matrix cuDSS path produced NaNs on
the Texas-scale problem.

**Speed effect**

Sequential case advancement gives up simultaneous multi-matrix execution but
still amortizes structural analysis across a shared pattern. It also enables the
direct live assembler. In this workload, correct sequential reuse was faster
than an unstable “more parallel” design.

**Accuracy effect**

Each nonlinear trajectory is isolated. Non-finite linear solutions cause the
case to fail the wave and return to the established path.

> **Meaning of “batch”:** at this checkpoint, a batch is a setup/reuse group,
> not several matrices solved simultaneously in one native cuDSS call. The
> driver does not enforce bounded waves until `d939666f`.

#### Change 4: contingency-specific chord Newton

**Where**

- `PFBatchNR::solveWave` in
  [`pf_batch_ca.hpp`](src/applications/modules/powerflow/pf_batch_ca.hpp)
- `factorizeValues` and `solveReuse` in
  [`cudss_batched_solver.hpp`](src/math/cudss/cudss_batched_solver.hpp)
- generic `refactorEvery` / `constantFactor` support in
  [`cudss_linear_solver_implementation.hpp`](src/math/cudss/cudss_linear_solver_implementation.hpp)

**What changed**

When `constantFactor=true`, the engine factors each eligible outage's first
Jacobian once and performs later steps as cheaper solves with new mismatch
vectors. Otherwise this checkpoint uses `solveOne` and refactorizes at each
Newton step.

**Reasoning**

A branch N-1 case warm-started from the solved base state is usually near its
new solution. Its first-outage Jacobian is often a useful fixed local model.
One factorization plus several triangular solves can be cheaper than three or
four factorization/solve pairs.

**Speed effect**

This is the main GPU arithmetic saving introduced here.

**Accuracy effect**

The true mismatch is recomputed after every step. A case that reaches tolerance
meets the same residual test for the same nonlinear equations; one that does not
is discarded and solved with exact CPU Newton.

**Rejected alternative**

One factorization of the intact base-network Jacobian was tried for all
contingencies. Most cases failed to converge because the fixed matrix did not
reflect the opened circuit. The committed constant-factor design factors each
contingency's own first Jacobian instead.

#### Change 5: hybrid CPU/GPU routing

**Where**

- Batch activation, eligibility, controller checks, and fallback in
  [`ca_driver.cpp`](src/applications/contingency_analysis/ca_driver.cpp)

**What changed**

The base solve, generator outages, islanding or structure-changing cases,
nonconverged cases, and cases needing Q-limit/shunt/LTC action remain on
PETSc/KLU. Area-interchange control disables the wave because no safe per-case
check exists.

**Reasoning**

KLU was faster for one-off solves at this size. cuDSS is useful only when
expensive setup/factor work is reused. The irregular tail should use the
algorithm best suited to it.

**Speed effect**

This avoids forcing every task through a slower scalar GPU path.

**Accuracy effect**

Controllers are outer nonlinear/discrete loops, not simple right-hand-side
changes. Re-solving those cases through the full path preserves their intended
behavior.

#### Change 6: bulk formatting and optional asynchronous writer

**Where**

- New
  [`src/applications/contingency_analysis/ca_async_writer.hpp`](src/applications/contingency_analysis/ca_async_writer.hpp)
- Output blocks in
  [`ca_driver.cpp`](src/applications/contingency_analysis/ca_driver.cpp)

**What changed**

Rows for one event are formatted into a large string block. With
`overlapIO=true`, one FIFO consumer thread per rank writes queued blocks while
the solver continues.

**Speed effect**

Bulk formatting avoids many small stream operations. Asynchronous mode can
overlap output with computation.

**Accuracy effect**

The queue preserves order within a rank and drains before close. Cross-rank
ordering was still determined by the later file-combination path.

**Limitation at this point**

The queue was optional, off by default, and unbounded. Rank 0 still performed
the final serial concatenation, so this commit did not remove the largest I/O
tail.

#### Measurements and lessons at this stage

- Texas-256, one rank: about 114.9 s CPU versus 86.4 s integrated GPU.
- Texas-64: about 30.7 s CPU versus 25.6 s GPU.
- A larger 24,251-bus/48k-reduced 64-case experiment: 116.0 s CPU versus
  131.4 s GPU.

These were subset or different-network experiments, not the required full
8,891-case, 20-rank benchmark. The larger case showed why network size alone
does not guarantee a GPU win: eligibility, fallback volume, output size, and
device contention matter.

#### Known safety gaps in this checkpoint

- `waveSize` existed in a configuration structure but the driver did not yet
  enforce bounded waves.
- The Union-Find screen was diagnostic rather than an active classifier.
- Fast-assembly validation checked too little, warned rather than failing
  closed, and later turned out not to cover the live production calls.
- A missing CSR destination could be silently skipped.
- 32-bit index conversion was not range checked.
- Batched convergence rows used zero placeholders for worst \(P/Q\) mismatch
  locations.
- Generic `refactorEvery` and the real wave's `constantFactor` behavior were not
  fully aligned.

This commit delivered the decisive ideas, not a finished production result.

---

### 4.4 `81d9015a` — linear-time bridge screening becomes actionable

Commit:
[`81d9015a`](https://github.com/alexluhuang/GriDSSPack/commit/81d9015a6c3569b45ebfcd9c767c7d6aca566c4f),
2026-07-27, with the candid subject `it doesnt work :(`.

#### Change 1: replace all-outage Union-Find with iterative Tarjan

**Where**

- `N1ConnectivityScreen::screenAllBranchOutages` in
  [`src/applications/modules/powerflow/pf_screen.hpp`](src/applications/modules/powerflow/pf_screen.hpp)

**What changed**

One iterative depth-first traversal computes discovery and low-link values and
marks all bridges. Unique edge IDs correctly represent parallel circuits.

**Speed effect**

The all-outage graph work changes from a repeated near-quadratic pass to
\(O(V+E)\).

**Accuracy effect**

For a valid active graph, the connectivity classification is exact. The change
does not approximate AC power flow; it only identifies cases that cannot use the
ordinary connected fast structure.

#### Change 2: use the screen during eligibility classification

**Where**

- `p_buildConnectivityScreen`, `BranchTog::localIndex`, and `prepare` in
  [`pf_batch_ca_assembler.hpp`](src/applications/modules/powerflow/pf_batch_ca_assembler.hpp)

**What changed**

The assembler maps each live circuit to a graph edge. A resolved simple
single-circuit nonbridge can bypass the stateful
`setContingency/getIslandCount/hasLoneBus` probe. Bridges and unknown/complex
events go to the general path.

**Speed effect**

Thousands of repeated topology mutations/checks can be avoided.

**Accuracy effect**

The shortcut is conservative for bridges and unresolved cases. Parallel edges
are distinguished correctly.

#### Change 3: repair tests and shorten Docker builds

**Where**

- [`src/math/test/pf_batch_ca_test.cpp`](src/math/test/pf_batch_ca_test.cpp)
- [`Dockerfile`](Dockerfile)

The synthetic assembler was updated for the live/chord interface. Docker set
`GRIDPACK_ENABLE_TESTS=OFF`, reducing build time but also meaning these tests
were no longer built in the production image.

#### Why this was an unstable checkpoint

The active graph was not yet restricted robustly to active/non-isolated buses,
the base graph was not required to be connected, endpoints lacked full
validation, the XML `screen` flag was still ignored, and screened cases relied
on a structural assumption that was not live-validated. A second diagnostic
graph pass also duplicated work. Later commits made the option real and then
hardened these assumptions.

---

### 4.5 Experiments between the screen checkpoint and the all-rank design

Several important experiments were not preserved as independent commits. They
are included because they explain later code, but their numbers are
session-history evidence rather than a Git-controlled ablation suite.

#### Experiment A: an invalid 78.77-second target

An early broker-associated run used `outputFormat=text` and did not emit the
required multi-gigabyte `csv_flat` result. The broker code was also not actually
integrated with the tested CA execution. Its 78.77-second time is not a valid
target for the full-output workload.

**Lesson:** benchmark the product contract, including required output, rather
than only the solver loop.

#### Experiment B: dedicated 19-CPU-plus-1-GPU broker

A proposed dedicated GPU owner/broker did not become a valid integrated
implementation. It reduced the CPU resources available to ordinary fallbacks,
complicated result ownership, and the tested form omitted required output.

**Lesson:** a special broker is useful only if dispatch, solve, fallback, and
output are all connected and measured. It was not retained.

#### Experiment C: the 0.84% improvement and the “446 eligible” count

The intermediate rank-owner test improved median time from about 93.06 s to
92.28 s—only 0.84%. The printed 446 eligible cases were one rank's local
reservation/share. They were not the number eligible in the global 8,891-task
stream.

The committed history contains no simple `world.rank()==0` guard that was later
deleted. Before bounded waves, ranks could greedily reserve their remaining
shares, and only rank 0 printed local statistics. That combination made local
counts look global and left an unbalanced execution shape.

**Lesson:** the problem was work ownership and accounting, not that only 446
physical contingencies could use the GPU. Later MPI-reduced counts found 7,099
initially eligible tasks globally.

#### Experiment D: wave-size search

Full-Texas session results were:

| Wave size | Time |
|---:|---:|
| 4 | 95.17 s |
| 8 | 92.28 s |
| 32 | 94.60 s |
| 256 | 113.70 s |

Four did not amortize setup as effectively. Large waves allowed a rank to hold
too much work and become the final straggler. Eight was the best tested balance.

#### Experiment E: first persistent assembler

An initial small-wave persistent-assembler attempt measured 92.74 s and was
reverted. Its state repair did not safely eliminate enough reconstruction work.
The later implementation persists only verified invariants and has an explicit
`beginWave` reset contract.

#### Experiment F: output strategies

- The direct object formatter was retained after it both accelerated output and
  exposed omitted parallel-circuit rows in the legacy parser.
- Shared append measured roughly 91–93 s in its early form and removed a serial
  merge that had cost about 24 s in one 97.35-second run.
- Whole-output buffered MPI-IO was schema-correct but used about 9 GB for
  Texas7k and measured about 96.26 s. It remained optional.

---

### 4.6 `d939666f` — bounded waves on every rank and complete output pipeline

Commit:
[`d939666f`](https://github.com/alexluhuang/GriDSSPack/commit/d939666fb28b08977506595d020004ef91dca6d9),
2026-07-27.

This was the first bounded repeated all-rank wave implementation with global
accounting and an accepted exhaustive-output run. Earlier parents already let
each process enter the batch path, but without this bounded execution shape.

#### Change 1: every rank repeatedly claims at most eight tasks

**Where**

- GPU configuration and the bounded task loop in
  [`src/applications/contingency_analysis/ca_driver.cpp`](src/applications/contingency_analysis/ca_driver.cpp)

**What changed**

Each of the 20 ranks:

1. asks the shared `TaskManager` for up to `waveSize` tasks;
2. classifies its local wave;
3. advances eligible cases through its own cuDSS context;
4. runs local CPU fallbacks;
5. returns to the shared queue for another wave.

There is no dedicated processing rank and no rank that merely relays work. The
20 ranks collectively inspect 100% of the task stream.

**Why size eight helps**

It amortizes rank-local setup without allowing a process to reserve a huge tail.
Frequent returns to the queue let faster ranks absorb more work. Generator and
fallback cases remain distributed instead of accumulating behind one GPU owner.

**Accuracy effect**

Scheduling changes task order and ownership, not equations. `task_id` remains
the event identity. Ineligible or failed cases are still solved normally.

#### Change 2: MPI-reduced global accounting

**Where**

- The `[GPU all-rank summary]` counters in
  [`ca_driver.cpp`](src/applications/contingency_analysis/ca_driver.cpp)

**What changed**

Local wave, inspected, eligible, direct-fallback, nonconvergence, and controller
counts are reduced to rank 0.

**Effect**

This does not accelerate a solve. It establishes that all tasks were inspected
and prevents one rank's local count from being mistaken for a global count.

A representative historical result was:

```text
inspected=8891
eligible=7099
direct_fallback=1792
nonconverged_fallback=279
controller_fallback=409
retained_gpu=6411
```

#### Change 3: GPU behavior becomes explicitly opt-in

**Where**

- Backend-selection block in
  [`ca_driver.cpp`](src/applications/contingency_analysis/ca_driver.cpp)

**What changed**

`<GPU><enabled>true</enabled>` became the master switch. With batched mode
active, the default solver intentionally remains PETSc so the base and fallback
tail use KLU; the wave invokes cuDSS directly.

**Speed effect**

It avoids a roughly 20-second cold cuDSS base solve that took only a few seconds
on KLU in the observed setup.

**Accuracy and usability effect**

The runtime mode is explicit. The same image and `ca.x` can run CPU-only when
the switch is false.

#### Change 4: direct object-to-CSV enumeration

**Where**

- `captureFlatRows` in
  [`ca_driver.cpp`](src/applications/contingency_analysis/ca_driver.cpp)

**What changed**

For `groupSize=1`, output walks every `PFBranch`, then every circuit returned by
`getLineIDs()`, reads endpoint voltage state and circuit flow/rating directly,
and writes the 15-column row.

The stock path instead:

1. produced formatted bus strings;
2. produced one branch-object string that could contain several circuit lines;
3. parsed those strings with `sscanf`;
4. formatted the values again as CSV.

**Speed effect**

Direct access removes Global Arrays string gathering and the
number-to-text-to-number-to-text round trip.

**Output-accuracy effect: why 4,268,160 more rows appeared**

One CSV row represents one:

```text
(event, from_bus, to_bus, circuit_id)
```

It does not represent one contingency by itself.

The Texas model has:

- 8,646 `PFBranch` objects seen by the legacy one-string/one-parse path;
- 9,140 distinct circuit keys found by enumerating every circuit;
- a difference of 494 circuits;
- 8,640 events that emit flat rows: the base plus 8,639 `OK` contingencies.

Therefore:

\[
(9{,}140-8{,}646)\times8{,}640
=494\times8{,}640
=4{,}268{,}160.
\]

The legacy branch string could contain more than one newline record for
parallel/additional circuits, but `captureFlatRows` performed only one `sscanf`
for that string and consumed only its first record. GriDSSPack's extra rows are
previously omitted physical circuit keys, not duplicate GPU calculations and
not 4.27 million additional contingencies.

At this historical commit the new formatter still had short fixed name/row
buffers, no general contingency-name escaping, and inconsistent rating use for
`viol`. Commit `67c80996` repaired those points. Circuit identifiers are still
assumed to be CSV-safe identifiers, as noted under current limitations.

#### Change 5: shared flat-file append

**Where**

- Output initialization/finalization in
  [`ca_driver.cpp`](src/applications/contingency_analysis/ca_driver.cpp)
- Append support in
  [`ca_async_writer.hpp`](src/applications/contingency_analysis/ca_async_writer.hpp)

**What changed**

The default sequence is:

1. rank 0 truncates `<outputFile>_flat.csv` and writes one header;
2. all ranks synchronize;
3. every rank lazily opens that same path with binary append mode;
4. one event's rows are formatted as one complete block;
5. the rank writes the block synchronously, or one local FIFO thread writes it
   when `overlapIO=true`;
6. writers drain and close;
7. no rank-0 flat-file concatenation runs.

**Speed effect**

The old final rank-0 read/rewrite of roughly 8–9 GB disappears. Output work is
distributed while computation is still distributed.

**Ordering and schema**

Rows remain FIFO-ordered within one rank but are nondeterministically interleaved
among ranks. `event_idx` and the circuit key provide stable identity, so
comparison does not depend on file order.

**Portability caveat**

The implementation uses C++ `std::ios::app`, not a collective atomic-record
protocol. On the validated local Linux filesystem, large event blocks were
observed as intact append writes and full scans found no malformed rows.
The C++ interface does not guarantee that one large insertion is one indivisible
filesystem write, and NFS or some parallel filesystems may not provide the same
behavior. `bufferFlatOutput=true` uses disjoint MPI-IO offsets but stores the
whole rank output in memory.

#### Change 6: make screen/output options operational

**Where**

- `GPU/screen` and `GPU/waveSize` parsing in `ca_driver.cpp`
- screen gating and reduced restore work in
  [`pf_batch_ca_assembler.hpp`](src/applications/modules/powerflow/pf_batch_ca_assembler.hpp)
- `PFAppModule::suppressOutput`

**What changed**

The screen flag now reaches the assembler. Screened simple cases skip a redundant
full voltage restore during classification. Unrequested calculation printing is
suppressed. `csv_flat` becomes the code default so benchmark runs include the
full result contract unless explicitly changed.

**Speed effect**

This removes topology/state and console work. Changing a default to `csv_flat`
does not itself make a run faster; it makes comparisons harder to game by
silently omitting output.

#### Change 7: graph regression tests

**Where**

- [`src/math/test/pf_screen_test.cpp`](src/math/test/pf_screen_test.cpp)
- [`src/math/CMakeLists.txt`](src/math/CMakeLists.txt)

Tests cover a cycle with a bridge tail, parallel circuits, and a 100,000-bus
chain. They add no runtime speed but protect the linear-time and nonrecursive
properties.

#### Full-scenario scaling and attribution at this milestone

With wave size eight and exhaustive Texas output:

| Ranks | Runtime | Inspected | Initially eligible |
|---:|---:|---:|---:|
| 1 | 1,108.70 s | 8,891 | 7,099 |
| 4 | 317.59 s | 8,891 | 7,099 |
| 8 | 167.45 s | 8,891 | 7,099 |
| 16 | 98.08 s | 8,891 | 7,099 |
| 20 | 86.26 s | 8,891 | 7,099 |

Twenty cuDSS contexts contend for one GB10, but the full application still
benefited from 20-way task and fallback parallelism.

The most useful four-mode comparison at this point was approximately:

| Mode | Time |
|---|---:|
| Stock CPU image | 217.70 s |
| GriDSSPack optimized CPU | 93.06 s |
| Intermediate rank-owner GPU | 92.28 s |
| All-rank GPU | 86.26 s |

The all-rank GPU path was about 7.31% faster than the optimized CPU result.
That is the honest GPU-era incremental result for this milestone. The much
larger stock-to-GriDSSPack difference came mostly from the release build,
assembly/output improvements, suppressed work, and removal of serial output
concatenation.

---

### 4.7 `ca8c245d` — Docker packaging and user documentation

Commit:
[`ca8c245d`](https://github.com/alexluhuang/GriDSSPack/commit/ca8c245ddf3984398e2e48c4370cb0f7a5defe04),
2026-07-28.

**Where**

- [`.dockerignore`](.dockerignore)
- [`Dockerfile`](Dockerfile)
- [`DOCKER_CA.md`](DOCKER_CA.md)
- [`README.md`](README.md)

**What changed**

- Added image labels/revision information.
- Added `ca-certificates` and removed apt metadata.
- Used `cmake --build --parallel 10`.
- Excluded generated build/output data from the Docker context.
- Added complete build, XML, CPU/GPU run, and verification instructions.
- Preserved the normal in-container command:

  ```text
  mpirun -n 20 ca.x input.xml
  ```

**Effect**

Docker builds became faster, smaller, and easier to reproduce. The installed
solver's equations, accuracy, scheduling, and runtime performance did not change
in this commit.

The documented validation reported about 84.745 s maximum application time,
8,891 inspected tasks, wave eight, and all 20 ranks receiving work. A retained
count of 6,414 in that run differs slightly from the earlier 6,411 snapshot;
eligibility remained 7,099, while fallback counts can vary with the exact
checkpoint/run.

---

### 4.8 `5d66fd83` — out-of-core accuracy comparison suite

Commit:
[`5d66fd83`](https://github.com/alexluhuang/GriDSSPack/commit/5d66fd83fe7314d6af5e9cd573408c1b54cd336a),
2026-07-28.

**Where**

- [`gpucputest/compare_results.py`](gpucputest/compare_results.py)
- [`gpucputest/run_comparison.sh`](gpucputest/run_comparison.sh)
- [`gpucputest/Dockerfile`](gpucputest/Dockerfile)
- [`gpucputest/README.md`](gpucputest/README.md)

**What changed**

The comparison tool converts multi-gigabyte CSV data into event-bucketed
Parquet with Dask-cuDF, then compares bounded buckets on the GPU. It checks:

- unique `(event, from, to, circuit)` keys and duplicate rejection;
- row coverage separately from numerical agreement;
- mean absolute difference, RMSE, bias, percentage metrics, and maxima;
- circular bus-angle difference;
- convergence outcomes and status transitions;
- bus metadata;
- application timing and rank load balance.

**Speed effect**

None on `ca.x`. It accelerates and makes feasible the verification of tens of
millions of result rows.

**Accuracy effect**

It changes confidence, not solver arithmetic. It can distinguish a small
floating-point difference from a missing circuit, duplicate row, schema drift,
or different convergence outcome.

---

### 4.9 `570bedb8` — committed Texas GPU/stock evidence

Commit:
[`570bedb8`](https://github.com/alexluhuang/GriDSSPack/commit/570bedb8251756590e1cee314c171855af76e610),
2026-07-28.

**Where**

- [`gpucputest/results/summary.md`](gpucputest/results/summary.md)
- [`gpucputest/results/comparison_report.json`](gpucputest/results/comparison_report.json)
- supporting result CSVs under [`gpucputest/results/`](gpucputest/results/)

**What changed**

Only evidence files were added. No runtime source changed.

**Stored performance**

| Metric | GriDSSPack GPU | Stock CPU |
|---|---:|---:|
| Maximum application time | 84.5626 s | 214.1355 s |
| Mean tasks/rank | 444.55 | 444.55 |
| Task-count coefficient of variation | 3.884% | 17.278% |

The reported end-to-end speedup was 2.532×.

**Stored convergence**

- 8,887 cases converged in both outputs.
- 8,639 were `OK`.
- 166 were `SLACK_OVERLOAD`.
- 82 were `ISLANDED`.
- Four were `DIVERGED`.
- Outcome agreement was 100%.

**Stored row coverage**

| Dataset | GriDSSPack | Stock | Difference |
|---|---:|---:|---:|
| Flat rows | 78,969,600 | 74,701,440 | +4,268,160 GriDSSPack |
| Convergence rows | 8,891 | 8,891 | 0 |
| Duplicate flat keys | 0 | 0 | 0 |

The coverage difference is exactly the 494 previously omitted circuit keys
across 8,640 row-emitting events explained above.

**Stored numerical result**

Matched rows had very small mean differences, but not zero and not uniformly
small maxima:

| Quantity | Mean absolute difference | Maximum absolute difference |
|---|---:|---:|
| Real power | \(6.97\times10^{-5}\) MW | 0.8464 MW |
| Reactive power | \(4.94\times10^{-4}\) MVAr | 40.5746 MVAr |
| MVA | \(1.71\times10^{-4}\) MVA | 10.2524 MVA |
| From-bus voltage | \(6.17\times10^{-7}\) p.u. | 0.014437 p.u. |
| To-bus voltage | \(6.13\times10^{-7}\) p.u. | 0.01549 p.u. |
| From angle | \(6.05\times10^{-5}\) degrees | 0.1718 degrees |
| To angle | \(6.03\times10^{-5}\) degrees | 0.1828 degrees |

There were 3,361 violation-flag disagreements, about 0.004499% of matched rows.
This result supported outcome-level agreement but did not prove bitwise
identity. More importantly, it exposed reporting and validation weaknesses that
prompted the next commit.

---

### 4.10 `67c80996` — correctness and performance hardening

Commit:
[`67c80996`](https://github.com/alexluhuang/GriDSSPack/commit/67c80996089afcaf5b7249200b322ab9b999a8a9),
2026-07-29.

This commit added an adversarial review and then changed the implementation in
the areas the review exposed. The tracked
[`CODE_REVIEW_GPU_CA.md`](CODE_REVIEW_GPU_CA.md) should be read as a review of
the pre-hardening `10e5cf26..570bedb8` state, not as a list of bugs still
present at current HEAD.

#### Change 1: make the fast-assembly correctness oracles live

**Where**

- Constructor flags, `assembleLive`, `updateLive`, `updateLiveRhs`,
  `assembleLiveJac`, and `p_validateLiveAssembly` in
  [`pf_batch_ca_assembler.hpp`](src/applications/modules/powerflow/pf_batch_ca_assembler.hpp)

**Problem before**

`GRIDPACK_BATCH_VALIDATE` and `GRIDPACK_BATCH_NOFAST` affected old
`assemble`/`update` methods, while the real sequential wave called the `Live`
methods. Validation therefore did not protect the production path, and
`NOFAST` could leave an unusable scatter map rather than provide the promised
reference mode.

**What changed**

- `GRIDPACK_BATCH_VALIDATE=1` now compares every requested live Jacobian/RHS
  with the canonical GridPACK mapper.
- It checks dimensions, row offsets, columns, and numeric values, requiring the
  maximum difference to be no more than \(10^{-12}\) times the reference scale.
- A mismatch throws instead of merely printing.
- `GRIDPACK_BATCH_NOFAST=1` routes tasks to the established nonfast/CPU path
  instead of attempting direct scatter.

**Accuracy effect**

The direct assembler is now continuously falsifiable. An assumption failure
becomes a fallback/error rather than a silent wrong result.

**Speed effect**

Validation is deliberately expensive and opt-in. Normal mode pays only the
lightweight flag branch.

#### Change 2: enforce sparse-matrix invariants

**Where**

- CSR construction/scatter checks in
  [`pf_batch_ca_assembler.hpp`](src/applications/modules/powerflow/pf_batch_ca_assembler.hpp)
- input validation in
  [`cudss_batched_solver.hpp`](src/math/cudss/cudss_batched_solver.hpp)

**What changed**

The code validates positive dimensions, 32-bit representability, CSR endpoint
rules, monotone row offsets, column bounds, and component-block coverage.
`p_findSlot` throws when a required matrix position is missing instead of
returning `-1` and allowing a silent skip.

**Accuracy and reliability effect**

Direct assembly is safe only if every component value has a valid destination.
These checks turn structural corruption, overflow, or an unsupported pattern
into an explicit fallback/failure.

**Speed effect**

Most checks occur during reusable setup, not every arithmetic operation.

#### Change 3: build a trustworthy active connectivity graph

**Where**

- input validation in
  [`pf_screen.hpp`](src/applications/modules/powerflow/pf_screen.hpp)
- `p_buildConnectivityScreen` in
  [`pf_batch_ca_assembler.hpp`](src/applications/modules/powerflow/pf_batch_ca_assembler.hpp)

**What changed**

Only active, non-isolated buses and in-service circuits enter the compact graph.
Endpoints are validated. The shortcut is used only if the active base graph is
connected; otherwise cases use the authoritative topology path.

**Accuracy effect**

Tarjan's algorithm is exact for any well-formed input graph. Application
correctness additionally requires that graph to represent the active electrical
network faithfully. Constructing the right graph and checking the connected-base
premise closes that application gap.

#### Change 4: adaptive chord refactorization

**Where**

- `PFBatchNR::solveWave` in
  [`pf_batch_ca.hpp`](src/applications/modules/powerflow/pf_batch_ca.hpp)
- `assembleLiveJac` and true mismatch capture in
  [`pf_batch_ca_assembler.hpp`](src/applications/modules/powerflow/pf_batch_ca_assembler.hpp)

**Problem before**

A fixed first Jacobian worked for most cases but historically sent 279 cases to
nonconvergence fallback. Some could have converged cheaply after one refreshed
local model.

**What changed**

After each chord step the engine computes:

\[
\rho=\frac{\lVert F(x_{k+1})\rVert_\infty}
           {\lVert F(x_k)\rVert_\infty}.
\]

Progress is considered poor when:

\[
\rho\ge\max(0.5,\;1-0.5d),
\]

where \(d\) is the damping factor. Poor progress or the configured refresh
interval triggers assembly and factorization of the current live Jacobian.
At most two adaptive poor-progress refreshes are attempted by default. Poor
progress immediately after a refresh, non-finite values, a refresh failure, or
the iteration cap sends the case to exact CPU Newton.

**Speed effect**

It keeps most of the triangular-solve savings while rescuing cases that need
one or two better local models. Session counts improved from 279 historical
nonconvergence fallbacks to 3, with 198 adaptive refactorizations.

**Accuracy effect**

Acceptance still uses the true nonlinear mismatch. The algorithm spends more
work when needed and fails over when that work does not justify trust.

#### Change 5: report real mismatch diagnostics

**Where**

- `BatchMismatchInfo`, RHS extraction, and convergence-row population in
  [`pf_batch_ca.hpp`](src/applications/modules/powerflow/pf_batch_ca.hpp),
  [`pf_batch_ca_assembler.hpp`](src/applications/modules/powerflow/pf_batch_ca_assembler.hpp),
  and [`ca_driver.cpp`](src/applications/contingency_analysis/ca_driver.cpp)

**What changed**

The fast RHS path records the bus and magnitude of the largest real- and
reactive-power mismatches. Convergence sidecars no longer contain historical
zero placeholders for these fields.

**Effect**

Voltage solutions are unchanged. Diagnostic accuracy and comparison integrity
improve.

#### Change 6: strict and bounded wave parsing

**Where**

- GPU `waveSize` parsing and memory cap in
  [`ca_driver.cpp`](src/applications/contingency_analysis/ca_driver.cpp)

**What changed**

`strtol` with complete validation replaces `atoi`. Invalid, nonpositive, or
overflowing values throw. `auto` retains the default eight. The effective wave
is capped by both:

- 256 cases; and
- 256 MiB per rank for two `double` state arrays per bus per case.

**Speed/reliability effect**

A typo can no longer silently become eight, and an enormous value cannot
create unbounded per-rank state: values above the effective memory/256-case cap
are reduced. An explicit wave size of 256 remains legal, so the cap alone does
not prevent the known wave-256 load-balance regression.

#### Change 7: safely persist the assembler across waves

**Where**

- one-time rank-local construction in
  [`ca_driver.cpp`](src/applications/contingency_analysis/ca_driver.cpp)
- `beginWave` and base-state repair in
  [`pf_batch_ca_assembler.hpp`](src/applications/modules/powerflow/pf_batch_ca_assembler.hpp)

**What changed**

Each rank now constructs its expensive base mapper, CSR pattern, scatter map,
and connectivity screen once. `beginWave` restores branch status, base voltage,
Y-bus/S-bus state, signatures, counters, and wave-local storage before accepting
new task IDs.

**Theory**

These objects are structural invariants of the network, not properties of one
eight-case wave. They can be reused only if every mutable case effect is
explicitly repaired.

**Speed effect**

Session profiling reduced the mapping/setup portion from about 4.94 s to
1.68 s. This is the successful persistent design that replaced the reverted
92.74-second experiment.

**Accuracy effect**

`beginWave` makes the restoration contract explicit. Exceptions mark the rank's
batch path unhealthy and move remaining work to exact CPU processing.

#### Change 8: isolate retained GPU output from CPU fallbacks

**Where**

- retained/fallback ordering and cleanup in
  [`ca_driver.cpp`](src/applications/contingency_analysis/ca_driver.cpp)
- `applyCaseForOutput`, `clearCaseForOutput`, and `restoreBaseState` in
  [`pf_batch_ca_assembler.hpp`](src/applications/modules/powerflow/pf_batch_ca_assembler.hpp)

**What changed**

Retained GPU cases are overlaid and emitted before CPU fallback cases can
rebuild global Y-bus/S-bus caches. A local branch/endpoint refresh is enough for
ordinary output and Q-limit checks; shunt/LTC checks request a full refresh.
Every overlay has matching cleanup, including exception paths.

**Accuracy effect**

One fallback can no longer contaminate cached data used to report an already
converged GPU case.

**Speed effect**

The common output path uses local refresh while preserving a full-refresh route
when controller semantics require it.

#### Change 9: exact cuDSS pattern cache and safer resource cleanup

**Where**

- [`cudss_linear_solver_implementation.hpp`](src/math/cudss/cudss_linear_solver_implementation.hpp)
- [`cudss_batched_solver.hpp`](src/math/cudss/cudss_batched_solver.hpp)

**What changed**

The generic solver now caches and compares complete `rowptr` and `colind`
arrays, not only \(n\) and `nnz`. Matrix wrappers and device allocations are
released per pointer so partial construction failures do not leak resources. In
the wave solver, a factorization/solve failure returns failure or a zeroed
correction; the caller falls back and the per-wave solver is then destroyed.

**Accuracy/reliability effect**

Symbolic analysis is reused only for the exact pattern it analyzed. This applies
the sparse-reuse rule from Section 3 rather than the unsafe “same size means
same structure” shortcut.

#### Change 10: harden CSV semantics

**Where**

- direct formatter helpers in
  [`ca_driver.cpp`](src/applications/contingency_analysis/ca_driver.cpp)

**What changed**

- Full contingency names are retained.
- Contingency names containing commas, quotes, or line endings are CSV-escaped.
- Row buffers grow and retry rather than truncating.
- Circuit IDs are retained as strings by the direct row builder rather than
  being parsed through the former fixed branch-string field; they are not
  independently CSV-escaped.
- On contingency rows, `rate_mva`, `loading_percent`, and `viol` all use the
  same selected A/B/C rating with documented fallback; base rows use A.

**Accuracy effect**

This fixes an internal inconsistency in which displayed selected rating/loading
could disagree with a rate-A-based `viol` flag, and it protects contingency-name
schema. The historical 3,361 GPU/stock flag differences cannot be attributed
solely to this edit because both historical paths used rate A for that flag. The
zero current CPU/GPU flag mismatch is a later full-comparison result, not a
single-change ablation. These edits do not change the solved voltage state.

#### Change 11: aggregate observability and regression coverage

**Where**

- global summary/timers in `ca_driver.cpp`
- expanded
  [`pf_batch_ca_test.cpp`](src/math/test/pf_batch_ca_test.cpp) and
  [`pf_screen_test.cpp`](src/math/test/pf_screen_test.cpp)

Tests now exercise adaptive stalling/refactorization, invalid graph inputs, and
fast-path assumptions. Logs report adaptive refactors, avoided screen checks,
and bridge counts globally rather than relying on rank-local impressions.

#### Current full-scenario result after hardening

Current-session measurements on the same Texas7k, 8,891-contingency, 20-rank,
wave-eight, exhaustive-output workload were:

| Mode | Time | Relative to stock | Relative to optimized CPU |
|---|---:|---:|---:|
| Stock CPU | 214.136 s | 1.00× | — |
| Current optimized CPU | 89.922 s | 2.38× | 1.00× |
| Current GPU | 68.260 s | 3.14× | 1.32× |

The current GPU summary was:

```text
inspected=8891
eligible=7099
direct_fallback=1792
adaptive_refactors=198
nonconverged_fallback=3
controller_fallback=457
retained_gpu=6639
```

Current CPU/GPU comparison found:

- 78,969,600 flat rows in each;
- 8,891 convergence rows in each;
- zero unmatched row keys;
- zero convergence-status differences;
- zero violation-flag differences;
- 8,887 converged and four diverged in both.

Mean absolute CPU/GPU differences for matched current rows were approximately:

- \(5.81\times10^{-6}\) MW in real power;
- \(1.28\times10^{-5}\) MVAr in reactive power;
- \(1.3\times10^{-8}\) p.u. in voltage;
- \(1.48\times10^{-5}\) degrees in angle.

These are session results rather than files committed under
`gpucputest/results/`. They should be archived before being treated as a
permanent release record.

---

### 4.11 `f0a97125` — change the comparison script's direct default dataset

Commit:
[`f0a97125`](https://github.com/alexluhuang/GriDSSPack/commit/f0a97125ee7af7071e0f444937640a2424fb546c),
2026-07-29.

**Where**

- [`gpucputest/compare_results.py`](gpucputest/compare_results.py)

**What changed**

The direct Python invocation's `DEFAULT_DATA_DIR` changed from the Texas
verification directory to `verification_gpu_training`.

**Effect**

There is no simulation, solver, accuracy, Docker-image, or `ca.x` speed change.
It only changes where the comparison utility looks when no path is supplied.

**Current maintenance caveat**

[`gpucputest/run_comparison.sh`](gpucputest/run_comparison.sh) still defaults its
host argument to `verification_gpu_7k` and passes `/data` explicitly inside the
container. The wrapper and direct Python defaults therefore differ. Explicit
paths remain unambiguous, but the defaults should eventually be reconciled.

## 5. How the current system works end to end

The current runtime is easiest to understand as a pipeline. “All-rank GPU” does
not mean every operation runs on the GPU. It means every MPI rank is allowed to
use its own cuDSS context for eligible work while continuing to perform CPU
assembly, controller checks, fallbacks, and output.

### 5.1 Startup and base solution

1. Every rank reads the XML and builds a complete rank-local network because
   `groupSize=1`.
2. The explicit CA-level GPU switch is read.
3. In batched mode the default GridPACK solver remains PETSc/KLU.
4. The intact base power flow is solved on CPU.
5. Base bus voltage, component structure, and output metadata are captured.

Keeping the base solve on CPU avoids cold cuDSS overhead for a single system.

### 5.2 One-time rank-local fast-path setup

Each rank constructs one `GridpackBatchAssembler`:

- canonical reduced Jacobian and CSR pattern;
- bus/component equation sizes;
- component-to-CSR scatter destinations;
- base voltage and branch status;
- active-graph Tarjan bridge map;
- local update metadata.

This setup is reused across that rank's later waves.

### 5.3 Dynamic bounded work acquisition

All ranks draw from the same `TaskManager`. Each rank reserves at most eight
tasks. Since there are 20 ranks, up to roughly 160 tasks can be owned across the
machine at a time, but each rank's state and solver remain independent.

The rank does not send its tasks to another process. It classifies and processes
them locally, then returns to the queue.

### 5.4 Eligibility and safe routing

The fast wave accepts structurally compatible branch outages. A task uses the
general path when it is:

- a generator contingency;
- unresolved or more complex than the supported simple branch case;
- a bridge/islanding case;
- structurally different;
- subject to unsupported area-interchange behavior;
- unable to converge under bounded chord/adaptive work;
- found to require a Q-limit, shunt, or LTC controller action;
- affected by a setup, validation, cuDSS, or cleanup exception.

This is a **fail-closed** design: uncertainty reduces speed for that case rather
than relaxing correctness.

### 5.5 Eligible case solve

For one eligible case:

1. Restore that case's warm-start voltage.
2. Remove its circuit using the local Y-bus update.
3. Evaluate the true mismatch and first Jacobian directly into the verified CSR
   pattern.
4. Numerically factor that case's Jacobian with cuDSS.
5. Solve for a correction with triangular solves.
6. Apply damping and update voltage.
7. Recompute the true AC mismatch.
8. Reuse factors while progress is satisfactory.
9. Reassemble/refactor at the current state if progress becomes poor.
10. Accept only if the true mismatch meets tolerance; otherwise mark it for CPU
    exact Newton.
11. Save the converged voltage state and restore the circuit.

Cases within the wave follow this sequence one after another. The wave shares
structural setup and cuDSS analysis; it does not execute eight nonlinear cases
simultaneously.

### 5.6 Controller check and output overlay

For a converged fast case, the driver overlays its saved voltage and outage on
the rank-local network. It asks whether Q limits or enabled controllers would
act. If so, the overlay is discarded and the full CPU controller loop solves
the case.

Retained GPU cases are output before fallback cases. That ordering is internal
to one wave and prevents a CPU fallback's cache rebuild from affecting a saved
GPU result.

### 5.7 Flat-file write and completion

The direct formatter enumerates every circuit and generates one complete event
block. The rank appends that block to the shared final file. After all waves:

- writers drain and close;
- flat-row counts are summed;
- convergence and bus sidecars are finalized;
- global GPU/fallback counters are reported;
- no rank-0 flat-file concatenation is needed.

The logical output order is given by keys, not physical file order.

## 6. Why the complete speedup has several sources

A useful conceptual cost model is:

\[
T_{\text{application}} =
T_{\text{startup}}+
T_{\text{classify}}+
T_{\text{assemble}}+
T_{\text{factor}}+
T_{\text{triangular solve}}+
T_{\text{controllers}}+
T_{\text{format}}+
T_{\text{write}}+
T_{\text{serial tail}}.
\]

A GPU sparse solver directly attacks only part of this sum. This is an example
of Amdahl's law: accelerating one fraction cannot overcome time left in all
other fractions.

### 6.1 Stock-to-optimized-CPU savings

The optimized CPU result benefits from changes that do not require cuDSS:

- Release rather than Debug compilation;
- suppression of unused per-case diagnostic printing;
- direct circuit-to-CSV enumeration for `groupSize=1`;
- bulk row formatting;
- shared final-file append;
- removal of rank-0 flat-file concatenation;
- more complete, internally consistent output logic.

The current CPU mode does **not** use the GPU wave's direct Jacobian assembler,
Tarjan eligibility shortcut, or chord factors because `GPU/enabled=false`
routes tasks through the ordinary contingency solve.

### 6.2 Optimized-CPU-to-GPU savings

The GPU mode adds:

- verified direct component-to-CSR Jacobian/RHS assembly;
- local circuit/Y-bus updates;
- one cuDSS structural analysis per rank-local wave instead of a cold analysis
  for each scalar solve, while the host-side CSR/scatter structure persists
  across waves;
- one case-specific factorization followed by multiple triangular solves;
- adaptive rather than immediate CPU fallback for slow chord cases;
- one persistent assembler per rank;
- Tarjan-based avoidance of repeated topology probes;
- bounded all-rank work distribution.

That combined layer reduced 89.922 s to 68.260 s in the current session.

### 6.3 Why no exact per-change seconds are claimed

Many mechanisms interact:

- Direct assembly makes chord iteration cheap enough to be useful.
- Chord iteration changes which cases fall back.
- Adaptive refresh changes both GPU work and CPU tail size.
- Wave size changes load balance, setup frequency, and memory.
- Output occurs while ranks finish at different times.

Only source-isolated ablation builds can attribute exact seconds. Existing logs
support some component observations—such as the 24-second historical merge tail
and persistent-mapping reduction—but not a unique additive decomposition of the
68.260-second result.

## 7. Speed and accuracy effect of each optimization class

| Change | Why it can be faster | What happens to numerical accuracy | Principal safeguard |
|---|---|---|---|
| Release build | Removes debug optimization barriers across CPU code | Same algorithms; floating-point order can differ | Full result comparison |
| cuDSS backend | GPU sparse factor/solve | Same linear system in FP64, not bitwise guaranteed | Tolerance checks and PETSc fallback |
| Structural-analysis reuse | Avoids repeated ordering/symbolic work | Reuse is valid only for the same CSR pattern | Full `rowptr`/`colind` comparison |
| Direct component-to-CSR | Avoids GA/PETSc mapping overhead | Same component formulas and destinations | Live canonical-mapper validation |
| Local Y-bus update | Touches only the changed circuit/endpoints | Exact for supported simple circuit outage | Eligibility limits; full fallback |
| Tarjan bridge screen | Finds all bridge outages in one linear pass | Same islanding answer as repeated tests for a connected active graph and one-edge outage | Active graph validation and conservative routing |
| Warm start | Starts near the new N-1 solution | Does not change equations/tolerance, but can influence the convergence basin in a multi-root problem | True mismatch convergence test |
| Chord factor reuse | Replaces refactorizations with triangular solves | Tests the same equations/tolerance; may follow a different trajectory or root | True mismatch, adaptive refresh, CPU exact fallback |
| Adaptive refactor | Rescues weak chord cases without immediate CPU restart | Uses fresher Jacobian; improves robustness | At most bounded refreshes, then fallback |
| Wave size eight | Amortizes setup while preserving load balance | Only task order/ownership changes | Stable event IDs and global accounting |
| All-rank GPU | Uses all processes for local GPU/CPU work | No equation change | Rank-local state and MPI-reduced counts |
| Persistent assembler | Reuses immutable maps across waves | Safe only after complete state repair | `beginWave`, exception fallback, validation |
| Direct CSV enumeration | Avoids text parse/reformat work | Improves circuit coverage | Key/duplicate/schema comparison |
| Shared append | Removes serial full-file rewrite | Values unchanged; physical order changes | Event/circuit keys and full-file scan |
| Async writer | Can overlap a rank's disk write with solve | Values unchanged; rank interleaving remains nondeterministic | Single local FIFO and drain-on-close |
| Buffered MPI-IO option | Gives disjoint explicit offsets | Values unchanged | Collective offsets; high-memory caveat |
| Comparison suite | Does not accelerate `ca.x` | Improves detection of missing/different results | Coverage, status, numeric, and schema checks |

The only deliberate numerical convergence tradeoff is chord Newton. The other
runtime optimizations either change work placement, avoid repeated exact work,
or choose among exact connectivity/output paths.

## 8. Consolidated record of approaches that did not work

| Attempt | Observed problem | Decision and lesson |
|---|---|---|
| Cold per-contingency cuDSS replacement | Slower than KLU because every case paid setup, copies, and contention | Retain scalar backend as an option; use GPU only where setup/factors can be reused |
| Native cuDSS multi-matrix batch | Produced NaNs on the Texas-scale path | Replace with proven single-system CSR workspace and sequential cases |
| Lockstep cases on one mutable network | Cases changed shared topology/state between assembly and update | Keep one case live through its nonlinear loop |
| Capture branch status after applying outage | Prior outages accumulated; failures appeared around case three | Capture base status first and restore explicitly |
| One intact-base Jacobian for all contingencies | Most cases did not converge | Factor each outage's own first Jacobian |
| Fixed case Jacobian with no adaptive rescue | 279 historical nonconvergence fallbacks | Refresh on poor progress, with a strict bound |
| All-outage repeated Union-Find | Near-quadratic total work and initially diagnostic only | Use one iterative Tarjan bridge pass and make it actionable |
| Unwired/inert screen and validation flags | Logs/options implied protection that production calls did not receive | Route flags through live methods and fail closed |
| Dedicated 19-CPU/1-GPU broker | Not integrated with complete task/output flow; sacrificed one general worker | Let every rank claim and process bounded waves locally |
| Treat rank 0's 446 eligible tasks as global | Local logging misrepresented the full stream | Add MPI-reduced global counters; full eligibility was 7,099 |
| Wave 256 | Rank hoarded work, used more state, and became a straggler at 113.70 s | Retain wave eight; cap size and memory |
| First persistent assembler | 92.74 s and insufficient safe reuse | Revert; later add explicit `beginWave` repair around verified invariants |
| Whole-output buffered MPI-IO default | Correct but about 9 GB memory and 96.26 s on Texas | Keep optional; shared append remains default on validated filesystem |
| Compare against a 78.77 s text-output run | Required exhaustive output was missing; broker was not connected | Reject as an invalid benchmark |
| Infer full-scale behavior from 64/256 cases | Direction and magnitude changed on full Texas and the larger training case | Require full N-1, full output, and fixed rank count for acceptance |

Some ideas were discussed but never implemented in the current path and should
not be reported as failed experiments: custom device-resident assembly kernels,
CUDA streams/graphs, DC/LODF prescreening, mixed precision, a learned cost
model, and a true fast-decoupled power-flow engine.

## 9. Current limitations and documentation debt

### 9.1 Shared append is validated, not universally portable

The default works on the tested local filesystem and produced intact
79-million-row files. It does not provide a formal cross-filesystem atomic-block
guarantee. A bounded incremental MPI-IO design would be more portable than
either uncoordinated append or whole-output buffering.

### 9.2 “Batch” is scheduling/reuse, not native simultaneous batch

The class name `CuDSSBatchedSolver` and some older comments can suggest one
eight-matrix factor/solve call. Current code processes cases sequentially and
shares analysis/setup. The native design was removed after NaNs.

### 9.3 Twenty contexts share one physical GPU

All 20 ranks are GPU-capable ranks, but they contend for one GB10. This can still
win because CPU task parallelism and factor reuse dominate for Texas7k. It is
not a general theorem. The larger training case showed only about a 3.56% GPU
improvement in an earlier run because eligibility, fallback/output work, and
contention differed.

### 9.4 Some XML/code options are not full current-wave features

- `fastDecoupled` exists in configuration structures but there is no implemented
  true fast-decoupled power-flow path.
- `hybridMemory` in the generic cuDSS solver remains reserved/no-op.
- Generic cuDSS deterministic/refinement options do not automatically configure
  the direct wave solver.
- `<LinearSolver><Backend>cudss</Backend>` expresses GPU intent, but when
  batched mode is active the driver deliberately leaves the default base/fallback
  backend on PETSc and calls cuDSS directly for the wave.
- The generic scalar backend does not clear its internal `p_factored` flag on
  every thrown factor/solve exception. The active batched CA path uses the
  separate per-wave wrapper and CPU fallback, but scalar-backend failure
  recovery remains less complete.

### 9.5 Controller coverage remains deliberately hybrid

Area interchange disables the fast wave. Generator, islanding,
structure-changing, nonconverged, and controller-acting cases use CPU KLU.
This is a correctness decision and a source of workload-dependent performance.

### 9.6 Some in-source and standalone documentation is historical

Older comments still describe:

- a base Jacobian shared across all cases, although the active chord path uses
  each contingency's first Jacobian;
- `qlim=false` as a wave requirement, although current code performs post-wave
  Q-limit checks and fallback;
- byte-identical asynchronous output, although cross-rank row order is not
  deterministic;
- a simultaneous batched solve, although current cases are sequential.

[`GPU_CA_IMPLEMENTATION.md`](GPU_CA_IMPLEMENTATION.md) and
[`CODE_REVIEW_GPU_CA.md`](CODE_REVIEW_GPU_CA.md) preserve useful historical
evidence but should not override current source. This report calls out the
current behavior explicitly.

### 9.7 CSV escaping assumes identifier-safe circuit IDs

Current code correctly escapes arbitrary contingency names. The direct
formatter writes circuit IDs as raw strings. PSS/E circuit identifiers in the
validated data are identifier-like and do not contain commas, quotes, or line
breaks, but the formatter does not enforce that assumption. General untrusted
circuit identifiers would need the same `csvField` treatment.

### 9.8 Production Docker disables building tests

`GRIDPACK_ENABLE_TESTS=OFF` shortens the image build. The tests exist and were
run in dedicated validation builds, but the ordinary production Docker build
does not compile/run them. A release process should build a separate test stage
or CI image before publishing.

### 9.9 Current result provenance should be archived

The committed `gpucputest/results` data describes the pre-hardening 84.5626 s
run. The 68.260 s current result and its equal-coverage comparison are
session evidence. Committing or release-archiving those outputs would make the
current claim independently auditable.

## 10. Conclusions

The development path moved through four broad realizations:

1. **A GPU library alone was insufficient.** Cold scalar cuDSS was slower than
   KLU.
2. **Reuse required application knowledge.** Fixed sparse structure, local
   branch effects, direct assembly, warm starts, and case-specific chord factors
   made the arithmetic cheaper.
3. **The complete pipeline mattered.** Bounded all-rank scheduling, direct CSV
   enumeration, and removal of the serial merge were necessary for full-output
   speed.
4. **Fast assumptions needed live proof and escape routes.** Pattern checks,
   active-graph validation, adaptive refresh, controller checks, diagnostics,
   schema hardening, and CPU fallback turned a promising prototype into the
   current robust design.

The current speedup is therefore a systems result:

- about 2.38× from stock to current optimized CPU;
- another 1.32× from current optimized CPU to opt-in GPU;
- about 3.14× stock-to-current end to end on the validated full Texas workload.

The current accuracy argument is not “the GPU must be right because it is
faster.” It is:

- the component equations are unchanged;
- the direct placement path is checked against the canonical mapper;
- topology shortcuts are exact under their validated graph/event premises and
  conservative otherwise;
- the true nonlinear mismatch controls convergence;
- adaptive work is bounded;
- unsafe or difficult cases return to exact CPU Newton;
- complete keyed outputs are compared at full scale.

## Reference appendices

### Appendix A. Commit ledger

| Order | Commit | Date | Files / line delta | Primary role | Changes `ca.x` runtime? |
|---:|---|---|---:|---|---|
| Base | [`cff541f4`](https://github.com/alexluhuang/GriDSSPack/commit/cff541f4a0fff89ddd71f254833efad50419cd68) | 2026-06-21 | Boundary | Last commit treated as stock | Baseline |
| 1 | [`10e5cf26`](https://github.com/alexluhuang/GriDSSPack/commit/10e5cf26a9572c117050302f1d40d0d95762e1a0) | 2026-07-17 | 23 files, +2,756/−13 | Release/CUDA build, backend, scalar cuDSS, batch/screen prototypes | Yes: Release and optional scalar backend |
| 2 | [`31a98d56`](https://github.com/alexluhuang/GriDSSPack/commit/31a98d56d40f9e02bb16cdeba371f78d21ac7d0d) | 2026-07-18 | 11 files, +1,901/−212 | Real wave integration, direct CSR, local Y-bus, chord, hybrid fallback, bulk output | Yes |
| 3 | [`81d9015a`](https://github.com/alexluhuang/GriDSSPack/commit/81d9015a6c3569b45ebfcd9c767c7d6aca566c4f) | 2026-07-27 | 4 files, +166/−14 | Iterative Tarjan screen and actionable classification | Yes |
| 4 | [`d939666f`](https://github.com/alexluhuang/GriDSSPack/commit/d939666fb28b08977506595d020004ef91dca6d9) | 2026-07-27 | 8 files, +499/−76 | All-rank bounded waves, global counts, direct CSV, shared append | Yes |
| 5 | [`ca8c245d`](https://github.com/alexluhuang/GriDSSPack/commit/ca8c245ddf3984398e2e48c4370cb0f7a5defe04) | 2026-07-28 | 4 files, +229/−4 | Docker packaging and run documentation | Build only |
| 6 | [`5d66fd83`](https://github.com/alexluhuang/GriDSSPack/commit/5d66fd83fe7314d6af5e9cd573408c1b54cd336a) | 2026-07-28 | 5 files, +1,449 | Accuracy-comparison suite | No |
| 7 | [`570bedb8`](https://github.com/alexluhuang/GriDSSPack/commit/570bedb8251756590e1cee314c171855af76e610) | 2026-07-28 | 8 files, +91,176/−1 | Stored Texas comparison evidence | No |
| 8 | [`67c80996`](https://github.com/alexluhuang/GriDSSPack/commit/67c80996089afcaf5b7249200b322ab9b999a8a9) | 2026-07-29 | 9 files, +1,605/−232 | Live validation, adaptive chord, persistent setup, sparse/output hardening | Yes |
| 9 | [`f0a97125`](https://github.com/alexluhuang/GriDSSPack/commit/f0a97125ee7af7071e0f444937640a2424fb546c) | 2026-07-29 | 1 file, +1/−1 | Comparison-tool default directory | No |

### Appendix B. Current code map

| File / symbol | Current responsibility |
|---|---|
| [`Dockerfile`](Dockerfile) | CUDA 13/cuDSS image, Release GridPACK build, PETSc/KLU retained |
| [`src/CMakeLists.txt`](src/CMakeLists.txt#L122) — `GRIDPACK_WITH_CUDSS` | Optional CUDA/cuDSS discovery and global compile definition |
| [`src/math/linear_solver_backend.hpp`](src/math/linear_solver_backend.hpp#L49) and [`.cpp`](src/math/linear_solver_backend.cpp) | Runtime PETSc/cuDSS selection and availability check |
| [`src/math/petsc/petsc_linear_solver.cpp`](src/math/petsc/petsc_linear_solver.cpp#L42) — solver factory | Constructs requested implementation, with PETSc fallback |
| [`src/math/cudss/cudss_csr_extractor.hpp`](src/math/cudss/cudss_csr_extractor.hpp#L85) — `PetscSeqCSRView` | Reads a serial PETSc AIJ matrix as CSR |
| [`src/math/cudss/cudss_linear_solver_implementation.hpp`](src/math/cudss/cudss_linear_solver_implementation.hpp#L203) — `p_prepare`, `p_solveImpl`, `p_resolveImpl` | Generic scalar cuDSS analysis/factor/solve and exact pattern cache |
| [`src/math/cudss/cudss_batched_solver.hpp`](src/math/cudss/cudss_batched_solver.hpp#L59) — `CuDSSBatchedSolver` | Rank-local single-system workspace reused by a wave; case factor and factor-reuse solves |
| [`src/applications/components/pf_matrix/pf_components.hpp`](src/applications/components/pf_matrix/pf_components.hpp#L347) — voltage-state methods | Saves/restores bus voltage magnitude and angle |
| [`src/applications/modules/powerflow/pf_app_module.hpp`](src/applications/modules/powerflow/pf_app_module.hpp#L165) | Exposes network/factory state and pre-solved convergence overlay |
| [`src/applications/modules/powerflow/pf_batch_ca.hpp`](src/applications/modules/powerflow/pf_batch_ca.hpp#L175) — `PFBatchNR::solveWave` | Exact or adaptive chord nonlinear iteration for eligible cases |
| [`src/applications/modules/powerflow/pf_batch_ca_assembler.hpp`](src/applications/modules/powerflow/pf_batch_ca_assembler.hpp#L73) — `GridpackBatchAssembler`; [`beginWave`](src/applications/modules/powerflow/pf_batch_ca_assembler.hpp#L196); [live validation](src/applications/modules/powerflow/pf_batch_ca_assembler.hpp#L939) | Eligibility, direct CSR/RHS, local Y-bus, case snapshots, validation, output overlays |
| [`src/applications/modules/powerflow/pf_screen.hpp`](src/applications/modules/powerflow/pf_screen.hpp#L54) — `N1ConnectivityScreen` | Iterative Tarjan bridge classification |
| [`src/applications/contingency_analysis/ca_driver.cpp`](src/applications/contingency_analysis/ca_driver.cpp#L357) — `CADriver::execute`; [`captureFlatRows`](src/applications/contingency_analysis/ca_driver.cpp#L964); [global summary](src/applications/contingency_analysis/ca_driver.cpp#L2599) | XML parsing, backend policy, TaskManager waves, fallback, output, global summary |
| [`src/applications/contingency_analysis/ca_async_writer.hpp`](src/applications/contingency_analysis/ca_async_writer.hpp#L42) — `AsyncRowWriter` | Synchronous or single-consumer FIFO block writer, optional append |
| [`src/math/test/cudss_batched_test.cpp`](src/math/test/cudss_batched_test.cpp) | Sparse cuDSS wrapper tests |
| [`src/math/test/pf_batch_ca_test.cpp`](src/math/test/pf_batch_ca_test.cpp) | Nonlinear exact/chord/adaptive behavior |
| [`src/math/test/pf_screen_test.cpp`](src/math/test/pf_screen_test.cpp) | Bridges, parallel circuits, invalid graphs, deep iterative traversal |
| [`gpucputest/compare_results.py`](gpucputest/compare_results.py) | Full output coverage/numerical/status/timing comparison |
| [`gpucputest/results/`](gpucputest/results/) | Committed pre-hardening Texas comparison evidence |

External sparse-solver reference:

- NVIDIA's [cuDSS workflow](https://docs.nvidia.com/cuda/cudss/getting_started.html)
  defines the analysis, factorization, and solve stages.
- The [`cudssExecute` reference](https://docs.nvidia.com/cuda/cudss/functions.html)
  documents reuse of analysis when matrix values change without a structural
  change.

### Appendix C. Attached Texas XML reference

The following describes the exact attached `input.xml` reviewed for this report.
It is reference information; build/run steps remain in [DOCKER_CA.md](DOCKER_CA.md).

#### C.1 Contingency-analysis settings

| XML setting | Attached value | Meaning and effect |
|---|---|---|
| `printCalcFiles` | `false` | Suppresses per-case calculation files and their I/O |
| `FullBranchN1` | `true` | Generates the supported branch/circuit N-1 set |
| `FullGeneratorN1` | `true` | Generates the supported generator N-1 set |
| `groupSize` | `1` | Complete serial network per MPI rank; required by direct rank-local path |
| `maxVoltage`, `minVoltage` | `1.1`, `0.9` | Voltage-violation reporting limits |
| `contingencyRating` | `C` | Contingency flat-output `rate_mva`, `loading_percent`, and row-level `viol` use C→B→A; base rows use A, and the separate legacy overload checker does not consume this C setting |
| `qlim` | `true` | Checks generator reactive limits; acting cases go through full controller solve |
| `qlimDeadband` | `0.1` | Avoids unstable toggling close to a Q limit |
| `LTC` | `false` | Redundant/inert in this CA block for `ca.x`; the effective setting is `Powerflow/LTC` below |
| `writeStats` | `false` | Skips legacy statistical-summary storage/output |
| `outputFormat` | `csv_flat` | Emits one 15-column row per event/circuit |
| `sharedFlatFile` | `true` | All ranks append event blocks to the final flat file |
| `bufferFlatOutput` | `false` | Does not hold complete rank output for end-of-run MPI-IO |
| `overlapIO` | absent → `false` | Historical/current cited Texas runs use synchronous append unless explicitly added |
| `outputFile` | `Texas7k_v2` | Prefix for flat, convergence, and bus artifacts |
| `GPU/enabled` | `true` | Master opt-in switch |
| `GPU/batched` | `true` | Activates rank-local wave engine rather than scalar default backend |
| `GPU/waveSize` | `8` | At most eight claimed tasks per rank-local wave |
| `GPU/warmStart` | `true` | Starts contingencies at the solved base voltage |
| `GPU/screen` | `true` | Enables active-graph Tarjan classification |

#### C.2 Power-flow and linear-solver settings

| XML setting | Attached value | Meaning and effect |
|---|---|---|
| `networkConfiguration` | `Texas7k_20210804.raw` | PSS/E network data |
| `initStart` | `warm` | Present in the deck but not parsed by `ca.x`; PFBus already uses its warm-initialization default |
| `SwitchedShunt` | `false` | No switched-shunt outer controller |
| `qlim`, `qlimDeadband` | `true`, `0.1` | Same Q-limit policy as CA block |
| `LTC` | `false` | No transformer-tap outer controller |
| `AreaInterchange` | `false` | Permits wave activation; `true` would disable it |
| `maxControllerIterations` | `10` | Bound for enabled outer controls |
| `maxIteration` | `50` | Nonlinear Newton/chord cap |
| `tolerance` | `1.0e-4` | True AC mismatch acceptance threshold |
| `maxQlimIterations` | `3` | Q-limit controller bound |
| `dampingFactor` | `1.0` | Applies the full computed correction |
| `phaseShiftSign` | `1.0` | Existing transformer phase convention |
| `LinearSolver/Backend` | `cudss` | Requests cuDSS generally; batched mode still keeps base/fallback default on PETSc and calls wave cuDSS directly |
| `constantFactor` | `true` | Selects contingency-specific chord path; current adaptive logic may still refactor |
| `refactorEvery` | absent | `constantFactor` initially sets the stride to the nonlinear cap |
| `chordCap` | absent | Defaults to `maxIteration` |
| PETSc `ksp_type` | `preonly` | One direct preconditioner application, no iterative Krylov loop |
| PETSc `pc_type` | `lu` | LU direct factorization |
| PETSc matrix solver | `klu` | CPU sparse direct solver for base/fallback/CPU mode |

#### C.3 Benchmark input identity

At report time:

| File | Bytes | SHA-256 |
|---|---:|---|
| `input.xml` | 1,638 | `f73ab6c7b0b8233adc990f51124499602268af8606996ccd14e8fec9149a8e48` |
| `Texas7k_20210804.raw` | 3,349,855 | `4b2287c9a7cc1dfe990c7f2f29bbead48e6f1ae9f53876cee4b2ad36cac89912` |

These hashes identify the files used to describe the workload. They should be
recorded alongside future timings because a changed RAW or XML deck can change
both eligibility and performance.

### Appendix D. Measurement ledger

| Stage | Workload / mode | Result | Evidence status |
|---|---|---:|---|
| Scalar backend | Texas-256, 1 rank | cuDSS 42.10 s; CPU 36.36 s | Session/history |
| Scalar backend | Texas-256, 20 ranks | cuDSS 13.22 s; CPU 4.86 s | Session/history |
| Early integrated | Texas-256, 1 rank | GPU 86.4 s; CPU 114.9 s | Subset session/history |
| Early integrated | Texas-64 | GPU 25.6 s; CPU 30.7 s | Subset session/history |
| Larger network | 24,251-bus/48k-reduced, 64 cases | GPU 131.4 s; CPU 116.0 s | Different-workload session/history |
| Rank-owner | Full Texas, 20 ranks, full output | GPU 92.28 s; optimized CPU median 93.06 s | Session/history; 0.84% |
| All-rank `d939` | Full Texas, 20 ranks, W8 | 86.26 s | Session/history |
| Committed result `570bedb8` | Full Texas, 20 ranks, W8 | GPU 84.5626 s; stock 214.1355 s | Stored result |
| Runtime from `67c80996`; reviewed at HEAD `f0a97125` | Full Texas, 20 ranks, W8 | GPU 68.260 s; optimized CPU 89.922 s; stock 214.136 s | Current session; `f0a97125` changes only comparison-tool defaults |

Only rows with the full Texas workload, 20 ranks, and exhaustive output should
be compared directly as release-level performance claims.

### Appendix E. Verification ledger

#### E.1 Committed pre-hardening evidence

[`gpucputest/results/summary.md`](gpucputest/results/summary.md) and the JSON/CSV
artifacts record:

- full keyed row coverage;
- duplicate checks;
- all convergence/status outcomes;
- numerical error metrics;
- timing and task-balance metrics.

They also preserve 4,268,160 stock-omitted circuit rows, 3,361 historical
violation differences, and historical mismatch-reporting weaknesses. Later
hardening fixes internal rating/flag consistency, but the 3,361 differences are
not attributable to that one issue.

#### E.2 Hardening validation performed in the development session

Dedicated CUDA/container validation included:

- contingency-analysis build;
- `pf_screen` bridge/parallel/deep/invalid-input tests;
- `cudss_batched` sparse wrapper tests;
- `pf_batch_ca` exact/chord/adaptive tests;
- a 64-case live-validation run with `GRIDPACK_BATCH_VALIDATE=1`;
- a 64-case reference run with `GRIDPACK_BATCH_NOFAST=1`;
- a 20-rank 64-case scheduling run;
- full Texas CPU/GPU comparison;
- schema/key/status/violation diff checks.

These passed in the described session. The host outside the container lacked
the project CUDA/GridPACK library environment, so host-only failures caused by
missing libraries are environment limitations, not evidence of a numerical
regression.

#### E.3 Acceptance criteria for a comparable release result

A result should record:

- source commit and image digest;
- RAW/XML hashes;
- GPU model, driver, CUDA, cuDSS, CPU, memory, and filesystem;
- rank count and `groupSize`;
- complete N-1 task count;
- wave size and global inspected/eligible/fallback counters;
- exact output mode and filters;
- wall time and maximum application time;
- flat/convergence/bus schema and row counts;
- duplicate and unmatched keys;
- convergence/status transitions;
- numerical error distributions and maxima.

Without these, a fast time may represent a different workload rather than a
faster implementation.

### Appendix F. Upstream divergence after the chosen base

The following upstream `feature/ca-scalability` commits are after `cff541f4` but
are not ancestors of current GriDSSPack:

| Upstream commit | Subject |
|---|---|
| `f814caf5` | Phantom-flow fix |
| `38484d5a` | Transformer ratings |
| `458feddb` | Unified contingency rating |
| `0ad54411` | Violation reporting |
| `71bb31ef` | Performance-index ranking and rosters |
| `ae4a3f46` | Text violation/summary output |
| `2785ee81` | Reject unknown output format |
| `93613075` | Composite performance index |
| `b92affa1` | Top severe contingencies |
| `e4d643e1` | Slimmer summary JSON |
| `f3f8d18d` | Voltage-deviation field |

Their absence is not necessarily a defect under the user-defined comparison
boundary, but it matters when deciding whether to merge current upstream
functionality into a later GriDSSPack release.

### Appendix G. Glossary

| Term | Plain-language meaning |
|---|---|
| AC | Alternating current; voltage/current have magnitude and relative timing angle |
| Admittance | Ease with which a voltage difference drives current; reciprocal of impedance |
| Structural / symbolic analysis | Planning from the sparse pattern: ordering, symbolic fill, storage, and dependencies |
| Batch / wave | A bounded rank-local task group sharing setup; not a native simultaneous eight-matrix solve |
| Branch | GridPACK object joining two buses; it can contain more than one physical circuit ID |
| Bridge | Graph edge whose removal increases the number of connected components |
| Chord / modified Newton | Nonlinear iteration that reuses a Jacobian/factorization for several steps |
| Circuit | Individually identified line/transformer connection within a branch object |
| Contingency | A simulated equipment outage |
| CSR | `row_ptr`, `col_idx`, and `values` sparse-matrix storage |
| cuDSS | NVIDIA GPU direct sparse solver library |
| Direct solver | Solves a linear system through factorization rather than an outer iterative approximation |
| Factorization | Computing triangular factors, usually \(L\) and \(U\), from current matrix values |
| Fill-in | New stored positions created in sparse factors during elimination |
| Fallback | Re-route to the established CPU per-contingency solve |
| Global Arrays (GA) | Distributed data/mapping infrastructure used by general GridPACK assembly |
| Island | Electrically disconnected component of the network |
| Jacobian | Derivative matrix mapping small voltage-state changes to changes in calculated \(P/Q\); mismatch changes with the opposite sign under this report's convention |
| KLU | CPU sparse direct solver used through PETSc in this deployment |
| Mismatch | Specified minus calculated bus power; physical equation error |
| MPI rank | One cooperating application process; not matrix rank or equipment rating |
| N-1 | Study of every supported single-component outage |
| Numeric factorization | Computing factor values after structural planning |
| PETSc | Numerical library providing GridPACK's established matrix/solver interface |
| PV bus | Generator-controlled voltage bus: specified \(P\) and voltage magnitude |
| PQ bus | Load-style bus: specified \(P\) and \(Q\) |
| Residual | Remaining equation error; in AC iteration, the nonlinear mismatch |
| Slack bus | Reference/balancing bus with specified voltage magnitude and angle |
| Sparse | Most possible matrix positions are absent/zero |
| Structural signature | Per-bus reduced-equation layout used to ensure dimensions stay compatible |
| Tarjan bridge search | One depth-first graph traversal that identifies every bridge |
| Triangular solve | Forward/back substitution using already-computed factors |
| Union-Find | Data structure that merges vertices into connected sets |
| Y-bus | Sparse complex admittance matrix mapping bus voltage to bus current |

### Appendix H. Ambiguous words used in this project

- **Exact Newton** means a current Jacobian each iteration, not exact arithmetic.
- **Chord** means fixed-Jacobian nonlinear iteration here, not an extra edge in a
  graph cycle.
- **Factor** can mean an \(L/U\) matrix or a multiplicative speedup; context must
  distinguish them.
- **Bridge** is a graph property, not a software adapter.
- **Rank** usually means MPI process here, but power engineering also uses
  “rating,” and linear algebra uses matrix rank.
- **Structural** means sparse equation pattern, not physical steel.
- **Residual** may mean the outer AC mismatch or the inner linear solve error;
  they must be named explicitly.
- **Accurate** may mean numerically close voltages, correct convergence status,
  complete circuit coverage, or valid reporting schema. This report states
  which meaning applies to each change.
