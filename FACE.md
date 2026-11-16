# Complete tiny LASSO optimizer faces

`optimal_face(X, y, penalty)` describes every optimizer of the supplied positive-penalty LASSO as the convex hull of exact rational vertices. It returns attained coordinate ranges, uniqueness, affine dimension, always-zero and always-nonzero coordinates, an average optimizer, and every subset-enumeration outcome.

This is a bounded geometric reference, not a scalable solver or a statistical confidence region. Exactness concerns validated binary64 inputs interpreted as rational numbers. It does not recover unknown values before rounding.

## From one optimizer to all of them

The objective is unchanged:

```text
min_beta ||y-X beta||^2/(2N) + lambda ||beta||_1, lambda > 0.
```

The existing small active/sign oracle supplies one exact KKT optimizer `beta0`. The new face enumeration then works on the original rectangular design, not the oracle's Gram equations.

All optimizers have the same prediction `f=X beta0`. Otherwise their midpoint would strictly decrease squared loss in prediction space while the L1 term could not increase, contradicting optimality. Their residual `r=y-f`, correlations `c=X^T r/N` and objective are therefore also shared. Positive lambda implies that their L1 norms agree.

Let `E={j:|c_j|=lambda}` and `s_j=sign(c_j)` on `E`. KKT says every optimizer has

```text
beta_j = 0 outside E
beta_j = s_j z_j on E, with z_j >= 0
B z = f, where B_j = s_j X_j.
```

Conversely, any `z>=0` with `Bz=f` gives the same residual and correlations, and its coefficient signs satisfy KKT, so it is an optimizer. These constraints describe the entire solution set, not just solutions found by particular sweep orders. The set is bounded: every member has the same finite L1 norm `sum(z_j)`. Hence it is a polytope and equals the convex hull of its vertices.

## Why independent-column subsets give every vertex

At a vertex, the columns indexed by strictly positive `z_j` must be linearly independent. If they were dependent, a nonzero null direction supported there would allow sufficiently small positive and negative moves while preserving nonnegativity and `Bz=f`. The alleged vertex would be a midpoint of two distinct feasible points.

Conversely, a feasible point supported on independent columns is a vertex. If it were a strict convex mixture, every zero coordinate would be zero in both constituent nonnegative vectors. The two vectors would therefore differ by a null direction on those independent columns, impossible.

The implementation enumerates every subset of `E`, including the empty subset. It uses exact rectangular elimination to find the unique solution when the selected columns have full column rank, checks **all** original rows for consistency, and rejects any negative coordinate. Feasible vectors are zero-filled and deduplicated; boundary zeros can cause multiple subsets to return the same vertex. Rank-deficient subsets are skipped, but their feasible continua are not lost: their vertices occur on smaller independent supports.

The returned subset statuses combine rank deficiency and inconsistency into `rank_deficient_or_inconsistent`. They are not distinct infeasibility certificates for singular systems. Completeness follows from the independent-support argument, not from claiming that every skipped singular subset is empty.

No ridge, pseudoinverse or tolerance changes the model. The seed oracle and face routine both depend on the existing validated-input/KKT conventions; their vertex calculations use different systems, but they are not wholly independent end-to-end implementations.

## Tight coordinate ranges, not a Cartesian box

Every coordinate's min/max over the vertices is its attained min/max over all optimizers. The report names a vertex attaining each endpoint. Convexity supplies every intermediate value for that individual coordinate. A vector built by independently selecting values from all the ranges need **not** be an optimizer.

`always_zero` means the coordinate is zero at every optimizer. `always_nonzero` means its range excludes zero. These are statements about this optimization problem, not feature relevance, identifiability of a data-generating process or causal importance. Equicorrelation membership alone is not possible activity: a coordinate may have `|c_j|=lambda` yet still be zero throughout the face.

Affine dimension is the exact rank of all vertex differences from the first vertex. One vertex means a unique optimizer and dimension zero. The returned vertex average is an optimizer; no minimum-L2-norm or preferred-fit claim is made for it.

## Controls

The original duplicate-column control has vertices

```text
[0,7/4,-1/2] and [7/4,0,-1/2].
```

Their full segment is optimal, with objective `5/8`. The first two coefficients each range over `[0,7/4]`; the third is always `-1/2`. The ranges are not independent: the first two must sum to `7/4`. The always-nonzero list is `[2]`, not a claim that this feature is causal.

A second control has distinct, dependent columns:

```text
X = [[1,0,1/2], [0,1,1/2]]
y = [3/2,3/2], lambda = 1/4.
```

Its two vertices are `[0,0,2]` and `[1,1,0]`, prediction `[1,1]`, objective `5/8` and dimension one. Every optimizer is

```text
[1-t,1-t,2t], 0<=t<=1.
```

Each coordinate can vanish at some optimizer, but no optimizer has all coordinates zero. A complete dyadic-grid test checks objective optimality if and only if this segment condition holds, rather than checking the vertices alone. This fixture was developed for the code and is not an untouched benchmark.

The boundary-zero control uses `X=[[1,1],[1,-1]]`, `y=[5/4,3/4]`, lambda `1/4`. Both columns are equicorrelated, yet the unique optimizer is `[3/4,0]`. The strict Gap Safe rule at this optimum does not certify the second zero because its dual margin is exactly zero. The complete face identifies it without weakening the strict screening rule. Neither routine changes the floating solver.

## Run and input contract

```python
from sparselab import optimal_face

report = optimal_face([[1,0,.5],[0,1,.5]], [1.5,1.5], .25)
assert report['dimension'] == 1
assert report['vertex_count'] == 2
```

```sh
python -m sparselab face examples/face-input.json
python examples/verify_wheel.py
```

The face domain is 1..32 rows and 1..5 columns, matching the seed oracle. Penalty must be a finite numeric value in `[1e-8,10000]`; bools and zero are rejected. Existing feature/response bounds and nonzero-column norm checks apply. Inputs are converted to validated binary64, then exactly represented as Fractions. Rational strings are not an alternate input parser. Internal values can grow; this is not a bit-complexity bound.

The JSON request has exactly `schema_version`, `X`, `y` and `penalty`, with integer version one. Existing 100,000-byte, duplicate-key, nonfinite, UTF-8 and unknown-field checks apply; a parser-recursion failure is a controlled input error. Reports preserve exact values as fraction strings. The original floating, screening and branch domains do not change.

## Verification and attribution

Twelve added tests check the duplicate continuum, distinct dependent columns, signed-duplicate simplices through dimension four, boundary equicorrelation zeros, all per-vertex KKT/dual/gap/prediction equations and coordinate endpoints. Thirty seeded integer designs compare every vertex with the existing Gram/sign oracle, including duplicate columns. That is a cross-check of two vertex constructions, not an independent seed optimizer.

Independent closed forms, a complete 2,197-point dyadic grid, rectangular-system consistency/rank controls, row replication, column permutations, response scaling/signs, zero columns, maximum dimensions, floating sweep-order comparisons, strict CLI and isolation checks supplement that shared-oracle comparison. The full suite has 74 tests. The installed-wheel verifier runs them all outside checkout, checks all eight module origins before and after, matches all seven complete CLI reports and regenerates all three original figures. The six previous reports and all three SVG files remain unchanged.

The optimizer polytope, common fitted values/L1 norm and tight coefficient bounds are established results. Ryan J. Tibshirani, [The Lasso Problem and Uniqueness](https://arxiv.org/html/1206.0313), Lemma 1 and section 4.2, describes the solution polytope and coefficient-bound LPs. Its unnormalized loss uses a penalty equal to `N*lambda` in this repository's convention. This code uses tiny exact independent-support enumeration instead of a general LP package; it does not implement LARS or claim a new uniqueness theorem. Existing coordinate-descent and screening sources remain in [DESIGN.md](DESIGN.md) and [SCREENING.md](SCREENING.md).

No predictive validation, causal feature meaning, statistical confidence, post-selection inference, private-data analysis, full regularization path, scalable sparse optimization or speedup claim is added.
