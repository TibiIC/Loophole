# Harder variants (prepared, not switched)

Status: report only. The task stays on Alpha 1.0 until the team decides otherwise. No geometry
has been changed.

## What the published material contains

Zhang et al. 2020's supplementary archive (`supp-material.zip`, sha256 `00dff255…`, the same
archive our Alpha 1.0 oracle comes from) holds 17 puzzles under `supp-material/puzzles/`:

| puzzle | solution path | puzzle repo @ `cd05c5d` |
|---|---|---|
| alpha (1.0, current task) | `alpha-1.0.simplified.txt` (22 poses) + `.ompl.txt` + `.npz` | yes |
| alpha-1.1 | yes | yes |
| alpha-1.15 | yes | yes |
| **doublealpha** (1.0) | `doublealpha-1.0.simplified.txt` (34 poses) + `.ompl.txt` + `.npz` | yes |
| **ag** (ag-2; "alpha-g") | `ag-2.simplified.txt` (17 poses) + `.ompl.txt` + `.npz` | yes |
| abc, aj, az, claw, duet-g1/g2/g4/g9, enigma, key_1, mobius | (directories present; not inspected in detail) | partly |

The puzzle repository (`gitlab.com/xinyazhang/puzzle-geometry`) has `puzzles/ag` and
`puzzles/doublealpha` at the commit we already pin. So both can be fetched at Docker build time
with sha256 checks, exactly like Alpha 1.0, and never committed.

## The two candidates

| | Alpha 1.0 (current) | double-alpha 1.0 | ag-2 ("alpha-g") |
|---|---|---|---|
| moving piece | alpha (2,530 faces) | alpha (2,416 faces) | ag (9,328 faces) |
| fixed piece | alpha (2,530 faces) | **two alphas joined** (`both__sf`, 5,290 faces) | ag (same mesh) |
| pieces congruent? | yes | **no**: the fixed piece is two alphas; the moving alpha matches each half | **yes** |
| published solution | 22 poses | 34 poses | 17 poses |
| sim cost (face-wedge pieces ≈ faces) | ≈ 5.1 k geoms | ≈ 7.7 k geoms (~1.5×) | ≈ 18.7 k geoms (~3.7×) |
| licence | no licence file; geometry derives from OMPL.app (Rice: no redistribution) → fetch at build | same; the alpha meshes derive from OMPL.app | same repo, no licence file → fetch at build |

## Oracle feasibility

Both have a published solution in the same format as Alpha 1.0. The existing pipeline applies
unchanged: fetch, `prepare()` (face-wedge decomposition), dense validation of the authors' path
in our simulator at 0.8 mm, conversion to rig moves, replay with the audit, then shortcut
smoothing (`scripts/smooth_oracle.py`).

* **double-alpha: feasible, moderate cost.** 34 poses, about 1.5× the geometry. The fixed body
  is one mesh made of two alphas, so it is a single fixed frame body, as now. Risks:
  * the `.cfg` start differs in scale (start at (−5, −1, 7), volume ±60–80), so `unit_scale`
    must be re-derived as for Alpha 1.0;
  * the dense validation of the published path must pass in our hulls. It passed for Alpha 1.0
    after the face-wedge fix, but is unverified here.
* **ag-2: feasible in principle, expensive.** Only 17 poses, but a 9.3 k-face mesh on both
  pieces: about 3.7× the contact geoms. Expect simulation, the free-escape test and roadmap
  building to be 3–4× slower (the escape test is ~0.3 s per action today), and memory to
  grow. Same unit-scale and validation risks.

## Why a non-congruent variant is the stronger escalation

All three agent runs that solved the redesigned task registered tube B onto tube A with ICP
(docs/ANOMALY.md, section 5). On congruent pieces this gives B's exact pose and one shared
shape model from two noisy point clouds.

* **ag-2 keeps this shortcut**: both pieces are the same mesh.
* **double-alpha weakens it.** The fixed piece is not a copy of the moving one, so it must be
  reconstructed and segmented on its own (two loops, more self-occlusion). The moving alpha can
  still be matched to one half of the fixed piece, which makes registration ambiguous rather
  than impossible. Its path is also longer (34 poses).

**Recommendation if the team escalates: double-alpha first**, because it breaks congruence at
modest simulation cost. ag-2 is harder mainly computationally, not for the agent's perception
pipeline.
