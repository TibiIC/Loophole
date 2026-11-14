# References

Every entry was checked against its DOI record (Crossref), arXiv listing, or publisher /
proceedings listing on 2026-11-14 ([WL94] and [Mir96] against the publisher record and the author's page) (title, authors, venue, year, volume/pages). None is
marked `[unverified]`. How each is used is noted in brackets.

## Alpha puzzle geometry, planning and scoring

- **[ZBKV20]** Xinya Zhang, Robert Belfer, Paul G. Kry, Etienne Vouga.
  *C-Space Tunnel Discovery for Puzzle Path Planning.* ACM Transactions on Graphics
  39(4), Article 104 (pp. 104:1–104:14), SIGGRAPH 2020.
  [doi:10.1145/3386569.3392468](https://doi.org/10.1145/3386569.3392468) ·
  [project page](https://xinyazhang.gitlab.io/puzzletunneldiscovery/)
  *[Source of the Alpha 1.0 puzzle set and of the published solution used as our oracle;
  motivation for the C-space tunnel potential. The article number is 104, not 100.]*

- **[JSCP15]** Lucas Janson, Edward Schmerling, Ashley Clark, Marco Pavone.
  *Fast Marching Tree: a Fast Marching Sampling-Based Method for Optimal Motion Planning
  in Many Dimensions.* The International Journal of Robotics Research 34(7):883–921, 2015.
  [doi:10.1177/0278364915577958](https://doi.org/10.1177/0278364915577958) ·
  [arXiv:1306.3532](https://arxiv.org/abs/1306.3532)
  *[Cost-to-come / cost-to-go. Verbatim (Sec. 6, SE(3) Alpha puzzle): "The problem is the
  famous 1.5 Alpha puzzle (Amato et al., 1998) … Simulation results show that the problem
  presents two homotopy classes of paths (Figure 13(a))." This is why progress is
  measured by cost-to-go, not by distance to one reference path.]*

- **[NHR99]** Andrew Y. Ng, Daishi Harada, Stuart J. Russell.
  *Policy Invariance under Reward Transformations: Theory and Application to Reward
  Shaping.* Proceedings of the 16th International Conference on Machine Learning (ICML
  1999), pp. 278–287.
  [ML Anthology](https://mlanthology.org/icml/1999/ng1999icml-policy/)
  *[Potential-based shaping preserves optimal policies; our shaping uses γ = 1, Φ = −ctg.]*

## Goal test: free escape by translation

- **[WL94]** Randall H. Wilson, Jean-Claude Latombe. *Geometric reasoning about
  mechanical assembly.* Artificial Intelligence 71(2):371–396, 1994.
  [doi:10.1016/0004-3702(94)90048-5](https://doi.org/10.1016/0004-3702(94)90048-5)
  *[Assembly planning: whether a part (subassembly) can be removed from the rest by a
  motion, in particular by a single translation to infinity. Our goal test is that notion
  for the free tube: it can slide to infinity along some straight line without touching
  the fixed one. We test a fixed set of 64 directions, which is a sufficient check.]*

- **[Mir96]** Brian Mirtich. *Impulse-based Dynamic Simulation of Rigid Body Systems.*
  PhD thesis, University of California, Berkeley, 1996.
  [project page](https://people.eecs.berkeley.edu/~jfc/mirtich/impulse.html)
  *[Conservative advancement: a body can move by less than its distance to the others
  without making contact. Our escape sweep advances by the (lower-bound) separation
  distance at each step.]*

## Training-only shaping (adapted, not reproduced)

- **[TLA+23]** Bingjie Tang, Michael A. Lin, Iretiayo Akinola, Ankur Handa, Gaurav S.
  Sukhatme, Fabio Ramos, Dieter Fox, Yashraj Narang. *IndustReal: Transferring
  Contact-Rich Assembly Tasks from Simulation to Reality.* Robotics: Science and Systems
  (RSS) 2023. [arXiv:2305.17110](https://arxiv.org/abs/2305.17110)
  *[Inspiration for ClearanceShapingReward. Their SDF-based dense reward samples N = 1000
  points on the plug mesh, evaluates them in the SDF of the same plug at its **target
  pose**, and uses −log of the mean SDF value: it measures alignment with the goal pose,
  not clearance between parts. Our reward is an adaptation to a separation task.]*

- **[NSA+22]** Yashraj Narang, Kier Storey, Iretiayo Akinola, Miles Macklin, Philipp
  Reist, Lukasz Wawrzyniak, Yunrong Guo, Adam Moravanszky, Gavriel State, Michelle Lu,
  Ankur Handa, Dieter Fox. *Factory: Fast Contact for Robotic Assembly.* Robotics:
  Science and Systems (RSS) 2022. [arXiv:2205.03532](https://arxiv.org/abs/2205.03532)
  *[Simulation framework underlying IndustReal.]*

## Linking numbers (diagnostic only)

- **[KL00]** Konstantin Klenin, Jörg Langowski. *Computation of writhe in modeling of
  supercoiled DNA.* Biopolymers 54(5):307–317, 2000.
  [doi:10.1002/1097-0282(20001015)54:5<307::AID-BIP20>3.0.CO;2-Y](https://doi.org/10.1002/1097-0282(20001015)54:5%3C307::AID-BIP20%3E3.0.CO;2-Y)
  *[Exact segment-pair formula for the Gauss integral of polylines; used for the logged
  open-curve linking diagnostic.]*

- **[RN11]** Renzo L. Ricca, Bernardo Nipoti. *Gauss' linking number revisited.* Journal
  of Knot Theory and Its Ramifications 20(10):1325–1343, 2011.
  [doi:10.1142/S0218216511009261](https://doi.org/10.1142/S0218216511009261)
  *[Background on the Gauss linking integral.]*

- **[PS02]** Józef H. Przytycki, Adam S. Sikora. *Topological Insights from the Chinese
  Rings.* Proceedings of the American Mathematical Society 130(3):893–902, 2002.
  [arXiv:math/0007134](https://arxiv.org/abs/math/0007134)
  *[Topological model for **string** puzzles (the Chinese Rings), relevant to the rope
  puzzles in this repo; not a model of the Alpha puzzle, whose two open pieces are
  topologically unlinked from the start: the obstruction is geometric.]*

## Rope and cable manipulation (Rat Trap reward design)

Checked on 2026-11-14 against the arXiv listings and the RSS proceedings / CMU RI listing.

- **[MB24]** Peter Mitrano, Dmitry Berenson. *The Grasp Loop Signature: A Topological
  Representation for Manipulation Planning with Ropes and Cables.* ICRA 2024.
  [arXiv:2403.01611](https://arxiv.org/abs/2403.01611)
  *[Topological representation for rope / cable manipulation, used to guide planning, not as a
  reward. Cited in docs/RATTRAP_DESIGN.md.]*

- **[BLK11]** Subhrajit Bhattacharya, Maxim Likhachev, Vijay Kumar. *Identification and
  Representation of Homotopy Classes of Trajectories for Search-based Path Planning in 3D.*
  Robotics: Science and Systems VII (RSS 2011), pp. 9–16.
  [roboticsproceedings.org/rss07/p02](https://www.roboticsproceedings.org/rss07/p02.html)
  *[The h-signature lineage: homotopy-class invariants for planning.]*

- **[TXR+26]** Stone Tao, Jie Xu, Hesam Rabeti, Yashraj Narang, Yijie Guo, Iretiayo Akinola.
  *SILO: Simulation-in-the-Loop Sim-to-Real Transfer for Multi-Stage Cable Routing.*
  [arXiv:2607.04616](https://arxiv.org/abs/2607.04616) (July 2026)
  *[§4.2: the multi-harness task is split into single-harness subtasks with a geometric success
  test (the two cable links nearest the gripper within a radius of the harness centre); within a
  stage the reward is dense proximity plus grasp / success indicators. The basis for our staged
  checkpoint progress.]*

## Shared, unpublished work

- **[HO26]** A colleague's disentanglement RL environment, `puzzle-handoff-v1`
  (package `disentanglement_env` 0.1.0, export manifest dated 2026-11-14), shared with us
  privately. **No licence is stated**; the archive is not in this repository or its history.
  Their puzzle is a rigid closed ring threaded on a static bent wire with a stopper: the same
  puzzle family as our Rat Trap, but with a rigid wire instead of a deformable cord, so the
  configuration space is 6-D rather than ~180-D. It is adapter-compatible with our rig because,
  like Alpha, it is a single 6-DoF rigid body against a fully static obstacle.
  *[Adapted, with credit in each docstring: (1) the potential-based shaping scaffold
  `lam * (gamma * Phi(s') - Phi(s))` with Phi forced to 0 on termination and kept at
  time-limit truncation (`core.py`, `step` / `potential`), now
  `loophole.reward.components.PotentialShaping`; (2) the squared orientation-alignment term
  and `local_tangent` (`core.py`, `potential_terms` / `local_tangent`), generalised as
  `loophole.reward.alignment.AperturePassageAlignment`; (3) the `motion_safe` swept-motion
  certificate (recursive bisection against `|dp|/2 + r_out |rv|/2`), now
  `loophole.sim.certify`. Not adopted: their Euclidean distance-to-goal potential (their
  3-seed, 100k-step test gave 0/300 successes with either reward), their 13-dimensional
  observation, and their reject-the-whole-action semantics.]*

## Corrections to earlier drafts

- [ZBKV20] is Article **104**, not 100.
- [KL00]'s title is *Computation of writhe in modeling of supercoiled DNA* (an earlier
  draft gave a different title).
- [TLA+23] is RSS 2023 only (an earlier draft also said "CoRL 2023").
- [PS02] concerns the Chinese Rings; an earlier draft attributed to it "topological
  methods in mechanical linkages" for the Alpha puzzle.
- Earlier drafts claimed |Lk| = 1 at the Alpha start and 0 at separation and a steady
  decrease; these were removed. For open centre-lines the Gauss integral is real-valued
  and changes under rotation in place, and is now logged only.
