# Software and methods

The software this work is built on and the general methods its code uses. The research sources
whose provenance carries an argument (the Alpha puzzle and its published solution, reward
shaping, the goal test, the topology and cable-routing work) are in [References](references.md).

Every paper below was checked on 2026-11-15 against its DOI record (Crossref), its arXiv listing,
or the publisher's proceedings page (title, authors, venue, year, pages). Software without a paper
is listed as a project link. Versions are the ones pinned in `pyproject.toml` or the scripts.

## Simulation and geometry

- **MuJoCo.** Emanuel Todorov, Tom Erez, Yuval Tassa. *MuJoCo: A physics engine for
  model-based control.* IEEE/RSJ International Conference on Intelligent Robots and Systems
  (IROS 2012), pp. 5026–5033.
  [doi:10.1109/IROS.2012.6386109](https://doi.org/10.1109/IROS.2012.6386109) ·
  [github.com/google-deepmind/mujoco](https://github.com/google-deepmind/mujoco) (`mujoco` 4.0.1)
  *[The physics engine for both puzzles (the rigid Alpha tubes, the Rat Trap's capsule-chain
  cord), the renderer behind every observation and clip, and the collision pipeline that the
  goal test and the audit share.]*

- **FCL.** Jia Pan, Sachin Chitta, Dinesh Manocha. *FCL: A general purpose library for
  collision and proximity queries.* IEEE International Conference on Robotics and Automation
  (ICRA 2012), pp. 3859–3866.
  [doi:10.1109/ICRA.2012.6225337](https://doi.org/10.1109/ICRA.2012.6225337) ·
  [github.com/flexible-collision-library/fcl](https://github.com/flexible-collision-library/fcl)
  (Python bindings `python-fcl` 0.7.0.11, development only)
  *[An independent collision checker: it confirms that the published Alpha 1.0 path is
  collision-free on the raw meshes, and our MuJoCo collision model is measured against it.]*

- **CoACD.** Xinyue Wei, Minghua Liu, Zhan Ling, Hao Su. *Approximate convex decomposition for
  3D meshes with collision-aware concavity and tree search.* ACM Transactions on Graphics
  41(4), pp. 1–18, 2022.
  [doi:10.1145/3528223.3530103](https://doi.org/10.1145/3528223.3530103) ·
  [colin97.github.io/CoACD](https://colin97.github.io/CoACD/) (`coacd` 1.0.14)
  *[Convex decomposition for arbitrary imported meshes (`loophole/sim/mesh_import.py`). The
  Alpha tubes use our own exact face-wedge decomposition instead.]*

- **trimesh**: [github.com/mikedh/trimesh](https://github.com/mikedh/trimesh) (`trimesh` 5.2.1).
  *[Mesh loading and processing.]*

- **Assimp**: [github.com/assimp/assimp](https://github.com/assimp/assimp).
  *[Not called directly: the mesh loader reproduces how OMPL.app loads meshes with Assimp, so
  poses written in OMPL's convention mean the same thing in our simulator.]*

## Motion planning

- **OMPL.** Ioan A. Sucan, Mark Moll, Lydia E. Kavraki. *The Open Motion Planning Library.*
  IEEE Robotics & Automation Magazine 19(4):72–82, 2012.
  [doi:10.1109/MRA.2012.2205651](https://doi.org/10.1109/MRA.2012.2205651) ·
  [ompl.kavrakilab.org](https://ompl.kavrakilab.org) ·
  [github.com/ompl/omplapp](https://github.com/ompl/omplapp) (archived)
  *[The Alpha 1.0 problem file and pose conventions follow OMPL. An earlier task used the
  OMPL.app Alpha 1.5 meshes; it is superseded.]*

- **RRT-Connect.** J. J. Kuffner, S. M. LaValle. *RRT-Connect: An efficient approach to
  single-query path planning.* IEEE International Conference on Robotics and Automation
  (ICRA 2000), vol. 2, pp. 995–1001.
  [doi:10.1109/ROBOT.2000.844730](https://doi.org/10.1109/ROBOT.2000.844730)
  *[Our SE(3) planner (`loophole/planning/rrt.py`). It produced references only for the
  procedural Alpha variants and the superseded Alpha 1.5 task. The Alpha 1.0 oracle is the
  published path, and the Rat Trap pipeline has no solver at all.]*

## Training

- **Gymnasium.** Mark Towers, Ariel Kwiatkowski, Jordan Terry, John U. Balis, Gianluca De Cola,
  Tristan Deleu, Manuel Goulão, Andreas Kallinteris, Markus Krimmel, Arjun KG, Rodrigo
  Perez-Vicente, Andrea Pierré, Sander Schulhoff, Jun Jet Tai, Hannah Tan, Omar G. Younis.
  *Gymnasium: A Standard Interface for Reinforcement Learning Environments.* 2024.
  [arXiv:2407.17032](https://arxiv.org/abs/2407.17032)
  *[The interface of `loophole.gym.AlphaEnv` and `make_vec_env`.]*

- **Reverse curriculum.** Carlos Florensa, David Held, Markus Wulfmeier, Michael Zhang, Pieter
  Abbeel. *Reverse Curriculum Generation for Reinforcement Learning.* Conference on Robot
  Learning (CoRL 2017), PMLR 78, pp. 482–495.
  [proceedings.mlr.press/v78/florensa17a](https://proceedings.mlr.press/v78/florensa17a.html) ·
  [arXiv:1707.05300](https://arxiv.org/abs/1707.05300)
  *[Training starts near the goal, moved back as the success rate grows: `start="curriculum"`.]*

- **Domain randomisation.** Josh Tobin, Rachel Fong, Alex Ray, Jonas Schneider, Wojciech
  Zaremba, Pieter Abbeel. *Domain randomization for transferring deep neural networks from
  simulation to the real world.* IEEE/RSJ International Conference on Intelligent Robots and
  Systems (IROS 2017), pp. 23–30.
  [doi:10.1109/IROS.2017.8202133](https://doi.org/10.1109/IROS.2017.8202133)
  *[The training-only randomisation flags: orientation, friction, contact stiffness,
  observation noise.]*

## Rat Trap reference conversion

- **Ramer–Douglas–Peucker.** Urs Ramer. *An iterative procedure for the polygonal approximation
  of plane curves.* Computer Graphics and Image Processing 1(3):244–256, 1972.
  [doi:10.1016/S0146-664X(72)80017-0](https://doi.org/10.1016/S0146-664X(72)80017-0) ·
  David H. Douglas, Thomas K. Peucker. *Algorithms for the reduction of the number of points
  required to represent a digitized line or its caricature.* Cartographica 10(2):112–122, 1973.
  [doi:10.3138/FM57-6770-U75U-7727](https://doi.org/10.3138/FM57-6770-U75U-7727)
  *[Simplifies the recorded handle trajectory of the human solve into discrete agent commands
  (tolerance 0.1 mm; `loophole/puzzles/rattrap/teleop.py`).]*

## The Rat Trap's progress measure: borrowed and ours

The checkpoint progress measure ([Reward design](RATTRAP_DESIGN.md)) was influenced by
**SILO** (Tao et al. 2026, `[TXR+26]` in [References](references.md)).

- **Borrowed from SILO:** localising progress to sub-stages, each with a simple geometric
  success test, and preparing the intermediate states in advance.
- **Ours:** what the stages are and how they are matched. The checkpoints are states of a human
  demonstration, and they are reached in order by ε-ball matching (largest deviation over the
  cord's nodes, ε calibrated on the demonstration).

## The puzzles

- **Rat Trap.** A commercially available wooden puzzle; its designer is unknown. We bought one
  and modelled it from the physical object.
- **Alpha.** Designed by Boris Yamrom (GE Corporate R&D) and made a standard benchmark through
  Nancy Amato's group (Parasol Lab) and OMPL.app, as the repo README describes; a reference entry
  for that origin is pending. The Alpha 1.0 geometry and published solution we use are from
  Zhang et al. 2020 (`[ZBKV20]` in [References](references.md)).

## Harness and tooling (project links)

- **Harbor**: [harborframework.com](https://harborframework.com) ·
  [github.com/harbor-framework/harbor](https://github.com/harbor-framework/harbor) (`harbor`
  0.25.0). *[Task format, containers, the `oracle` and `nop` agents, and the verifier step.]*
- **Model Context Protocol**: [modelcontextprotocol.io](https://modelcontextprotocol.io), and its
  Python SDK, which includes FastMCP:
  [github.com/modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk)
  (`mcp` 1.30.0). *[The agent-facing simulator server.]*
- **OpenCode**: [opencode.ai](https://opencode.ai) ·
  [github.com/anomalyco/opencode](https://github.com/anomalyco/opencode).
  *[The coding agent used for the agent runs.]*
- **Docsify**: [docsify.js.org](https://docsify.js.org). *[This site.]*
