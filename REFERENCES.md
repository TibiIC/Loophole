# References

Every entry was checked against its DOI record (Crossref), arXiv listing, or publisher /
proceedings listing on 2026-11-14 (title, authors, venue, year, volume/pages). None is
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
