# CS 15-888 (Fall 2025) — Computational Game Solving

Tuomas Sandholm & Ioannis Anagnostides — Carnegie Mellon University

Source: <https://www.cs.cmu.edu/~sandholm/cs15-888F25/>

This file collects all reading (paper) links, slide links, and lecture-note links from the schedule, organized by lecture.

---

## Lecture 1 — M 8/25 — Introduction
- **Topic:** Introduction to game theory. Game representations. Normal form, extensive form. Solution concepts. Properties of 2-player 0-sum games.
- **Slides:** <https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture_1_Introduction.pdf>

## Lecture 2 — W 8/27 — Perfect-information games 1
- **Topic:** Tree search methods for two-player perfect-information games: minmax search, alpha-beta pruning, iterative deepening, quiescence search, singular extension, evaluation function learning, endgame databases, horizon problem, search depth pathology, chess.
- **Slides:** <https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture_2_tree-form+perfect-info+games.pdf>

## Lecture 3 — W 9/3 — Perfect-information games 2
- **Topic:** Monte Carlo Tree Search (MCTS). AlphaGo and AlphaGo Zero.
- **Reading:** Silver et al., *Nature* 2017 — <https://www.nature.com/articles/nature24270>
- **Slides:** <https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture_3_MCTS_AlphaGo_AlphaZero.pdf>

## Lecture 4 — M 9/8 — Normal-form games
- **Topic:** LP formulation of zero-sum normal-form equilibrium computation. Fictitious play and FTRL. Online learning and regret minimization. Mirror descent. Multiplicative weights (MWU). Regret matching (RM) and RM+. Self-play and connection to game-theoretic equilibria.
- **Reading:**
  - §4.1 & 4.6 of *MAS* — <http://www.masfoundations.org/mas.pdf>
  - §4.2–4.3 of *AGT* — <https://www.cs.cmu.edu/~sandholm/cs15-892F13/algorithmic-game-theory.pdf>
- **Lecture notes:** <https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture4-LPs-Regret.pdf>
- **Slides:** <https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture4-Slides.pdf>

## Lecture 5 — W 9/10 — Extensive-form games 1
- **Topic:** Extensive-form games. Behavioral representation of a strategy. Kuhn's theorem. Sequence-form representation and sequence-form LP. Construction of CFR and proof of correctness using regret circuits.
- **Reading:**
  - §5.2 of *MAS* — <http://www.masfoundations.org/mas.pdf>
  - §3.10–3.11 of *AGT* — <https://www.cs.cmu.edu/~sandholm/cs15-892F13/algorithmic-game-theory.pdf>
  - Zinkevich et al., *NIPS* 2007 — <https://proceedings.neurips.cc/paper/2007/file/08d98638c6fcd194a4b1e6992063e944-Paper.pdf>
  - Farina et al., *ICML* 2019 — <http://proceedings.mlr.press/v97/farina19b/farina19b.pdf>
  - (F21 Lecture Notes on CFR) — <http://www.cs.cmu.edu/~sandholm/cs15-888F21/L05_cfr.pdf>
- **Lecture notes:** <https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture5-CFR.pdf>
- **Slides:** <https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture5-Slides.pdf>

## Lecture 6 — W 9/17 — Learning in general-sum games: correlated equilibria and Phi-regret
- **Topic:** Correlated and coarse correlated equilibrium. Phi-regret and its connections to types of correlated equilibrium. The GGM framework, and Blum-Mansour as a special case.
- **Reading:**
  - Blum & Mansour, *JMLR* 2007 — <https://www.jmlr.org/papers/volume8/blum07a/blum07a.pdf>
  - Gordon, Greenwald & Marks, *ICML* 2008 — <https://www.cs.cmu.edu/~ggordon/gordon-greenwald-marks-icml-phi-regret.pdf>
- **Lecture notes:** <https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture6-Gordon.pdf>
- **Slides:** <https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture6-Slides.pdf>

## Lecture 7 — M 9/22 — Algorithms for minimizing Phi-regret: nonlinear deviations and ellipsoid
- **Topic:** Achieving swap regret among low-degree deviations. Linear-swap correlated equilibria as a special case. Nonlinear deviations. Expected fixed points. Ellipsoid against hope.
- **Reading:**
  - Zhang et al., *NeurIPS* 2024 — <https://arxiv.org/pdf/2402.09670>
  - Farina & Pipis, *NeurIPS* 2024 — <https://arxiv.org/pdf/2402.16316>
  - Daskalakis et al., *STOC* 2025 — <https://arxiv.org/pdf/2412.20291>
- **Lecture notes:** <https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture7-Nonlinear.pdf>
- **Slides:** <https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture7-Slides.pdf>

## Lecture 8 — M 9/29 — Faster no-regret learning dynamics and last-iterate convergence
- **Topic:** Near-optimal regret using optimism. Connections to last-iterate convergence.
- **Reading:**
  - Anagnostides et al., *ICML* 2022 — <https://arxiv.org/pdf/2203.12056.pdf>
  - Anagnostides et al., *NeurIPS* 2022 — <https://arxiv.org/pdf/2204.11417>
- **Lecture notes:** <https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture8-Optimism.pdf>
- **Slides:** <https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture8-Optimism.pdf>

## Lecture 9 — W 10/1 — Extensive-form games 2: CFR speedups
- **Topic:** Alternation. Reweighted updates of regrets and strategies, LCFR, DCFR. Dynamic pruning in imperfect-information games. Warm starting from given strategies. Optimistic regret minimization algorithms.
- **Reading:**
  - Brown & Sandholm, *ICML* 2017 — <http://www.cs.cmu.edu/~sandholm/cs15-888F21/pruning.icml17.pdf>
  - Brown & Sandholm, *AAAI* 2019 — <http://www.cs.cmu.edu/~sandholm/cs15-888F21/reweighting.aaai19.pdf>
  - Farina, Kroer & Sandholm, *AAAI* 2021 — <https://www.cs.cmu.edu/~gfarina/2021/predictive-approachability-aaai21/predictive-approachability.aaai21.pdf>
- **Slides:** <https://www.cs.cmu.edu/~sandholm/cs15-888F25/CFRspeedups-Slides.pdf>

## Lecture 10 — M 10/6 — Game abstraction 1
- **Topic:** Practical state of the art. Lossless abstraction: *GameShrink*. Lossy state abstraction. Potential-aware, earth-mover-distance abstraction.
- **Reading:** Brown et al., *AAMAS* 2015 — <https://www.cs.cmu.edu/~sandholm/hierarchical.aamas15.pdf>
- **Slides:** <https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture_10_practical_game_abstraction.pdf>

## Lecture 11 — W 10/8 — Game abstraction 2
- **Topic:** Abstraction algorithm for distributed equilibrium finding. Action-abstraction algorithms. Reverse mapping. Abstraction pathology. Lossy abstraction with solution-quality bounds. Application of the theory to game modeling.
- **Reading:** Kroer & Sandholm, *NeurIPS* 2018 — <https://proceedings.neurips.cc/paper/2018/file/aa942ab2bfa6ebda4840e7360ce6e7ef-Paper.pdf>
- **Slides:** <https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture_11_game_abstraction2.pdf>

## Lecture 12 — M 10/20 — Libratus (I)
- **Topic:** History of game theory and AI for poker, rules of Texas hold'em, man-machine match setup.
- **Reading:** Brown & Sandholm, *Science* 2018 — <https://www.science.org/doi/10.1126/science.aao1733>
- **Slides:** <https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture_12_Libratus_and_subgame_solving.pdf>

## Lecture 13 — W 10/22 — Libratus (II)
- **Topic:** Subgame solving in imperfect-information games.
- **Reading:** Brown & Sandholm, *Science* 2018 — <https://www.science.org/doi/10.1126/science.aao1733>
- **Slides:** <https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture_12_Libratus_and_subgame_solving.pdf>

## Lecture 14 — M 10/20 — Libratus (III)
- **Topic:** Solving multiple subgames, self-improver, transforming poker knowledge.
- **Reading:** Brown & Sandholm, *Science* 2018 — <https://www.science.org/doi/10.1126/science.aao1733>
- **Slides:** <https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture_12_Libratus_and_subgame_solving.pdf>

## Lecture 15 — W 10/22 — Knowledge-limited subgame solving (KLSS)
- **Topic:** Limits of subgame solving with common knowledge. Subgame solving without common knowledge. Superhuman Fog-of-War chess. Safe KLSS.
- **Reading:**
  - Zhang & Sandholm, *NeurIPS* 2021 — <https://proceedings.neurips.cc/paper/2021/file/c96c08f8bb7960e11a1239352a479053-Paper.pdf>
  - Liu et al., *ICML* 2023 — <https://openreview.net/pdf?id=5YNVtHulIX>
- **Slides:** <https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture_13_KLUSS_fow-chess.pdf>

## Lecture 16 — M 10/27 — Pluribus (multi-player poker)
- **Topic:** Depth-limited subgame solving.
- **Reading:** Brown & Sandholm, *Science* 2019 — <https://www.science.org/doi/10.1126/science.aay2400>
- **Slides:** <https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture_14_Pluribus_and_depth-limited_subgame_solving.pdf>

## Lecture 17 — W 10/29 — Deep learning in game solving
- **Topic:** Monte Carlo CFR (MCCFR), and sampling approaches. Deep CFR as an alternative to abstraction. DREAM, ESCHER.
- **Reading:**
  - Lanctot et al., *NeurIPS* 2009 (MCCFR) — <http://mlanctot.info/files/papers/nips09mccfr.pdf>
  - Brown et al., *ICML* 2019 (Deep CFR) — <https://arxiv.org/pdf/1811.00164.pdf>
  - McAleer et al., *ICLR* 2023 (ESCHER) — <https://openreview.net/pdf?id=35QyoZv8cKO>
- **Lecture notes:** <https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture17-deeplearning.pdf>
- **Slides:** <https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture17-Slides.pdf>

## Lecture 18 — M 11/3 — Team games (guest: Brian Zhang)
- **Topic:** Team maxmin equilibrium and TMECor; why the latter is often significantly better. Realization polytope: low dimensional but hard to represent; ways around that in practice. Team DAG and Team PSRO.
- **Reading:**
  - Farina et al., *NeurIPS* 2018 — <http://www.cs.cmu.edu/~gfarina/2018/collusion-3players-nips18/collusion_3players.nips18.cr.pdf>
  - Zhang, Farina & Sandholm, *ICML* 2023 — <https://arxiv.org/abs/2202.00789>
  - McAleer et al., *NeurIPS* 2023 — <https://proceedings.neurips.cc/paper_files/paper/2023/file/8e4ccc9ca6ae2225c4cbb7782ab48daf-Paper-Conference.pdf>
  - Zhang et al., *EC* 2024 — <https://arxiv.org/abs/2308.16017>
- **Slides:** <https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture_18_team_games.pdf>

## Lecture 19 — W 11/5 — Automated mechanism design (guest: Brian Zhang)
- **Topic:** Stackelberg equilibria, correlated equilibria, mechanism design, and information design. Optimal (revenue-maximizing) auctions.
- **Reading:**
  - Zhang et al., *NeurIPS* 2023 — <https://arxiv.org/abs/2306.05216>
  - Kamenica & Gentzkow, *AER* (Bayesian Persuasion) — <https://web.stanford.edu/~gentzkow/research/BayesianPersuasion.pdf>
- **Slides:** <https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture19-mechanism_design.pdf>

## Lecture 20 — M 11/10 — Deep learning in game solving 3 (guest: Stephen McAleer)
- **Topic:** Game-theoretic LLM training. DeepNash for expert-level Stratego. Magnetic Mirror Descent.
- **Reading:**
  - Perolat et al., *Science* 2022 (DeepNash / Stratego) — <https://www.science.org/doi/10.1126/science.add4679>
  - Perolat et al., *ICML* 2021 (Magnetic Mirror Descent) — <https://proceedings.mlr.press/v139/perolat21a.html>
  - Sokota et al., *ICLR* 2023 — <https://arxiv.org/pdf/2206.05825.pdf>
- **Slides:** *(not posted on the course page)*

## Lecture 21 — W 11/12 — Double-oracle-based methods
- **Topic:** Double oracle (DO), policy space response oracles (PSRO), extensive-form double oracle (XDO), anytime PSRO, self-play PSRO, diversity in PSRO. AlphaStar and OpenAI Five.
- **Reading:**
  - Heinrich & Silver, *arXiv* 2016 (Fictitious Self-Play) — <https://arxiv.org/pdf/1603.01121.pdf>
  - Vinyals et al., *Nature* 2019 (AlphaStar) — <https://www.nature.com/articles/s41586-019-1724-z>
  - OpenAI et al., 2019 (OpenAI Five) — <https://arxiv.org/abs/1912.06680>
  - Lanctot et al., *NeurIPS* 2017 (PSRO) — <https://arxiv.org/pdf/1711.00832.pdf>
  - McAleer et al., *NeurIPS* 2021 (XDO) — <https://arxiv.org/abs/2103.06426>
  - McAleer et al., *ICLR* 2024 (Anytime PSRO) — <https://arxiv.org/pdf/2207.06541.pdf>
- **Slides:** <https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture21-Slides.pdf>

## Lectures 22–26 — Final exam & project presentations
No slides or readings.

---

## Summary counts
- **Papers / book sections (readings):** 34 links
- **Slide decks:** 18 links (Lectures 12–14 share one file)
- **Course lecture-notes PDFs:** 6 (Lectures 4, 5, 6, 7, 8, 17)

## Notes
- The course page URL-encodes underscores as `%5F` (e.g. `Lecture%5F1%5F...`). The decoded forms used above (plain `_`) resolve to the same files.
- Lectures 12, 13, and 14 all point to the same Libratus & subgame-solving slides file.
