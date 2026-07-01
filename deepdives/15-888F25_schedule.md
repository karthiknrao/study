# 15-888 Computational Game Solving — Fall 2025 Schedule

Source: https://www.cs.cmu.edu/~sandholm/cs15-888F25/

| #  | Date      | Topic | Presenters | Reading(s) | Slides/Notes |
|----|-----------|-------|------------|------------|--------------|
| 1  | M 8/25    | Introduction | — | Course organization. Game theory, representations. Normal form, extensive form. Solution concepts. Properties of 2-player 0-sum games. | [Slides](https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture_1_Introduction.pdf) |
| 2  | W 8/27    | Perfect-information games 1 | — | Tree search, minmax, alpha-beta, iterative deepening, quiescence, evaluation functions, endgame databases, chess. | [Slides](https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture_2_tree-form+perfect-info+games.pdf) |
| 3  | W 9/3     | Perfect-information games 2 | — | Monte Carlo Tree Search (MCTS). AlphaGo and AlphaGo Zero. [Silver et al., Nature 2017] | [Slides](https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture_3_MCTS_AlphaGo_AlphaZero.pdf) |
| 4  | M 9/8     | Normal-form games | — | LP for zero-sum equilibrium. Fictitious play, FTRL, regret minimization, Mirror Descent, MWU, RM/RM+, self-play. [MAS §4.1, 4.6] [AGT §4.2-3] | [Notes](https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture4-LPs-Regret.pdf) · [Slides](https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture4-Slides.pdf) |
| 5  | W 9/10    | Extensive-form games 1 | — | Behavioral strategies, Kuhn's theorem, sequence-form LP, CFR. [Zinkevich et al., NIPS 2007] [Farina et al., ICML 2019] [F21 Notes] | [Notes](https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture5-CFR.pdf) · [Slides](https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture5-Slides.pdf) |
| 6  | M 9/15    | Correlated equilibria and Phi-regret | — | CCE, Phi-regret, GGM, Blum-Mansour. [Blum & Mansour, JMLR 2007] [Gordon et al., ICML 2008] | [Notes](https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture6-Gordon.pdf) · [Slides](https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture6-Slides.pdf) |
| 7  | W 9/17    | Phi-regret minimization | — | Swap regret, nonlinear deviations, ellipsoid against hope. [Zhang et al., NeurIPS 2024] [Farina & Pipis, NeurIPS 2024] [Daskalakis et al., STOC 2025] | [Notes](https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture7-Nonlinear.pdf) · [Slides](https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture7-Slides.pdf) |
| 8  | M 9/22    | Faster no-regret dynamics | — | Optimism, last-iterate convergence. [Anagnostides et al., ICML 2022] [Anagnostides et al., NeurIPS 2022] | [Notes](https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture8-Optimism.pdf) · [Slides](https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture8-Slides.pdf) |
| 9  | W 9/24    | CFR speedups | — | Alternation, LCFR/DCFR, dynamic pruning, warm starting, optimistic RM. [Brown & Sandholm, ICML/AAAI 2017/19] [Farina et al., AAAI 2021] | [Slides](https://www.cs.cmu.edu/~sandholm/cs15-888F25/CFRspeedups-Slides.pdf) |
| 10 | M 9/29    | Game abstraction 1 | — | GameShrink, lossy state abstraction, EMD. [Brown et al., AAMAS 2015] | [Slides](https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture_10_practical_game_abstraction.pdf) |
| 11 | W 10/1    | Game abstraction 2 | — | Distributed equilibrium abstraction, action abstraction, pathology. [Kroer & Sandholm, NeurIPS 2018] | [Slides](https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture_11_game_abstraction2.pdf) |
| 12 | M 10/6    | Libratus (I) | — | Poker AI history, rules, setup. [Brown & Sandholm, Science 2018] | [Slides](https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture_12_Libratus_and_subgame_solving.pdf) |
| 13 | W 10/8    | Libratus (II) | — | Subgame solving. [Brown & Sandholm, Science 2018] | [Slides](https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture_12_Libratus_and_subgame_solving.pdf) |
| 14 | M 10/20   | Libratus (III) | — | Multi-subgame solving, self-improver. [Brown & Sandholm, Science 2018] | [Slides](https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture_12_Libratus_and_subgame_solving.pdf) |
| 15 | W 10/22   | Knowledge-limited subgame solving | — | KLSS, Fog-of-War chess. [Zhang & Sandholm, NeurIPS 2021] [Liu et al., ICML 2023] | [Slides](https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture_13_KLUSS_fow-chess.pdf) |
| 16 | M 10/27   | Pluribus | — | Depth-limited subgame solving. [Brown & Sandholm, Science 2019] | [Slides](https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture_14_Pluribus_and_depth-limited_subgame_solving.pdf) |
| 17 | W 10/29   | Deep learning in game solving | — | MCCFR, Deep CFR, DREAM, ESCHER. [Lanctot et al., NeurIPS 2009] [Brown et al., ICML 2019] [McAleer et al., ICLR 2023] | [Notes](https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture17-deeplearning.pdf) · [Slides](https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture17-Slides.pdf) |
| 18 | M 11/3    | Team games | Brian Zhang | TMECor, realization polytope, Team DAG, Team PSRO. [Farina et al., NeurIPS 2018] [Zhang et al., ICML 2023] [McAleer et al., NeurIPS 2023] [Zhang et al., EC 2024] | [Slides](https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture_18_team_games.pdf) |
| 19 | W 11/5    | Automated mechanism design | Brian Zhang | Stackelberg/CE, mechanism/information design, auctions. [Zhang et al., NeurIPS 2023] [Kamenica & Gentzkow, AER] | [Slides](https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture19-mechanism_design.pdf) |
| 20 | M 11/10   | Deep learning in game solving 3 | Stephen McAleer | Game-theoretic LLM training, DeepNash/Stratego, Magnetic Mirror Descent. [Perolat et al., Science 2022] [Perolat et al., ICML 2021] [Sokota et al., ICLR 2023] | — |
| 21 | W 11/12   | Double oracle methods | — | DO, PSRO, XDO, AlphaStar, OpenAI Five. [Heinrich & Silver, 2016] [Vinyals et al., Nature 2019] [OpenAI 2019] [Lanctot et al., NeurIPS 2017] [McAleer et al., NeurIPS 2021/ICLR 2024] | [Slides](https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture21-Slides.pdf) |
| 22 | M 11/17   | Final exam | — | — | — |
| 23 | W 11/19   | Project presentations | Andrew; Aaron | — | — |
| 24 | M 11/24   | Project presentations | Siddharth; Enzo and Wookho | — | — |
| 25 | M 12/1    | Project presentations | Rohan; Jet; Boxiang | — | — |
| 26 | W 12/3    | Project presentations | Pranav, Andrew, Itai; Alessandro, Yassine, Mark; Juho | — | — |
