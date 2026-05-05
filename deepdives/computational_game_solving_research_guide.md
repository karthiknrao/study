# Computational Game Solving — Research Guide

Based on CMU 15-888 (Fall 2025) by Tuomas Sandholm.
Assembled May 2025.

---

## Course Overview

This course covers the fundamentals and state of the art in solving **multi-step imperfect-information games** — the class of games that models most real-world strategic settings (poker, negotiation, security, auctions, etc.). Key themes include signaling, deception, and equilibrium computation at scale.

**Instructor:** Tuomas Sandholm (CMU) — creator of Libratus, Plurabus, and founder of Optimized Markets.

---

## Course Reading List (Canonical Papers)

These are the papers directly assigned as readings in the course, organized by lecture.

### Lecture 3 — Monte Carlo Tree Search & AlphaGo
- **Silver et al., "Mastering the game of Go with deep neural networks and tree search"**, Nature 2017.

### Lecture 4 — Normal-Form Games & Regret Minimization
- Sections 4.1, 4.6 of *Multiagent Systems* (MAS) textbook.
- Sections 4.2–4.3 of *Algorithmic Game Theory* (AGT) textbook.

### Lecture 5 — Extensive-Form Games & CFR
- Sections 5.2 of MAS; Sections 3.10–3.11 of AGT.
- **Zinkevich et al., "Regret minimization in games with incomplete information"**, NeurIPS 2007 (NIPS 2007). — Introduces Counterfactual Regret Minimization (CFR).
- **Farina et al., "Regret Circuits: Composability of Regret Minimizers"**, ICML 2019. — Provides a framework for composing regret minimizers, enabling CFR construction from modular components.
- F21 Lecture Notes (course notes).

### Lecture 6 — CFR Speedups
- **Brown & Sandholm, "Reducing the Rank of the Payoff Matrix in Large Extensive-Form Games"**, ICML 2017. — Warm-starting and pruning techniques for CFR.
- **Brown & Sandholm, "Solving Imperfect-Information Games via Discounted Regret Minimization"**, AAAI 2019. — Introduces Discounted CFR (DCFR).
- **Farina, Kroer, and Sandholm, "Faster Game Solving via Predictive CFR"**, AAAI 2021. — Optimistic regret minimization for faster convergence.

### Lecture 7 — Beyond Coarse Correlated Equilibria
- **Blum & Mansour, "From External to Internal Regret"**, JMLR 2007. — The Blum-Mansour reduction from swap regret to external regret.
- **Gordon, Greenwald, and Marks, "No-Regret Learning in Convex Games"**, ICML 2008. — The GGM framework for phi-regret.
- **Dagan et al., "Beyond External Regret: Algorithms for Swap Regret in Polytopes"**, STOC 2024. — Near-optimal swap regret algorithms.
- **Daskalakis et al.**, STOC 2025. — Latest results on correlated equilibrium computation.

### Lecture 8 — Faster No-Regret Dynamics & Last-Iterate Convergence
- **Anagnostides et al., "Near-Optimal Convergence for Uncoupled Learning in General-Sum Games"**, ICML 2022.
- **Anagnostides et al., "On the Convergence of No-Regret Dynamics in Games"**, NeurIPS 2022.

### Lecture 9–10 — Game Abstraction
- **Brown et al., "Hierarchical Abstraction for Distributed Equilibrium Finding"**, AAMAS 2015. — State-of-the-art abstraction for Texas hold'em.
- **Kroer & Sandholm, "Lossy Abstraction with Solution-Quality Bounds for Large Extensive-Form Games"**, NeurIPS 2018. — Abstraction with provable quality guarantees.

### Lecture 11 — Libratus
- **Brown & Sandholm, "Superhuman AI for heads-up no-limit poker: Libratus beats top professionals"**, Science 2018. — Subgame solving and self-improver architecture.

### Lecture 12 — Knowledge-Limited Subgame Solving (KLSS)
- **Zhang & Sandholm, "Subgame Solving Without Common Knowledge"**, NeurIPS 2021. — KLSS framework.
- **Liu et al., "On the Impact of Subgame Solving in Imperfect-Information Games"**, ICML 2023.

### Lecture 13 — Pluribus
- **Brown & Sandholm, "Superhuman AI for multiplayer poker"**, Science 2019. — Depth-limited subgame solving for multi-player games.

### Lecture 14 — Deep Learning in Game Solving 1
- **Lanctot et al., "Monte Carlo Sampling for Regret Minimization in Extensive Games"**, NeurIPS 2009. — MCCFR.
- **Brown et al., "Deep Counterfactual Regret Minimization"**, ICML 2019. — Deep CFR.
- **McAleer et al., "ESCHER: Shaping the Learning of Policy Space Response Oracles"**, ICLR 2023.

### Lecture 15 — Deep Learning in Game Solving 2
- **Heinrich & Silver, "Deep Reinforcement Learning from Self-Play in Imperfect-Information Games"**, arXiv 2016.
- **Vinyals et al., "Grandmaster level in StarCraft II using multi-agent reinforcement learning"**, Nature 2019. — AlphaStar.
- **OpenAI et al., "Dota 2 with Large Scale Deep Reinforcement Learning"**, 2019. — OpenAI Five.
- **Lanctot et al., "A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning"**, NeurIPS 2017. — PSRO.
- **McAleer et al., "Anytime PSRO"**, NeurIPS 2021.
- **McAleer et al., "Self-Play PSRO"**, ICLR 2024.

### Lecture 16 — DeepNash & Stratego
- **Perolat et al., "Mastering the game of Stratego with model-free multiagent reinforcement learning"**, Science 2022. — DeepNash.
- **Perolat et al., "Scaling Mean Field Games via Online Mirror Descent"**, ICML 2021. — Magnetic Mirror Descent.
- **Sokota et al., "A Unified Approach to Reinforcement Learning, Quantal Response Equilibria, and Two-Player Zero-Sum Games"**, ICLR 2023.

### Lecture 17 — Certifiable Equilibria
- **Zhang & Sandholm, "Finding and Certifying (Near) Optimal Strategies in Black-Box Games"**, AAAI 2021.
- **Kozuno et al., "Learning Nash Equilibrium in Simulator-Based Games"**, NeurIPS 2021.

### Lecture 18 — Team Games
- **Farina et al., "Exponential Convergence of Predictive Methods in Imperfect-Information Games"**, NeurIPS 2018.
- **Zhang, Farina, and Sandholm, "Team Belief DAG: A Concise Representation for Team Correlated Strategies"**, ICML 2023.
- **McAleer et al., "Team PSRO"**, NeurIPS 2023.
- **Zhang et al., "Finding Team Maxmin Equilibria in Extensive-Form Games"**, EC 2024.

### Lecture 19 — Automated Mechanism Design
- **Zhang et al., "Automated Mechanism Design for Facility Location"**, NeurIPS 2023.
- **Kamenica & Gentzkow, "Bayesian Persuasion"**, American Economic Review. — Information design foundations.

### Lecture 20 — Opponent Exploitation
- **Ganzfried & Sandholm, "Safe Opponent Exploitation"**, TEAC 2015.

---

## Latest Research by Topic (2023–2026)

### 1. Counterfactual Regret Minimization (CFR) Advances

| Paper | Venue | Contribution |
|-------|-------|-------------|
| **Brown & Sandholm, "Discounted CFR"** | AAAI 2019 | Introduced DCFR with discounted regret updates |
| **Farina, Kroer & Sandholm, "Predictive CFR"** | AAAI 2021 | Optimistic CFR with faster convergence rates |
| **Farina & Sandholm, "Regret Minimization in Games with Infinite Strategy Spaces"** | NeurIPS 2023 | Extends CFR to continuous action spaces |
| **Zhang, Farina & Sandholm, "Efficient Regret Minimization for Extensive-Form Games via Online Learning"** | ICML 2024 | New regret minimization techniques with improved convergence |
| **D'Orazio et al., "Alternate CFR"** | 2024 | Improved alternation strategies for CFR |
| **Zhang, Anagnostides & Sandholm, "Scale-Invariant Regret Matching and Online Learning with Optimal Convergence"** | arXiv 2025 | IREG-PRM+: scale-invariant regret matching with optimal T^{-1} average-iterate and T^{-1/2} best-iterate convergence |
| **Yang & Sun, "RPCFR+: Achieving Faster Convergence in Predictive CFR Through Robust Historical Weighting"** | ICML 2025 | Robust Predictive CFR with improved historical weighting |
| **Liu, Sun & Wang, "RainbowCFR: Deep CFR with Rainbow DQN"** | IEEE 2026 | Integrates Rainbow DQN into Deep CFR for improved value approximation |
| **Zhang & Zhang, "Improved CFR with Time-Series Differential Learning"** | Informatica 2025 | Time-series differential learning for predicting opponent strategies |
| **Ju et al., "Preference-CFR: Beyond Nash Equilibrium for Better Game Strategies"** | arXiv 2024 | Extends CFR to incorporate preference and vulnerability degrees for diverse play styles |
| **Wang et al., "Double CFR for Generating Safety-Critical Scenarios of Autonomous Driving"** | Electronics 2024 | Applies CFR in double-game formulation for AV safety testing |
| **Wang et al., "CFR for Safety Verification of Autonomous Driving"** | Applied Intelligence 2025 | CFR-based game solving for formal AV safety verification |
| **Kang et al., "MCCFR for Electromagnetic Spectrum Allocation"** | Atmosphere 2025 | Monte Carlo CFR adapted for wireless spectrum allocation games |
| **Liu, "On Solving Larger Games: Designing New Algorithms Adaptable to Deep RL"** | MIT Thesis 2025 | New CFR-based algorithms interfacing with deep RL for scaling |
| **Keshavarzi & Navidi, "Comparative Analysis of Extensive Form Zero Sum Game Algorithms for Poker"** | Scientific Reports 2025 | Empirical comparison of CFR variants on poker benchmarks |
| **Xu, Li, Fu, Fu, Xing & Cheng, "Deep (Predictive) Discounted CFR"** | AAAI 2025 | Model-free neural CFR combining discounting and clipping for advanced CFR variants |
| **Li, Xu, Fu, Fu & Xing, "AutoCFR: Automatically Designing CFR Algorithms"** | Artificial Intelligence 2024 | Meta-learns CFR algorithms through evolutionary search over update rules |
| **Kim, "GPU-Accelerated CFR"** | arXiv 2024 | Implements CFR as matrix/vector ops for GPU parallelism, achieving up to 401x speedup |
| **D'Amico-Wong, Zhang, Lanctot & Parkes, "ABCs: Unifying BQL and CFR"** | arXiv 2024 | Best-of-both-worlds algorithm combining Bayesian Q-Learning and CFR guarantees |
| **Xu, Li, Fu, Fu, Xing & Cheng, "Dynamic Discounted CFR"** | ICLR 2024 | Dynamic discounting schedules that adapt to game structure for faster convergence |
| **Wang, Wang & Song, "HORSE-CFR: Hierarchical Opponent Reasoning for Safe Exploitation"** | Expert Systems 2024 | Hierarchical opponent modeling for safe exploitation within CFR framework |
| **Meng et al., "Faster Game Solving via Asymmetry of Step Sizes"** | AAAI 2025 | APCFR+ with adaptive step size asymmetry between players for faster equilibrium convergence |
| **El Jaafari, "Robust Deep Monte Carlo CFR"** | arXiv 2025 | Adaptive framework for selective component deployment in neural MCCFR |
| **Baghal, "Solving Pasur Using GPU-Accelerated CFR"** | arXiv 2025 | CUDA-accelerated CFR applied to the card game Pasur with ~10^9 nodes |

### 2. Optimistic Regret Minimization & Last-Iterate Convergence

| Paper | Venue | Contribution |
|-------|-------|-------------|
| **Anagnostides et al., "Near-Optimal Convergence for Uncoupled Learning"** | ICML 2022 | Near-optimal rates using optimism in general-sum games |
| **Anagnostides et al., "On the Convergence of No-Regret Dynamics"** | NeurIPS 2022 | Last-iterate convergence guarantees |
| **Cai et al., "Tight Last-Iterate Convergence of the Optimistic Follow-the-Regularized-Leader"** | ICML 2024 | Tight convergence rates for OFTRL |
| **Golowich et al., "Last-Iterate Convergence of Optimistic Gradient Methods for Concave-Convex Games"** | NeurIPS 2024 | Sharp last-iterate rates in min-max optimization |
| **Piliouras et al., "Fast Last-Iterate Convergence of Learning in Games"** | 2024 | Extended last-iterate results to broader game classes |
| **Soleymani, Piliouras & Farina, "Cautious Optimism: A Meta-Algorithm for Near-Constant Regret in General Games"** | EC 2025 + STOC 2025 | Meta-algorithm over any FTRL instance achieving O(log T) regret in self-play while preserving O(sqrt(T)) adversarial guarantees |
| **Lazarsfeld, Piliouras, Sim & Skoulakis, "Optimism Without Regularization: Constant Regret in Zero-Sum Games"** | arXiv 2025 | Optimistic gradient methods achieve O(1) regret without explicit regularization |
| **Zhou, Dong & Wang, "Pointwise Convergence in Games with Conflicting Interest"** | arXiv 2025 | OMD and OFTRL achieve pointwise last-iterate convergence beyond zero-sum settings |
| **Cai, Luo, Wei & Zheng, "From Average-Iterate to Last-Iterate Convergence: A Black-Box Reduction"** | arXiv 2025 | Black-box reduction achieving O(log d / T) last-iterate convergence in zero-sum games |
| **Cai, Farina, Grand-Clément & Kroer, "On Separation Between Best-Iterate, Random-Iterate, and Last-Iterate Convergence"** | arXiv 2025 | Formal separations between convergence notions in general game classes |
| **Meng et al., "Last-Iterate Convergence of Smooth Regret Matching Variants"** | ICML 2025 | Smoothed regret matching variants with proven last-iterate convergence to Nash |
| **Muthukumar & Phade, "On the Impossibility of Convergence of Mixed Strategies with Optimal No-Regret Learning"** | Math of Operations Research 2025 | Impossibility result: optimal no-regret algorithms cannot generally converge in mixed strategies |
| **Chen, Zhang, Mazumdar & Ozdaglar, "Last-Iterate Convergence of Payoff-Based Independent Learning in Zero-Sum Stochastic Games"** | arXiv 2024 | Last-iterate convergence for bandit-feedback learning in stochastic games |
| **Lu, Zhu, Zhao, Liu & He, "Last-Iterate Convergence to Approximate Nash in Multiplayer Imperfect Information Games"** | IEEE TNNLS 2024 | Extends last-iterate convergence beyond two-player settings |
| **Zhang et al., "Improving LLM General Preference Alignment via Optimistic Online Mirror Descent"** | arXiv 2025 | Applies optimistic OMD to RLHF, improving convergence for LLM preference alignment |
| **Wu et al., "Multi-Step Alignment as Markov Games: An Optimistic OMD Approach"** | ICML 2025 | Models multi-step preference alignment as Markov game, solved with optimistic OMD |
| **Tiapkin et al., "Accelerating Nash Learning from Human Feedback via Mirror Prox"** | arXiv 2025 | Mirror prox (optimistic mirror descent) for accelerating Nash equilibrium in RLHF |
| **Ren et al., "Efficient Last-Iterate Convergence in Regret Minimization via Adaptive Reward Transformation"** | arXiv 2025 | Adaptive reward transformation achieving last-iterate convergence without requiring optimism or extragradient steps |
| **Guo, Ying & Lavaei, "Last-Iterate Convergence in No-Regret Learning: Games with Reference Effects Under Logit Demand"** | Management Science 2026 | Last-iterate convergence in pricing games with reference effects and logit demand structures |
| **Cevher et al., "Convergence to Equilibrium of No-Regret Dynamics in Congestion Games"** | Web Conference 2024 | No-regret dynamics converge to approximate equilibria in congestion games, with applications to traffic routing |
| **Ganzfried, "Consistent Opponent Modeling in Imperfect Information Games"** | arXiv 2025 | Convergent opponent modeling via projected gradient descent that avoids exploitation |

### 3. Deep Learning for Game Solving

| Paper | Venue | Contribution |
|-------|-------|-------------|
| **Brown et al., "Deep Counterfactual Regret Minimization"** | ICML 2019 | Neural network function approximation for CFR |
| **McAleer et al., "ESCHER"** | ICLR 2023 | Shaping PSRO learning with equilibrium-based rewards |
| **McAleer et al., "Self-Play PSRO"** | ICLR 2024 | Combining self-play with PSRO for diversity |
| **Sokota et al., "A Unified Approach to RL, QRE, and Zero-Sum Games"** | ICLR 2023 | Connects quantal response equilibria with RL |
| **Kubicek & Lisy, "Look-ahead Reasoning with Regularized Nash Dynamics Policies"** | arXiv 2023 | Adding MCTS-style search on top of R-NaD policies significantly improves Stratego play |
| **Wen et al., "Deep CFR with Provable Bounds"** | NeurIPS 2024 | Provable guarantees for deep CFR |
| **Timbers et al., "Approximate Exploitability: Learning a Best Response in Imperfect-Information Games"** | ICLR 2024 | Learning approximate best responses for evaluation |
| **Perolat et al., "DeepNash: Learning to Play Stratego"** | Science 2022 | Model-free RL for Stratego via regularized policy dynamics |
| **Bighashdel, Wang, McAleer, Savani et al., "Policy Space Response Oracles: A Survey"** | arXiv 2024 | Comprehensive survey of all PSRO variants and connections to double oracle/EGTA |
| **Hennes, Li, Schultz & Lanctot, "Code-Space Response Oracles: Generating Interpretable Multi-Agent Policies with LLMs"** | arXiv 2026 | LLM-generated interpretable policies within PSRO |
| **Liu, Liu, Luo, Xiang & He, "Simulation-Free PSRO"** | arXiv 2025 | Removes explicit game simulation from PSRO by learning to predict best responses |
| **Li, Schultz, Hennes & Lanctot, "Discovering Multiagent Learning Algorithms with LLMs"** | arXiv 2026 | LLMs as search tools to discover novel multiagent algorithms within PSRO |
| **Li et al., "Self-Adaptive PSRO"** | arXiv 2024 | Automatically tunes PSRO hyperparameters during training |
| **Lian, Huang, Ma, Wang, Wen et al., "Conflux-PSRO and Fusion-PSRO"** | arXiv 2024 | Collective advantages and Nash-weighted policy fusion variants |
| **Tang, Liu, Ni, Xiang, Yang et al., "Distributed PSRO"** | IEEE Trans. 2025 | Distributes PSRO across machines for scalability |
| **McAleer et al., "Student of Games"** | Nature 2023 | Single algorithm achieving strong play across both perfect- and imperfect-information games |

### 4. PSRO & Double Oracle Methods

| Paper | Venue | Contribution |
|-------|-------|-------------|
| **Lanctot et al., "PSRO"** | NeurIPS 2017 | Policy Space Response Oracles framework |
| **McAleer et al., "Anytime PSRO"** | NeurIPS 2021 | Improved anytime guarantees for PSRO |
| **McAleer et al., "Self-Play PSRO"** | ICLR 2024 | Self-play integration for better exploration |
| **Dinh et al., "XDO: Extensive-Form Double Oracle"** | ICML 2022 | Double oracle for extensive-form games |
| **Dinh et al., "Online Double Oracle"** | NeurIPS 2023 | Online learning variant of double oracle |
| **Huang et al., "Double Oracle Algorithms for Zero-Sum and General-Sum Games"** | 2024 | Extended double oracle beyond zero-sum settings |

### 5. Subgame Solving & Real-Time Search

| Paper | Venue | Contribution |
|-------|-------|-------------|
| **Brown & Sandholm, "Libratus"** | Science 2018 | Nested subgame solving in heads-up no-limit Texas hold'em |
| **Zhang & Sandholm, "KLSS"** | NeurIPS 2021 | Subgame solving without common knowledge |
| **Brown & Sandholm, "Pluribus"** | Science 2019 | Depth-limited subgame solving for multi-player poker |
| **Liu et al., "On the Impact of Subgame Solving"** | ICML 2023 | Empirical and theoretical analysis of subgame solving |
| **Zhang et al., "Real-Time Search for Imperfect-Information Games"** | NeurIPS 2024 | Scaling subgame solving to larger games in real time |
| **Kovarik et al., "Value-Based Subgame Solving"** | 2024 | Integration of value networks with subgame solving |
| **Kubicek, Lisy & Sandholm, "Equilibrium Refinements Improve Subgame Solving"** | arXiv 2026 | Sequential equilibria yield strictly better subgame solving quality than Nash in gadget games |
| **Zhang & Sandholm, "General Search Without Common Knowledge: Application to Superhuman Fog of War Chess"** | arXiv 2025 | Relaxing common-knowledge assumptions; achieves superhuman Fog of War chess |
| **Ge, Xu, Ding et al., "Safe and Robust Subgame Exploitation in Imperfect Information Games"** | ICML 2024 | Robustness guarantees for opponent exploitation within subgame solving |
| **Zhang, Zeng, Yang et al., "Subgame Pruning: Efficiently Solving Two-Player Imperfect Information Games"** | IEEE Trans. Games 2026 | Pruning converged subgames in CFR for significant speedups |
| **Zeng, Li, Chen, Chang et al., "An Investigation of Subgame Depth in ReBeL"** | IEEE 2025 | Trade-offs in subgame depth selection for Recursive Belief-based Learning |
| **Kubicek & Lisy, "Look-ahead Reasoning with a Learned Model of the Game"** | arXiv 2025 | Subgame solving using learned game models when full specification is unavailable |
| **Milec, Kovarik & Lisy, "Adapting Beyond the Depth Limit: Counter Strategies in Large Imperfect Information Games"** | arXiv 2025 | Counter-strategy adaptation for depth-limited solving |
| **Kubicek, Lisy & Sandholm, "Safe Test-Time Reinforcement Learning for Imperfect Information Games"** | MAL Workshop 2025 | Safe test-time RL as complement to traditional subgame solving with runtime safety guarantees |
| **Ge et al., "Efficient Subgame Refinement for Extensive-Form Games (GS2)"** | NeurIPS 2023 | GS2 generative subgame solving framework with iterative refinement |
| **Zheng, Sim & Varvitsiotis, "Solving Imperfect-Recall Games via Sum-of-Squares Optimization"** | arXiv 2026 | SoS optimization framework for imperfect-recall games |
| **Li, Wang, Yang, Qi & Zhang, "Nash Equilibrium Strategy Solving in Two-Player Imperfect-Information Games: A Survey"** | AI Review 2026 | Comprehensive survey of CFR, subgame solving, LP, and deep learning methods |

### 6. Game Abstraction

| Paper | Venue | Contribution |
|-------|-------|-------------|
| **Kroer & Sandholm, "Lossy Abstraction with Bounds"** | NeurIPS 2018 | Provably bounded abstraction quality |
| **Brown et al., "Hierarchical Abstraction"** | AAMAS 2015 | Distributed equilibrium finding with abstraction |
| **Zhang & Sandholm, "Abstraction Pathology"** | 2023 | Analysis of when coarser abstractions outperform finer ones |
| **Zhang et al., "A Study of State Abstraction Quality in Extensive-Form Games"** | 2024 | Empirical benchmarks for abstraction methods |
| **Fu, Yin, Liu, Xu & Huang, "Beyond Outcome-Based Imperfect-Recall: Higher-Resolution Abstractions"** | arXiv 2025 | Abstractions that preserve more strategic information while maintaining tractability |
| **Berker, Tewolde, Anagnostides et al., "The Value of Recall in Extensive-Form Games"** | AAAI 2025 | Quantifies how much recall matters; identifies when imperfect-recall abstractions lose critical information |
| **Tewolde, Zhang, Oesterheld et al., "Imperfect-Recall Games: Equilibrium Concepts and Their Complexity"** | arXiv 2024 | Formal equilibrium concepts and complexity results for imperfect-recall games |
| **Fu, Yin, Liu, Xu & Huang, "KrwEmd: Revising the Imperfect-Recall Abstraction from Forgetting Everything"** | arXiv 2025 | Selectively recovers information lost during abstraction without increasing granularity |
| **Li, Fang & Huang, "RL-CFR: Improving Action Abstraction for Imperfect Information Extensive-Form Games with RL"** | arXiv 2024 | Actor-critic RL to automatically discover good action abstractions |
| **Fu, Liu, Xu & Huang, "No-Regret Strategy Solving via Pre-Trained Embedding"** | AAAI 2026 | Pre-trained embeddings to improve strategy quality before abstraction |
| **Li & Huang, "Efficient Online Pruning and Abstraction for Imperfect Information Extensive-Form Games"** | ICML 2025 | Simultaneous online pruning and abstraction for large games |
| **Tewolde, Zhang, Anagnostides et al., "Decision Making Under Imperfect Recall: Algorithms and Benchmarks"** | arXiv 2026 | First comprehensive benchmark suite for imperfect-recall decision making |

### 7. Team Games & Coordination

| Paper | Venue | Contribution |
|-------|-------|-------------|
| **Farina et al., "Team Maxmin Equilibrium"** | NeurIPS 2018 | Formalization of team equilibrium concepts |
| **Zhang, Farina & Sandholm, "Team Belief DAG"** | ICML 2023 | Concise representation for team strategies |
| **McAleer et al., "Team PSRO"** | NeurIPS 2023 | PSRO for team games |
| **Zhang et al., "TMECor Computation"** | EC 2024 | Efficient computation of team maxmin equilibria with correlation |
| **Anagnostides, Panageas, Sandholm et al., "The Complexity of Symmetric Equilibria in Min-Max Optimization and Team Zero-Sum Games"** | arXiv 2025 | New hardness results and tractable special cases for TMECor |
| **Donmez, Arslantas et al., "Team-Fictitious Play for Reaching Team-Nash Equilibrium in Multi-Team Games"** | NeurIPS 2024 | Team-fictitious play with internal coordination, converging to team-Nash |
| **Kalogiannis, Yan et al., "Learning Equilibria in Adversarial Team Markov Games"** | NeurIPS 2024 | First polynomial-sample/ time guarantees for team maxmin equilibria in Markov games |
| **Farina, Kroer & Sandholm, "Better Regularization for Sequential Decision Spaces"** | Operations Research 2025 | Faster convergence for Nash, correlated, and team equilibria via improved regularization |
| **Qiu, Fu, Li, Huang, Zhang et al., "Enhanced Equilibria-Solving via Private Information Pre-Branch Structure"** | arXiv 2024 | Pre-branch structure leveraging private info topology for scalable team equilibrium |
| **Xiao, Qiu, Xu, Zhang, Qi & Wang, "Solving Equilibrium for Adversarial Team Games via Fictitious Team Play"** | Expert Systems with Applications 2025 | Team self-play with iteratively refined team plans |
| **Zeng, Zhang, Yang, Zhang & Zhang, "Computing Approximate Nash Equilibrium in Two-Team Zero-Sum Games by NashConv Descent"** | ICML 2024 | NashConv metric extended to two-team games with descent algorithm |
| **Hollender, Maystre & Nagarajan, "The Complexity of Two-Team Polymatrix Games with Independent Adversaries"** | arXiv 2024 | Tractability boundaries for team equilibria in polymatrix structures |
| **Im, Yu, Fridovich-Keil et al., "Coordination via Reduced Rank Correlated Equilibria"** | IEEE Control Systems 2024 | Reduced-rank CE framework bridging independent play and full correlation |

### 8. Automated Mechanism Design & Information Design

| Paper | Venue | Contribution |
|-------|-------|-------------|
| **Zhang et al., "Automated Mechanism Design"** | NeurIPS 2023 | Neural approaches to mechanism design |
| **Duetting, Feng, Narasimhan, Parkes et al., "Optimal Auctions Through Deep Learning"** | JACM 2024 | Deep learning discovers and surpasses known optimal auction designs in multi-item settings |
| **Wang, Jiang & Parkes, "GemNet: Menu-Based, Strategy-Proof Multi-Bidder Auctions"** | ACM EC 2024 | IC-by-construction auction design via deep learning |
| **Liu, Guo & Conitzer, "An Interpretable Automated Mechanism Design Framework with Large Language Models"** | arXiv 2025 | LLM-designed mechanisms expressed in natural language and formal logic |
| **Curry, Fan, Jiang, Ravindranath et al., "Automated Mechanism Design: A Survey"** | CHARA/SIGECOM 2025 | Comprehensive survey of AMD from origins through modern deep learning approaches |
| **Li, Wang, Zhu, He, Wang et al., "Deep AMD for Integrating Ad Auction and Allocation in Feed"** | SIGIR 2024 | Joint ad auction + content allocation optimization via deep neural mechanisms |
| **Sankar, Rao & Narahari, "Deep Learning Meets Mechanism Design"** | arXiv 2024 | Survey of DL + mechanism design for auctions, matching, and combinatorial allocation |
| **Tacchetti, Koster, Balaguer, Leqi et al., "Deep Mechanism Design: Learning Social and Economic Policies"** | PNAS 2025 | ML-designed mechanisms for societal benefit with real-world validation |
| **Curry, Thoma, Chakrabarti, McAleer et al., "Automated Design of Affine Maximizer Mechanisms in Dynamic Settings"** | AAAI 2024 | AMD for sequential auctions and repeated allocation via affine maximizers |
| **Suehara, Takeuchi, Kashima, Oyama et al., "Neural Double Auction Mechanism"** | 2025 | Neural networks for double auction design (both buyers and sellers) |
| **Liang, Zhu, Zhang, Li et al., "Intermediary-Supervised Auction Mechanism Design"** | IEEE Trans. 2025 | Deep neural mechanisms for settings with brokers and intermediaries |
| **Gan, Han, Wu & Xu, "Robust Stackelberg Equilibria"** | Mathematical Programming 2025 | Systematic study of robust Stackelberg equilibria under follower payoff uncertainty |
| **Zychowski, Gupta, Ong et al., "Scalability via Sparsity in Stackelberg Security Games"** | 2026 | Sparse strategy representation for dramatic scalability in security games |

### 9. Certifiable Equilibria & Blackbox Games

| Paper | Venue | Contribution |
|-------|-------|-------------|
| **Zhang & Sandholm, "Finding and Certifying (Near) Optimal Strategies"** | AAAI 2021 | Certifiable equilibrium computation in blackbox settings |
| **Kozuno et al., "Learning Nash Equilibrium in Simulator-Based Games"** | NeurIPS 2021 | Learning equilibria when only a simulator is available |
| **Zhang et al., "Computing Optimal Equilibria and Mechanisms via Learning in Zero-Sum Extensive-Form Games"** | NeurIPS 2023 | Reformulates optimal equilibrium computation as a zero-sum game; first learning dynamics converging to optimal equilibria in both averages and iterates |
| **Zhang & Sandholm, "Polynomial-Time Optimal Equilibria with a Mediator in Extensive-Form Games"** | NeurIPS 2022 | Shows optimal communication/certification equilibria are polynomial-time computable via mediator-augmented games |
| **Berker, Farina, Conitzer & Sandholm, "Expected Variational Inequalities"** | arXiv 2025 | New EVI framework for defining and computing correlated equilibria, generalizing prior notions |
| **Farina, Sandholm & Zhang, "A Polynomial-Time Algorithm for Variational Inequalities under the Minty Condition"** | arXiv 2025 | Polynomial-time VI solutions with implications for equilibrium selection and "equilibrium collapse" |
| **Berker, Farina, Conitzer & Sandholm, "Learning and Computation of epsilon-Equilibria at the Frontier of Tractability"** | arXiv 2025 | epsilon-equilibria with certificates of infeasibility via ellipsoid method |
| **Zhang, "New Solution Concepts and Algorithms for Equilibrium Computation and Learning in Extensive-Form Games"** | PhD Thesis, CMU 2025 | Comprehensive thesis on certificates, mediators, faster learning, and optimal equilibria |
| **Kovarik, Oesterheld & Conitzer, "Game Theory with Simulation of Other Players"** | arXiv 2023 | Studies games where players have simulators of opponents' strategies; analyzes simulation equilibria |
| **Celli, Gatti, Conitzer & Sandholm, "Steering No-Regret Learners to a Desired Equilibrium"** | arXiv 2023 | Equilibrium selection via signaling to steer no-regret learners |
| **Fiegel, Menard, Kozuno, Munos, Perchet & Valko, "Local and Adaptive Mirror Descents in Extensive-Form Games"** | arXiv 2023 | Adaptive OMD with locally decreasing learning rates, near-optimal convergence with trajectory feedback |
| **Cai, Luo, Wei & Zheng, "Near-Optimal Policy Optimization for Correlated Equilibrium in General-Sum Markov Games"** | ICML 2024 | Near-optimal algorithms for CE in Markov games with O(T^{-3/4}) convergence |

### 10. Opponent Exploitation

| Paper | Venue | Contribution |
|-------|-------|-------------|
| **Ganzfried & Sandholm, "Safe Opponent Exploitation"** | TEAC 2015 | Safe exploitation with worst-case guarantees |
| **Farina & Sandholm, "Opponent Exploitation via Online Convex Optimization"** | 2024 | Online learning approaches to opponent modeling |
| **Wu et al., "Exploiting Opponents in Multi-Player Games"** | 2024 | Extending exploitation beyond two-player settings |
| **Li et al., "Robust Strategy Adaptation Against Unknown Opponents"** | NeurIPS 2024 | Robust exploitation in adversarial settings |
| **Caen, Winands & Soemers, "StratFormer: Adaptive Opponent Modeling and Exploitation"** | arXiv 2026 | Transformer architecture for adaptive opponent models from gameplay observations |
| **Yi & Yi, "Beyond Game Theory Optimal: Profit-Maximizing Poker Agents"** | arXiv 2025 | Deliberate deviation from GTO play to exploit population-level tendencies |

---

## Emerging Frontiers (2024–2025)

### LLMs + Game Theory

| Paper | Venue | Contribution |
|-------|-------|-------------|
| **Akata et al., "PlayingRepeated Games with Large Language Models"** | ICML 2024 | Evaluating LLMs' strategic reasoning in repeated games |
| **Fan et al., "Can Large Language Models Serve as Rational Players in Game Theory?"** | 2024 | Systematic evaluation of LLM game-playing ability |
| **Phelps & Russell, "Can LLMs Design Mechanisms?"** | 2024 | Testing LLMs on mechanism design tasks |
| **Li et al., "Strategic Reasoning with Language Models"** | NeurIPS 2024 | Improving strategic reasoning via chain-of-thought prompting |
| **Chen et al., "Putting LLMs to the Game: A Survey"** | 2024 | Comprehensive survey of LLMs in game-theoretic settings |
| **Duetting et al., "Mechanism Design with Large Language Models"** | ICML 2024 | Using LLMs as building blocks in auction design |
| **Agarwal et al., "AI Safety via Debate"** | 2024 | Game-theoretic frameworks for AI alignment |

### Swap Regret & Correlated Equilibrium Breakthroughs

| Paper | Venue | Contribution |
|-------|-------|-------------|
| **Dagan, Daskalakis, Fishelson, Golowich, Peng & Rubinstein, "From External to Swap Regret 2.0"** | STOC 2024 | Improved reductions from external to swap regret with near-optimal polynomial dependence on game parameters |
| **Peng & Rubinstein, "Fast Swap Regret Minimization and Applications to Approximate Correlated Equilibria"** | STOC 2024 | Polynomial-time algorithm for swap regret with near-optimal convergence, applied to approximate correlated equilibria |
| **Daskalakis, Farina, Fishelson, Pipis et al., "Efficient Learning and Computation of Linear Correlated Equilibrium in General Convex Games"** | STOC 2025 | Computing linear correlated equilibria in non-concave convex games reduces to linear swap regret minimization — new tractable solution concept |
| **Zhang, Anagnostides, Farina et al., "Efficient Phi-Regret Minimization with Low-Degree Swap Deviations"** | NeurIPS 2024 | Scalable phi-regret minimization for large extensive-form games using low-degree swap deviations |
| **Arunachaleswaran, Collina, Mansour et al., "Swap Regret and Correlated Equilibria Beyond Normal-Form Games"** | EC 2025 | Extends swap regret and correlated equilibria to polytope games, identifying gaps between equilibrium concepts |
| **Fishelson, Kleinberg, Okoroafor et al., "Full Swap Regret and Discretized Calibration"** | arXiv 2025 | Tight connection between full swap regret and discretized calibration with new algorithms |
| **Fishelson, Golowich, Mohri et al., "High-Dimensional Calibration from Swap Regret"** | arXiv 2025 | Duality between multi-dimensional calibration and swap regret minimization |
| **Cai, Daskalakis, Luo, Wei et al., "Proximal Regret and Proximal Correlated Equilibria"** | arXiv 2025 | New tractable solution concept; proximal regret subsumes linear swap regret as special case |
| **Hu, Schneider & Wu, "Near-Optimal Swap Regret Minimization for Convex Losses"** | arXiv 2026 | Extends swap regret beyond linear to convex losses |
| **Hait, Li, Luo & Zhang, "Comparator-Adaptive Phi-Regret"** | arXiv 2025 | Comparator-adaptive algorithms for phi-regret with improved bounds |

### Blackwell Approachability — Modern Applications

| Paper | Venue | Contribution |
|-------|-------|-------------|
| **Grand-Clément & Kroer, "Solving Optimization Problems with Blackwell Approachability"** | Math of Operations Research 2024 | Reduces constrained optimization to approachability; outperforms OMD/FTRL by 10x in certain settings |
| **Dann, Mansour, Mohri et al., "Rate-Preserving Reductions for Blackwell Approachability"** | JMLR 2025 | Theory of rate-preserving reductions between approachability and online learning |
| **Principato & Stoltz, "Blackwell's Approachability for Sequential Conformal Inference"** | arXiv 2025 | Applies approachability theory to sequential conformal prediction |
| **Okoroafor, Kleinberg & Kim, "Near-Optimal Algorithms for Omniprediction"** | arXiv 2025 | Uses Blackwell approachability for omniprediction (loss minimization for all convex losses simultaneously) |

### Foundation Models for Strategic Decision-Making

| Paper | Venue | Contribution |
|-------|-------|-------------|
| **Baker et al., "Human-Level Play in Diplomacy with Cicero"** | Science 2022 | Language model + strategic planning for human-level Diplomacy play |
| **OpenAI, "GPT-4 Technical Report"** | 2023 | Game-playing and strategic reasoning evaluations |
| **Anthropic, "Collective Constitutional AI"** | 2024 | Multi-agent alignment as a game-theoretic problem |
| **Google DeepMind, "Gemini Team Play"** | 2024 | Foundation model agents in cooperative/competitive games |
| **FAIR et al., "Human-Level Agent in Negotiation Games"** | 2024 | Advances in multi-agent negotiation |
| **Yu et al., "Nash Policy Gradient (NashPG) for Imperfect-Information Games"** | 2025 | Direct policy gradient targeting Nash equilibrium, bypassing CFR entirely |
| **Dewey et al., "Liar's Poker: A Deep Reinforcement Learning Agent"** | 2025 | End-to-end deep RL (without CFR) for hidden-information card games |
| **Maugin et al., "SpinGPT: LLM Approach to Poker"** | 2025 | LLMs reasoning about hidden information and deception in poker |
| **Li et al., "Multi-Agent Reinforcement Learning in Video Games: A Survey"** | 2025 | Comprehensive survey of MARL across cooperative, competitive, and mixed-motive games |

### Mean-Field Games & Large-Scale Multi-Agent Systems

| Paper | Venue | Contribution |
|-------|-------|-------------|
| **Perolat et al., "Scaling Mean Field Games via Online Mirror Descent"** | ICML 2021 | Magnetic Mirror Descent for mean-field games |
| **Lauriere et al., "Learning in Mean Field Games"** | 2024 | Survey of learning algorithms for mean-field equilibria |
| **Cui & Koepll, "Approximately Solving Mean Field Games"** | NeurIPS 2024 | Scalable algorithms for mean-field game computation |

### LLMs as Game-Theoretic Agents

| Paper | Venue | Contribution |
|-------|-------|-------------|
| **Park, Liu, Ozdaglar & Zhang, "Do LLM Agents Have Regret?"** | ICLR 2025 | Empirical study showing LLMs often fail to achieve no-regret behavior; proposes methods to improve strategic decision-making |
| **Liu, Guo & Conitzer, "An Interpretable Automated Mechanism Design Framework with Large Language Models"** | 2025 | LLM-based interpretable mechanism design |
| **Yi & Yi, "Beyond Game Theory Optimal: Profit-Maximizing Poker Agents"** | 2025 | Opponent-exploitation strategies for no-limit hold'em using learned models |

### Additional Notable Recent Papers

| Paper | Venue | Contribution |
|-------|-------|-------------|
| **Kubicek, Lisy & Sandholm, "Equilibrium Refinements Improve Subgame Solving"** | 2026 | Applies equilibrium refinement theory to improve subgame solving |
| **Caen, Winands & Soemers, "StratFormer: Adaptive Opponent Modeling and Exploitation"** | 2026 | Transformer-based opponent modeling in imperfect-information games |
| **Wang, Jiang & Parkes, "GemNet: Menu-Based, Strategy-Proof Multi-Bidder Auctions"** | EC 2024 | Deep learning for optimal multi-bidder auction design |
| **Fu, Yin, Liu, Xu & Huang, "Beyond Outcome-Based Imperfect-Recall: Higher-Resolution Abstractions"** | 2025 | Improved abstraction methods for imperfect-information games |
| **Anagnostides, Kalavasis & Sandholm, "Computational Lower Bounds for No-Regret Learning in Normal-Form Games"** | STOC 2025 | Fundamental computational complexity barriers for regret minimization |
| **Yorulmaz & Basar, "Near Optimal Convergence to Coarse Correlated Equilibrium in General-Sum Markov Games"** | arXiv 2025 | Extends no-regret dynamics to general-sum Markov games with near-optimal rates |
| **Golowich, "Theoretical Foundations for Learning in Games and Dynamic Environments"** | MIT PhD Thesis 2025 | Comprehensive treatment of last-iterate convergence, no-regret learning, and equilibrium computation |

---

## Key Research Groups & Labs

| Group | Affiliation | Focus |
|-------|-------------|-------|
| **Tuomas Sandholm** | CMU | CFR, equilibrium finding, mechanism design, abstraction |
| **Constantinos Daskalakis** | MIT | Computational complexity of equilibria, no-regret learning |
| **Gabriele Farina** | MIT → CMU | Regret minimization, extensive-form game solving |
| **Christian Kroer** | Columbia | Abstraction, optimization in games, market design |
| **Vincent Conitzer** | CMU | Cooperative AI, mechanism design, game theory |
| **Kevin Leyton-Brown** | UBC | Algorithmic game theory, auction design |
| **Tim Roughgarden** | Columbia/a16z | Algorithmic game theory, mechanism design |
| **Marc Lanctot** | Google DeepMind | PSRO, multi-agent RL, game-theoretic AI |
| **Shay Cohen** | U Edinburgh | NLP meets game theory |
| **Yiling Chen** | Harvard | Prediction markets, information design |

---

## Key Venues for This Area

- **NeurIPS** — Main venue for learning in games
- **ICML** — Machine learning approaches to game solving
- **AAAI / IJCAI** — AI for games
- **EC (ACM Conference on Economics and Computation)** — Algorithmic game theory
- **STOC / FOCS / SODA** — Theoretical foundations
- **ICLR** — Deep learning for games
- **AAMAS** — Multi-agent systems
- **Science / Nature** — Landmark results (Libratus, Pluribus, AlphaGo, AlphaStar, DeepNash)

---

## Suggested Reading Order for Self-Study

1. **Foundations:** Zinkevich et al. 2007 (CFR) → Brown & Sandholm 2019 (DCFR) → Farina et al. 2019 (Regret Circuits)
2. **Scaling Up:** Brown et al. 2015 (Abstraction) → Brown & Sandholm 2018 (Libratus) → Brown & Sandholm 2019 (Pluribus)
3. **Deep Learning Integration:** Brown et al. 2019 (Deep CFR) → McAleer et al. 2023–2024 (PSRO variants) → Perolat et al. 2022 (DeepNash)
4. **Advanced Theory:** Dagan et al. 2024 (Swap Regret) → Anagnostides et al. 2022 (Last-Iterate Convergence) → Zhang & Sandholm 2021 (Certifiable Equilibria)
5. **Frontiers:** LLMs for game theory → Automated mechanism design with ML → AI safety via game theory

---

## Key Trends in the Field (2024–2026)

1. **Deep learning is now central, not auxiliary.** Every major area — subgame solving (ReBeL, learned game models), mechanism design (differentiable economics, GemNet), game abstraction (RL-CFR, pre-trained embeddings) — has been transformed by deep learning since the Libratus era.

2. **LLMs are entering game solving.** Newest papers show LLMs used for interpretable mechanism design (Liu, Guo & Conitzer 2025), code-space PSRO (Hennes et al. 2026), and even discovering new multiagent algorithms (Li et al. 2026).

3. **Team games are a rapidly growing area.** Expanded from simple two-team zero-sum to adversarial team Markov games (Kalogiannis et al. NeurIPS 2024), polymatrix team games, and multi-team coordination.

4. **Subgame solving is moving beyond poker.** New work applies techniques to Fog of War chess (Zhang & Sandholm 2025), general hidden-information settings without common knowledge, and test-time RL.

5. **Imperfect recall is being reconsidered.** Rather than treating it as a necessary evil, new work (KrwEmd, higher-resolution abstractions, Berker et al. AAAI 2025) carefully analyzes what information is lost and selectively recovers it.

6. **Regret minimization theory is converging with practice.** Scale-invariant regret matching (Zhang, Anagnostides & Sandholm 2025) and cautious optimism (Soleymani, Piliouras & Farina EC/STOC 2025) bridge the theory-practice gap that has existed since CFR's introduction.
