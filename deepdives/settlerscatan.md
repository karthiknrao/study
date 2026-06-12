# AI for Settlers of Catan — Papers & Resources

A deep-dive reading list and resource map for AI approaches to the board game Settlers of Catan. Organized by topic, with the canonical entry point in each subfield called out first.

> **Quick start:** if you only bookmark one thing, make it [Catanatron](https://github.com/bcollazo/catanatron) — the open-source Python simulator most modern Catan AI research is built on. Everything below plugs into it.

---

## 1. Framework: Catanatron (the de-facto platform)

- **Catanatron** — [github.com/bcollazo/catanatron](https://github.com/bcollazo/catanatron)
  - High-performance Python simulator; runs thousands of games/minute.
  - Ships with a `Player` OOP API, a hand-crafted `AlphaBetaPlayer` (the strongest human-crafted baseline, n=2 in their leaderboard), and a Gymnasium interface for RL.
  - Web UI for inspecting/playing games at [catanatron.com](https://www.catanatron.com/).
  - **Docs:** [docs.catanatron.com](https://docs.catanatron.com/) — covers the Gym env, data/ML utilities (`create_board_tensor`, `create_sample_vector`).
  - **Leaderboard:** [PyPI page](https://pypi.org/project/catanatron/) ranks bots via head-to-head sims.
  - **Notebook example:** [Overview.ipynb on Colab](https://colab.research.google.com/github/bcollazo/catanatron/blob/master/examples/Overview.ipynb)
  - **Codebase tour:** [DeepWiki entry](https://deepwiki.com/bcollazo/catanatron)

**Why it matters:** Catan is stochastic, multi-player, partially-observable, and includes free-form negotiation. Re-implementing a fast simulator is the main barrier to entry — Catanatron eliminates it. Every approach below plugs in here.

---

## 2. Foundational MCTS work (where the field started)

Catan is the canonical "MCTS works where alpha-beta struggles" benchmark. These papers are the standard starting point.

- **Szita, Chaslot & Spronck, "Monte-Carlo Tree Search in Settlers of Catan"** (ACG 2010; ACG 2012 revised version) — [PDF](https://spronck.net/pubs/ACG12Szita.pdf) · [Springer chapter](https://link.springer.com/chapter/10.1007/978-3-642-12993-3_3)
  - The foundational paper applying MCTS to a multi-player, non-deterministic game. Demonstrates MCTS is competitive with hand-crafted heuristics.
- **Roelofs, "Monte Carlo Tree Search in a Modern Board Game Framework"** (BSc thesis, Maastricht) — [PDF](https://project.dke.maastrichtuniversity.nl/games/files/bsc/Roelofs_Bsc-paper.pdf)
  - More recent survey of MCTS for Catan and other modern board games.
- **Chaslot, "Monte-Carlo Tree Search"** (PhD thesis, Maastricht) — [PDF](https://project.dke.maastrichtuniversity.nl/games/files/phd/Chaslot_thesis.pdf)
  - Broader MCTS thesis; the Objective Monte-Carlo (OMC) variant is what Szita et al. extend for Catan.

**Background reading (general MCTS):**
- Sutton, "A Survey of Monte Carlo Tree Search Methods" — [PDF](http://www.incompleteideas.net/609%20dropbox/other%20readings%20and%20resources/MCTS-survey.pdf)
- Wikipedia overview — [Monte Carlo tree search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)
- Chessprogramming wiki — [Monte-Carlo Tree Search](https://www.chessprogramming.org/Monte-Carlo_Tree_Search)

**Note on a research gap:** I did not surface a strong, standalone DQN/PPO Catan paper in this search. Most modern Catan RL work builds on Catanatron's Gym env rather than publishing as standalone papers. This is likely because Catan's stochasticity + multi-agent dynamics + long horizons make single-agent RL a poor fit out of the box; most work targets self-play or heuristic-MCTS hybrids.

---

## 3. LLM agents for Catan (the active research frontier)

The 2024–2025 wave. Catan exposes the failure mode of "LLM as a step-wise reasoner" very clearly — long horizons, evolving state, multi-agent dynamics — and recent work has been pushing on exactly that.

### 3.1 The HexMachina paper (must-read)

- **Belle, Barnes, Amayuelas, Bercovich, Wang & Wang (UCSB), "Agents of Change: Self-Evolving LLM Agents for Strategic Planning"** (June 2025; revised Oct 2025) — [arXiv:2506.04651](https://arxiv.org/abs/2506.04651) · [HTML](https://arxiv.org/html/2506.04651v2) · [PDF](https://arxiv.org/pdf/2506.04651) · [project page](https://nbelle1.github.io/agents-of-change/) · [HuggingFace](https://huggingface.co/papers/2506.04651) · [AlphaXiv](https://www.alphaxiv.org/abs/2506.04651v1)
  - **Problem:** prompt-centric LLM agents (ReAct, Reflexion) re-interpret large, evolving game states each turn, saturate context, and lose strategic consistency.
  - **Solution — HexMachina:** a continual-learning multi-agent system that separates:
    - *Environment discovery* — inducing a Catan API adapter from scratch, without documentation.
    - *Strategy improvement* — evolving a compiled player through code refinement and self-simulation.
  - **Result:** in Catanatron head-to-heads, HexMachina beats the strongest human-crafted baseline (`AlphaBetaPlayer`) at a **54% win rate**, and beats prompt-driven / no-discovery ablations.
  - **Takeaway:** the design lesson is *artifact-centric* continual learning — preserve executable artifacts so the LLM works on *strategy*, not per-turn state re-interpretation. This generalizes beyond Catan.

### 3.2 LLM strategic-decision-making and negotiation

- **"Game-TheoreticLLM: Agent Workflow for Negotiation Games"** — [OpenReview PDF](https://openreview.net/pdf?id=Q5iCjZmpE4)
  - Evaluates GPT-4, Claude, etc. on complete- and incomplete-information games. Finds LLMs frequently deviate from rational strategies as complexity rises. Directly relevant to whether you can trust an LLM in Catan's negotiation phase.
- **"A Comprehensive Survey of Computational Persuasion"** (ACM CSUR, 2025) — [link](https://dl.acm.org/doi/10.1145/3800687)
  - Reviews persuasion/negotiation agents, including Catan trade scenarios. Reports GPT-4 is overall the best negotiating LLM in their evaluation.
- **Fu et al., "Improving Language Model Negotiation with Self-Play and In-Context Critiques"** — [Semantic Scholar](https://www.semanticscholar.org/paper/Improving-Language-Model-Negotiation-with-Self-Play-Fu-Peng/2e6b6de08f459e2165b11ed8d2103916966b0fcf)
  - Self-play + critic-LM loop for negotiation. Catan trades are a natural follow-up.

### 3.3 LLM game-agent reading lists

- **git-disl/awesome-LLM-game-agent-papers** — [GitHub](https://github.com/git-disl/awesome-LLM-game-agent-papers)
  - Maintained reading list for LLM game agents across many games. Use it to discover what general techniques transfer.
- **tmgthb/Autonomous-Agents** — [GitHub](https://github.com/tmgthb/Autonomous-Agents)
  - Broader autonomous-agent resources; useful tangentially.

---

## 4. Adjacent: Diplomacy (the negotiation reference, not Catan)

Diplomacy is the obvious sibling problem. Most of the deepest existing playbook for "LLM negotiator + strategic reasoning" lives here. If you build an LLM-in-Catan agent, these are the techniques you'd likely port.

- **Meta AI, "Human-level play in the game of Diplomacy by combining language models with strategic reasoning" (CICERO)** — [ResearchGate](https://www.researchgate.net/publication/365666035_Human-level_play_in_the_game_of_Diplomacy_by_combining_language_models_with_strategic_reasoning)
  - The canonical human-level negotiation agent. Combines a dialogue model with a strategic RL planner.
- **"Democratizing Diplomacy: A Harness for Evaluating Any Large Language Model on Full-Press Diplomacy"** — [arXiv:2508.07485](https://arxiv.org/pdf/2508.07485) · [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/41067/45028)
  - Plug-and-play harness to evaluate any LLM in Diplomacy. Worth modeling if you build a Catan-LLM eval harness.
- **"Monte Carlo Tree Search for the Game of Diplomacy"** (Paquette et al., 2020) — [ACM DL](https://dl.acm.org/doi/10.1145/3411408.3411413)
  - UCT-based MCTS for Diplomacy. Useful methodological reference for search + stochasticity + multi-agent settings.

---

## 5. Gaps and open questions

Honest notes on what my search did *not* surface — these are likely fruitful directions for new work:

- **No confirmed standalone DQN/PPO Catan paper.** RL research on Catan clearly happens (Catanatron's Gym env is a tell), but the published-academic-paper surface is thin. Either the results aren't dramatic enough for top venues, or researchers ship code rather than papers.
- **Opponent modeling in Catan specifically** is sparse. The opponent-prediction literature in Poker is much richer; Catan-specific work has room to grow.
- **Trade/dialogue as a learned subproblem** — only one focused paper (Fu et al.) surfaced. Most Catan-LLM work either dodges the negotiation phase or hand-rolls simple rules. CICERO-style methods ported to Catan are an obvious gap.
- **Standardized Catan benchmarks for LLM eval** — there's no equivalent of Diplomacy's "full-press" harness. HexMachina's Catanatron setup is the closest thing.

---

## 6. Suggested reading order

If you want to get up to speed efficiently:

1. **Catanatron docs** ([docs.catanatron.com](https://docs.catanatron.com/)) — get the simulator running, run a 1v1 of AlphaBeta vs Random.
2. **Szita et al. (2010)** — the canonical Catan-MCTS paper. Brief, foundational.
3. **Roelofs BSc thesis** — broader MCTS-in-modern-games framing, more accessible.
4. **Belle et al. (2025), "Agents of Change"** — the most recent, most relevant LLM-in-Catan paper. Read in full.
5. **CICERO** — even though it's Diplomacy, the dialogue + strategic-RL architecture is the deepest playbook for what an LLM-Catan player would need.
6. **Game-TheoreticLLM** — grounding for what LLMs are and aren't good at in negotiation games.

That order takes you from "run a simulator" to "understand the state of the art" in roughly a weekend.
