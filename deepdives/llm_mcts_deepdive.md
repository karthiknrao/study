# Deep Dive: Monte Carlo Tree Search + Large Language Models

## Why Combine Them?

Standard LLM decoding is **greedy and irrevocable** — each token commits to a path it can't backtrack from. MCTS treats language generation as a sequential decision process where the LLM proposes candidates and a search procedure evaluates them. This enables **test-time compute scaling**: using more inference-time compute to get better answers without retraining.

---

## How LLMs Map to MCTS Components

Traditional MCTS has four phases. Here's how LLMs transform each:

| Phase | Traditional | LLM-Augmented |
|-------|------------|----------------|
| **Selection** | UCB1/UCT balances exploration vs. exploitation | UCT + LLM action priors, self-evaluation scores, or learned Process Reward Models |
| **Expansion** | Apply available actions to generate children | LLM generates *k* candidate next thoughts/steps/code edits |
| **Simulation** | Random playout to estimate value | Replaced by LLM self-refinement, world model predictions, or environment interaction |
| **Backpropagation** | Propagate reward up to root | Standard backup with LLM-derived rewards: self-scores, PRM values, pairwise comparisons |

---

## The Major Paradigms

### 1. Search Over Thought Space

**Tree of Thoughts (ToT)** — NeurIPS 2023 (Yao et al.)
- Decomposes reasoning into discrete "thought" steps arranged as tree nodes
- LLM serves as both **generator** (propose thoughts) and **evaluator** (score them)
- Uses BFS/DFS rather than full MCTS
- *Result*: Game of 24 accuracy went from 4% (CoT) → **74%** (ToT) with GPT-4

**Graph of Thoughts (GoT)** — AAAI 2024 (Besta et al.)
- Extends ToT to arbitrary directed graphs — allows merging, refining, and looping thoughts

### 2. LLM as World Model + MCTS Planning

**RAP (Reasoning via Planning)** — EMNLP 2023 (Hao et al.)
- LLM acts as a world model: predicts next state, provides action priors, estimates confidence
- Full MCTS with UCB selection over reasoning states
- Outperforms CoT and ToT on Blocksworld planning and math reasoning

**LLM-MCTS** — NeurIPS 2023 (Zhao et al.)
- LLM as commonsense world model for large-scale task planning (household robotics)
- Dramatically prunes search space vs. uninformed MCTS

### 3. Unified Agent Frameworks

**LATS (Language Agent Tree Search)** — ICML 2024 (Zhou et al.)
- Unifies reasoning, acting, and planning through MCTS
- Incorporates **self-reflection** — failed trajectories generate reflections that guide future search
- *Results*: 94.4% HumanEval (code), 75.9 on WebShop with GPT-3.5

### 4. AlphaZero-Style Training Loops

This is the most impactful emerging direction — closing the loop between MCTS inference and model training.

**ReST-MCTS*** — NeurIPS 2024 (Zhang et al.)
- MCTS infers per-step process rewards from oracle final answers
- Trains a Process Reward Model (PRM), which guides MCTS, which collects better traces, iteratively
- Directly mirrors AlphaZero's self-play loop but for language reasoning

**MCTS + Iterative Preference Learning** — NeurIPS 2024 Spotlight (Xie et al.)
- MCTS decomposes instance-level rewards into step-level preference pairs
- Trains LLM via DPO in an AlphaZero-style loop

### 5. Self-Refinement as Rollout

**MCTSr (MCT Self-Refine)** — 2024 (Zhang et al., Fudan)
- Replaces random rollouts with LLM self-critique + refinement
- *Result*: LLaMA-3 8B reaches GPT-4 level on mathematical Olympiad benchmarks

**LLaMA-Berry** — NAACL 2025
- Self-Refine MCTS + pairwise preference reward model
- Ranks reasoning paths via pairwise comparisons instead of absolute scores

### 6. Code Generation

**GIF-MCTS** — NeurIPS 2024
- Three action types: **G**enerate, **I**mprove, **F**ix within MCTS
- Nodes are code states; evaluation validates against test cases

**SRA-MCTS** — IJCAI 2025
- MCTS generates intermediate reasoning data for code, then fine-tunes smaller models on it

**Alpha-SQL** — 2025
- MCTS for zero-shot text-to-SQL, using "LLM-as-Action-Model" for SQL construction actions

### 7. Adaptive Inference-Time Scaling

**AB-MCTS** — NeurIPS 2025 (Sakana AI)
- Dynamically decides whether to "go wider" (sample new solutions) or "go deeper" (refine existing ones)
- Released as **TreeQuest** open-source framework
- State of the art for frontier coding/engineering tasks

---

## Key Insight: Small Models + Search = Large Model Performance

A consistent finding across papers: **7-8B parameter models augmented with MCTS can match or exceed GPT-4** on specific benchmarks. MCTS is a force multiplier for smaller, cheaper models.

---

## Open Challenges

1. **Speed**: MCTS is 10-100x slower than CoT. Speculative decoding techniques (SC-MCTS*) are starting to address this
2. **Reward design**: LLM self-evaluation can be poorly calibrated — pairwise comparisons (LLaMA-Berry) and PRMs help
3. **Beyond trees**: Graph structures (GoT, ThoughtSculpt) with merging/revision outperform strict trees
4. **Training integration**: The AlphaZero loop (search → better training data → better model → better search) is just beginning to be explored for LLMs

---

## Complete Paper Reference Table

| Domain | Key Papers | Benchmarks | Notable Results |
|---|---|---|---|
| **Math Reasoning** | MCTSr, LLaMA-Berry, ReST-MCTS*, MCTS+Pref Learning, RAP | GSM8K, MATH, AIME, OlympiadBench | MCTSr: LLaMA-3 8B reaches GPT-4 level on Olympiad; ReST-MCTS* exceeds ToT and Best-of-N |
| **Code Generation** | LATS, SRA-MCTS, GIF-MCTS, AlphaCode, Alpha-SQL | HumanEval, MBPP, LiveCodeBench | LATS: 94.4% HumanEval; GIF-MCTS surpasses all baselines on CWMB |
| **Task Planning** | RAP, LLM-MCTS, LGMCTS | Blocksworld, VirtualHome, household tasks | LLM-MCTS outperforms MCTS alone and LLM-as-policy alone by wide margin |
| **Game Playing** | MC Planning for Text Games | Jericho benchmark | Outperforms RL and LLM baselines |
| **Web Navigation** | LATS | WebShop | 75.9 average with GPT-3.5 |
| **General Reasoning** | ToT, GoT, ThoughtSculpt, SC-MCTS*, AB-MCTS | Game of 24, HotpotQA, creative writing | ToT: 4% → 74% on Game of 24 |
| **Mathematical Discovery** | FunSearch, MCTS-AHD | Cap set problem, bin packing | FunSearch: new mathematical constructions beyond best known |
| **Knowledge Graphs** | Graph-MCTS, REKG-MCTS | KGQA benchmarks | Consistent improvements over standard LLM augmentation |
| **SQL Generation** | Alpha-SQL | Spider, Spider-Syn | Zero-shot text-to-SQL via MCTS |

---

## Sources

- [Tree of Thoughts (NeurIPS 2023)](https://arxiv.org/abs/2305.10601)
- [Reasoning via Planning / RAP (EMNLP 2023)](https://arxiv.org/abs/2305.14992)
- [LLM-MCTS (NeurIPS 2023)](https://arxiv.org/abs/2305.14078)
- [Language Agent Tree Search / LATS (ICML 2024)](https://arxiv.org/abs/2310.04406)
- [Graph of Thoughts (AAAI 2024)](https://arxiv.org/abs/2308.09687)
- [ReST-MCTS* (NeurIPS 2024)](https://arxiv.org/abs/2406.03816)
- [MCTS Boosts Reasoning via Iterative Preference Learning (NeurIPS 2024)](https://arxiv.org/abs/2405.00451)
- [MCT Self-Refine / MCTSr](https://arxiv.org/abs/2406.07394)
- [LLaMA-Berry (NAACL 2025)](https://arxiv.org/abs/2410.02884)
- [ThoughtSculpt (NAACL Findings 2025)](https://arxiv.org/abs/2404.05966)
- [GIF-MCTS (NeurIPS 2024)](https://arxiv.org/abs/2405.15383)
- [SRA-MCTS (IJCAI 2025)](https://arxiv.org/abs/2411.11053)
- [FunSearch (Nature 2024)](https://www.nature.com/articles/s41586-023-06924-6)
- [AlphaCode (Science 2022)](https://www.science.org/doi/10.1126/science.abq1158)
- [AB-MCTS (NeurIPS 2025)](https://arxiv.org/abs/2503.04412)
- [SC-MCTS*](https://arxiv.org/abs/2410.01707)
- [AlphaLLM-CPL](https://arxiv.org/abs/2410.06508)
- [MCTS-AHD (ICML 2025)](https://arxiv.org/abs/2501.08603)
- [LE-MCTS (NAACL 2025)](https://aclanthology.org/2025.naacl-long.515.pdf)
- [REKG-MCTS (ACL Findings 2025)](https://aclanthology.org/2025.findings-acl.484.pdf)
- [Alpha-SQL](https://arxiv.org/abs/2502.17248)
- [LLM-Search Survey (TMLR 2025)](https://github.com/xinzhel/LLM-Search)
