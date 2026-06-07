# Deep Dive: Using LLMs to Play Games Like Solitaire

## 0. Why this is an interesting question

Solitaire (Klondike, FreeCell, Spider, etc.) is a weirdly hard testbed for LLMs. On paper it looks easier than Go or Diplomacy:

- **Single-agent** (no opponent modelling)
- **Perfect or near-perfect information** in most variants (FreeCell is fully observable; Klondike has hidden stock/waste)
- **Discrete actions** with a manageable branching factor

But it has features that are *toxic* to vanilla LLM reasoning:

1. **Long horizons.** Klondike deals can take 100+ moves; Spider more. Chain-of-thought gets brittle with depth — the longer the autoregressive trace, the more it drifts.
2. **Large, structured state.** Multiple tableaus, foundations, stock, free cells. The model has to *track* state across turns — LLMs are stateless.
3. **Stochasticity** (Klondike, Spider). Early moves irreversibly reveal cards. You need a *policy* that hedges.
4. **Forced moves / dead ends.** Getting stuck is common, so the agent must detect and backtrack.
5. **Sparse reward.** You only know if you won at the end. Credit assignment over 100+ moves is hard.

So LLM methods that work on short, single-shot games (Game of 24, mini-crosswords) don't transfer cleanly. The interesting research is in *how* you adapt them.

---

## 1. Taxonomy of methods

The literature splits into eight families. They're not mutually exclusive — most strong systems stack them.

| Family | Core idea | Representative paper |
|---|---|---|
| **A. Direct policy** | Prompt with state, ask for a move | Slay the Spire LLM agent (Khalifa et al., 2024) |
| **B. Chain-of-thought reasoning** | "Think before you act" | Standard CoT (Wei et al. 2022) |
| **C. Tree-of-thoughts / search** | Explore multiple reasoning paths, self-evaluate | ToT (Yao et al., 2023) |
| **D. LLM-as-world-model + MCTS** | LLM simulates the next state; MCTS plans | RAP (Hao et al., 2023) |
| **E. LLM as agent + value + reflection (MCTS)** | Full LATS-style loop | LATS (Zhou et al., 2023) |
| **F. Tool-use / code-as-action** | LLM writes a Python move, executes it | Cardiverse (Li et al., 2025), code-interpreter agents |
| **G. Fine-tuning on game data** | SFT/RL on strong-AI traces | "Can LLMs Master Complex Card Games?" (Wang et al., 2025) |
| **H. Self-play / iterative improvement** | Play against itself, learn from outcomes | MARS, Strat-Reasoner, DICE |

Below, the deep dive on each, with what's been shown to work and what hasn't.

---

## 2. Family-by-family

### A. Direct policy prompting

**Idea**: Pack the game state into a prompt; ask the LLM for the next move (sometimes with CoT).

The cleanest study is the Slay the Spire work from Khalifa et al. (CHIWORK 2024) — [Language-Driven Play](https://dl.acm.org/doi/fullHtml/10.1145/3649921.3650013). They used GPT-4 with rich prompts describing the current state, deck, relics, etc. **Findings:**

- GPT-4 can play Slay the Spire competently on *short-term* decisions.
- Performance degrades sharply when long-term planning is required.
- Replacing card names with random 6-character strings actually *improved* performance — because GPT-4 is biased by name semantics (e.g., "Strike" sounding strong). This is a famous and counter-intuitive result.
- Out-of-distribution card variants break the model; a follow-up (AAAI AIIDE 2025, "LLM Game Rule Understanding Through Out-of-Distribution Fine-tuning") showed targeted fine-tuning on variants helps.

For solitaire, this translates to: a vanilla GPT-4 with a state dump can probably beat a *random* player, and can plausibly win ~10–20% of Klondike deals — but won't approach a heuristic search solver.

**Key takeaway**: Direct prompting is a baseline, not a solution. It establishes floor and ceiling quickly.

### B. Chain-of-thought

The LLM is asked to reason step-by-step before producing a move. Solves some issues:

- Forces the model to enumerate legal moves first (avoids hallucinated moves).
- Lets it consider consequences ("if I move this 7 to here, then…").

**Limitations**:

- LLMs cannot reliably *simulate* a Klondike board in their head for 10+ plies. They lose track of which cards are where, especially in tableau-based games.
- Error compounds — a wrong sub-fact in move 5 poisons move 30.
- No backtracking — once it commits to a move in its trace, it doesn't revisit.

**Variants that help slightly**: self-consistency (sample N CoTs, vote), ReAct (interleave thought/action/observation), but these don't solve the core problem.

### C. Tree-of-Thoughts (ToT)

[Yao et al., 2023](https://arxiv.org/abs/2305.10601) — the canonical paper. Maintains a tree of partial reasoning states, generates K candidates at each step, self-evaluates them, and either BFS/DFS explores the most promising.

**On Game of 24 (a math puzzle)**, ToT with GPT-4 jumps from 4% (CoT) to 74% success. But Game of 24 has a tiny branching factor and a known goal. **Solitaire is qualitatively different**: branching factor 5–30, depth 50–200, and no clear terminal value function the LLM can self-evaluate reliably.

In practice ToT on solitaire dies because the *self-evaluation* step fails — the model cannot accurately score "this partial layout is more promising than that one" without actually simulating. So you get breadth without meaningful pruning.

### D. RAP — LLM as world model + MCTS

[Reasoning via Planning (Hao et al., 2023)](https://arxiv.org/abs/2305.14992). The killer idea: **use the same LLM twice**:

1. As a *world model* — given (state, action), generate the next state.
2. As a *reasoning agent* — given state, propose candidate actions.
3. MCTS uses a third LLM call (or self-eval) as a *reward* signal.

The agent grows a search tree, the LLM simulates rollouts, and the tree is back-propagated with reward.

**This is the most directly relevant paradigm for solitaire.** Solitaire has:

- A real environment (or you can use a Python simulator) — so the world model can be *replaced* by a deterministic simulator (no hallucinated next states).
- Discrete actions.
- A clear win/lose terminal.

The hybrid "**LLM proposes moves, simulator verifies**" is essentially a strong way to do RAP, and it's what most serious systems converge on.

### E. LATS — Language Agent Tree Search

[Zhou et al., 2023](https://arxiv.org/abs/2310.04406). The most general framework. Uses the LLM as:

- **Agent** (proposes actions)
- **Value function** (scores states)
- **Optimizer / reflection** (re-evaluates trajectories after seeing results)

MCTS wraps all three. LATS achieved 94.4% on HumanEval and roughly 2x'd GPT-3.5 on HotPotQA over ReAct. The framework is what's important — it generalizes RAP by adding reflection and external feedback.

For solitaire, LATS gives you a clean blueprint: simulate externally, ask LLM for move proposals, score with LLM + terminal, back-propagate, reflect on dead ends.

### F. Tool use / code-as-action

[Cardiverse (Li et al., EMNLP 2025)](https://arxiv.org/abs/2502.07128) is the cleanest example for card games. The LLM doesn't just output a move — it outputs a *Python function* that encodes the gameplay policy. The function is executed against the game state, so:

- Moves are always legal (no hallucinated moves).
- The LLM can express complex heuristics in code.
- The same framework is used both to *generate* and *play* games.

For solitaire, this is very practical: the LLM acts more like a heuristic-design assistant. You give it the rules and a goal ("maximize in-game score"), and it returns a policy function. Then you execute that function for actual play.

A simpler version: LLM outputs a move in a constrained format (e.g., JSON describing the move), the runtime validates it, executes it, and feeds the new state back.

### G. Fine-tuning on gameplay data

The most important 2025 paper for your question: [Wang et al., "Can LLMs Master Complex Card Games?" (arXiv 2509.01328)](https://arxiv.org/abs/2509.01328). Findings:

- Tested 8 card games of varying complexity (Texas Hold'em, Blackjack, Solitaire variants included).
- **LLMs SFT'd on data from strong game AIs approach the strong AI's performance**, even surpassing prompt-only baselines by large margins.
- **Multi-game training**: similar-rule games *reinforce* each other (transfer), dissimilar-rule games *conflict* (catastrophic forgetting across dissimilar rules).
- **General capabilities degrade** as games are added, but mixing in general instruction data recovers most of it.
- Tested Qwen2.5, Llama3.1, GLM4 across model sizes — the pattern holds.

**Direct implication for solitaire**: if you can get a strong Klondike/FreeCell player (heuristic search works extremely well on FreeCell, e.g., the FC-Solve project), generate ~10⁵ trajectories, SFT a small open model, you'll get an LLM that plays at near-heuristic-solver level. This is the most cost-effective path to a *strong* LLM-based solitaire player.

For training data, you don't need a neural net — heuristic solvers, A*, even optimal solvers on smaller variants work. The paper uses "high-quality gameplay interaction data" from strong game AIs and prompt templates.

### H. Self-play and iterative improvement

- **MARS** (multi-agent self-play) — have the LLM play against itself, mine trajectories, fine-tune. Classic AlphaZero shape.
- **Strat-Reasoner** — group-relative RL on self-play data; reports gains on Hanabi/TicTacToe/ConnectFour/Poker.
- **MCT Self-Refine** — use MCTS rollouts as the *reward signal* for fine-tuning, à la AlphaZero.
- **MCTS-Boosted Reasoning via Iterative Preference Learning** (NeurIPS 2024) — directly uses MCTS look-ahead to generate preference pairs for DPO.

For solitaire, self-play is awkward because it's *single-agent and the "opponent" is randomness*. Self-play is more useful in multi-agent imperfect-info games (Hanabi, Poker, Diplomacy). For solitaire, the cleaner approach is **synthetic-data generation from a strong solver** (family G) or **MCTS-rollout preference data** (family E + DPO).

---

## 3. What about solitaire specifically?

### State of the art for *non-LLM* solitaire AI

Before designing an LLM system, know what you're trying to beat:

- **FreeCell**: nearly all ~1.82M standard deals have been proven winnable with best-first search and good heuristics. Solvers like FC-Solve reach 99%+ solve rates. State representation is clean.
- **Klondike**: ~82% of standard 1-draw-3 deals are winnable, demonstrated by the *Solvitaire* project (long-running search-based solver). Solvitaire also determined winnability for 73 patience variants. Hard because of hidden cards and information-set explosion.
- **Spider Solitaire** (1/2/4 suits): deal-level winnability is known to be very high (4-suit deals: 1 in 5 deals reportedly winnable in 1-suit, much lower for 4-suit). Search-based solvers exist but struggle with the combinatorial state space.

A heuristic search Klondike player is *already very strong*. A trained LLM is competing against decades of search/RL work — the value proposition of an LLM is *not* raw strength but:

- **Generality across rules** (one model handles many solitaire variants)
- **Interpretability** (you can read its reasoning)
- **Natural-language interaction** ("I'm stuck, what should I try?")
- **Robustness to rule changes** (an LLM can adapt to a variant it hasn't seen)

### Direct LLM solitaire work (what's published)

Sparse but growing:

1. **[Wang et al. 2025 — Can LLMs Master Complex Card Games?](https://arxiv.org/abs/2509.01328)** — Solitaire is one of the 8 card games. Their pipeline (SFT on solver-generated data) is currently the most reproducible approach to get a *strong* LLM solitaire player.

2. **[Game-RL (ICLR 2026)](https://openreview.net/forum?id=e4FqU4SyHL)** — explicitly generates multimodal training data for VLMs, including **FreeCell, Spider Solitaire, and Klondike**. Their "Code2Logic" pipeline renders game frames from simulator rollouts and uses them to train VLMs that play with image inputs. Useful if your target is a vision-language model.

3. **[AIIDE 2025 — LLM Game Rule Understanding Through OOD Fine-tuning](https://ojs.aaai.org/index.php/AIIDE/article/download/36804/38942/40881)** — focused on rule variants of solitaire. Shows vanilla GPT-4 fails on rule variants even when it plays base solitaire OK; targeted SFT fixes this.

4. **[The Winnability of Klondike Solitaire and Many Other Patience Games (arXiv 1906.12314)](https://arxiv.org/html/1906.12314v5)** — Solvitaire paper, foundational for *knowing* what's solvable. Practical engine for generating training data and ground-truth winnability labels.

5. **[Cardiverse (EMNLP 2025)](https://arxiv.org/abs/2502.07128)** — the Python-policy approach. Their agent beat an LLM-as-baseline on several card game prototypes. The *idea* generalizes to solitaire cleanly.

There's a [Stanford thesis](https://escholarship.org/content/qt3qf0s989/qt3qf0s989.pdf) that specifically studies LLM solitaire variants and the impact of rule changes on gameplay. Worth reading if you're building in this space.

### What hasn't been published (as far as I can see)

- A careful head-to-head: *vanilla GPT-4 vs. heuristic Klondike player vs. MCTS-augmented LLM vs. SFT'd small model on win rate over N deals*. The literature has comparisons but not clean apples-to-apples.
- A production-grade LLM-Klondike deployment with LATS-style MCTS and reflection. The pieces exist, but the integration paper doesn't.
- RL-from-self-play on solitaire with a modern LLM. This is the obvious next paper.

---

## 4. What's actually hard, and what to do about it

| Failure mode | Why it happens | Mitigation |
|---|---|---|
| Hallucinated moves (illegal) | LLM doesn't track legal actions | Validate against simulator; constrain output format |
| State-tracking loss | Stateless model, long games | Keep explicit state; re-inject it in every prompt |
| Short-horizon myopia | CoT can't see far | MCTS over real simulator; LLM proposes, simulator scores |
| Stuck-state blindness | No mechanism to detect dead ends | Detect via simulator (no legal moves → reflect → restart) |
| Information-set confusion | Hidden cards in Klondike | Either: expose full state (relax to perfect-info) for play, or condition LLM on belief distribution |
| Compute cost | MCTS × LLM proposals is expensive | Use small fine-tuned model for proposals, big model for value; cache |
| Brittleness to rule changes | Learned heuristic overfits | SFT on diverse rule sets; explicit rules in prompt |

---

## 5. Recommended architecture (if you wanted to build one)

If your goal is the *best* LLM-based Klondike player right now, the stack would be:

1. **Use a deterministic simulator** (e.g., a Python implementation of Klondike rules). Ground truth. Free.
2. **Strong-AI data generation**: run a heuristic Klondike solver (Solvitaire, or a custom A*/MCTS) over ~10⁵ deals. Save (state, action, outcome) triples.
3. **SFT a small open model** (Qwen2.5-7B or Llama-3.1-8B) on those triples. State serialized as text or a structured format. Wang et al. 2025 show this gets you near-solver win rate.
4. **At inference, wrap the SFT'd model in an LATS-style loop**: the model proposes moves, simulator validates and applies them, and on stuck states trigger a reflection pass ("you've made no progress in 20 moves, what's wrong?"). LLM generates a new strategy; if still stuck, restart.
5. **(Optional) RL on top**: use the simulator as the reward signal, run GRPO/DPO with the SFT model as starting point. This is the Game-RL pattern.

If your goal is *research*, the more interesting play is the **LLM-as-world-model** direction: ask the LLM to predict the next state, compare against simulator, and use the *discrepancy* as a training signal. This is what [RLVR-World](https://ise.thss.tsinghua.edu.cn/~mlong/doc/RLVR-World-NeurIPS25.pdf) (NeurIPS 2025) does for text-based games and translates naturally to card games.

---

## 6. Open problems

1. **Long-horizon credit assignment.** AlphaZero's MCTS gives dense rewards; LLMs over CoT get one signal at the end. MCT Self-Refine is a start, but the gap is real.
2. **Belief tracking under partial information.** Klondike has hidden cards. Current LLM systems mostly either ignore this (play with full observability) or fail. Belief-state prompting is underexplored.
3. **Rule generalization.** The AIIDE 2025 paper shows current LLMs break on rule variants. The *Learning to Learn* version — train an agent that adapts to rules from a small description — is a clear research target.
4. **LLM world model faithfulness.** LLMs hallucinate next-states. Measuring and constraining this is open.
5. **Self-evaluation reliability.** ToT, RAP, and LATS all assume the LLM can self-score partial states. On solitaire, that assumption is shaky. Replacing self-eval with a learned value head is the obvious fix.

---

## 7. TL;DR

- **Vanilla prompting** of GPT-4 on solitaire: weak baseline, ~10–20% Klondike win rate.
- **CoT / ToT**: marginal improvement, fails on long horizons.
- **LLM-as-world-model + MCTS (RAP / LATS)**: the right *paradigm* but needs a real simulator to anchor the world model.
- **Fine-tuning on strong-AI data** ([Wang et al. 2025](https://arxiv.org/abs/2509.01328)): the most reliable path to a strong LLM solitaire player today.
- **Tool use / code-as-action** ([Cardiverse](https://arxiv.org/abs/2502.07128)): cleanest for prototyping and rule generalization.
- **Self-play / RL on top of SFT**: where the next 12 months of progress likely comes from.

The honest answer to "can LLMs play solitaire" is: *out of the box, no*. With a simulator, a small amount of fine-tuning, and a search loop around them, *yes, and they can be near-solver strength*. The interesting work isn't the prompt — it's the scaffolding.

---

## Key references

- [Wang et al. — Can LLMs Master Complex Card Games?](https://arxiv.org/abs/2509.01328) (2025)
- [Hao et al. — Reasoning via Planning (RAP)](https://arxiv.org/abs/2305.14992) (2023)
- [Zhou et al. — Language Agent Tree Search (LATS)](https://arxiv.org/abs/2310.04406) (2023)
- [Yao et al. — Tree of Thoughts](https://arxiv.org/abs/2305.10601) (2023)
- [Khalifa et al. — Language-Driven Play (Slay the Spire)](https://dl.acm.org/doi/fullHtml/10.1145/3649921.3650013) (CHIWORK 2024)
- [Li et al. — Cardiverse](https://arxiv.org/abs/2502.07128) (EMNLP 2025)
- [Game-RL (ICLR 2026)](https://openreview.net/forum?id=e4FqU4SyHL) — covers FreeCell, Spider, Klondike as VLM data
- [Solvitaire — Winnability of Patience Games](https://arxiv.org/html/1906.12314v5)
- [LLM Game Rule Understanding — AIIDE 2025](https://ojs.aaai.org/index.php/AIIDE/article/download/36804/38942/40881)
- [Shinn et al. — Reflexion](https://arxiv.org/abs/2303.11366) (verbal self-reflection)
- [MCT Self-Refine / LLM-MCTS survey pointer](https://arxiv.org/abs/2405.00451)
- [Awesome LLM Game Agent Papers (survey)](https://github.com/git-disl/awesome-LLM-game-agent-papers)
