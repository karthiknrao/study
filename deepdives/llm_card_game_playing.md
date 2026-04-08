# Deep Dive: LLMs Playing Card Games

## 1. Why Card Games Matter for LLM Research

Card games are uniquely challenging for LLMs because they sit at the intersection of several hard AI problems:

- **Imperfect information** — Players can't see all cards (unlike chess/go)
- **Theory of Mind (ToM)** — Must reason about what others know/believe
- **Strategic deception** — Bluffing, slow-playing, trap-setting
- **Long-horizon planning** — Multi-turn strategies with delayed payoffs
- **Probabilistic reasoning** — Evaluating odds under uncertainty
- **Cooperative reasoning** — Some games (Hanabi) require teamwork with limited communication

This makes them richer testbeds than most standard LLM benchmarks.

---

## 2. Key Papers & Benchmarks

### Poker

**PokerBench** (AAAI 2025) — The most comprehensive poker-specific benchmark. Key findings:
- 11,000 curated Texas Hold'em scenarios
- **GPT-4 scored only 53.55% accuracy** — the best among pretrained models
- Fine-tuned Llama-3-8B reached 78.26% after supervised training
- Simple SFT is insufficient for optimal poker strategy — models need more advanced methods
- [GitHub](https://github.com/pokerllm/pokerbench) | [Paper](https://arxiv.org/abs/2501.08328)

**"How Far Are LLMs from Professional Poker Players?"** (2025) — Revisits game-theoretic reasoning with agentic tool use on Leduc Hold'em, a simplified poker variant. Tests semi-bluffing, slow-playing, and range narrowing.
- [Paper](https://arxiv.org/abs/2602.00528)

**"Readable Minds: Emergent Theory-of-Mind-Like Behavior in LLM Poker Agents"** (April 2026) — Found that:
- **Strategic deception grounded in opponent models occurs ONLY in memory-equipped conditions** (Fisher's exact p < 0.001)
- Domain expertise doesn't gate ToM emergence but enhances its application
- Tier 2 bluffing (bluffing because the opponent model predicts they'll fold) requires memory
- [Paper](https://arxiv.org/abs/2604.04157)

### Hanabi (Cooperative)

**LLM-Hanabi** (EMNLP 2025) — Uses Hanabi as a ToM testbed:
- Players can't see their own cards — must rely on sparse hints from teammates
- Tests 1st-order ToM (what does my partner intend?) and 2nd-order ToM (what does my partner think I believe?)
- **Key finding: 1st-order ToM is a stronger predictor of success than 2nd-order ToM** — understanding partner rationale matters more than recursive reasoning
- Large reasoning models (LRMs) significantly outperform standard LLMs
- [GitHub](https://github.com/HKUST-KnowComp/LLM-Hanabi) | [Paper](https://arxiv.org/abs/2510.04980)

**"Sparks of Cooperative Reasoning: LLMs as Strategic Hanabi Agents"** (2025) — Largest evaluation to date: 17 SOTA LLMs in 2-5 player Hanabi.
- [Paper](https://arxiv.org/abs/2601.18077)

### Guandan (Chinese Trick-Taking)

**"Can Large Language Models Master Complex Card Games?"** (NeurIPS 2025) — The most comprehensive multi-game study:
- Tests 8 diverse card games through supervised fine-tuning on expert gameplay data
- Includes Guandan, a complex Chinese 4-player cooperative/imperfect-info game
- Theory of Mind prompting significantly improves collaborative performance
- Reveals critical gaps in managing long-horizon states
- [GitHub](https://github.com/THUDM/LLM4CardGame) | [Paper](https://arxiv.org/abs/2509.01328)

### Trading Card Games

**TCG-Bench** (EACL 2026 Findings) — Novel approach to the data contamination problem:
- Custom-designed two-player TCG (inspired by Magic: The Gathering) that doesn't exist in training data
- Separates public game engine from private card implementations
- **Tunable difficulty controller** + bilingual (English/Arabic) evaluation
- Found a **47.4% cross-lingual performance gap** at 32B scale
- [Leaderboard](https://tcg-bench.com/) | [Paper](https://aclanthology.org/2026.findings-eacl.353/)

### General Game-Theoretic Benchmarks

| Benchmark | Focus | Venue |
|-----------|-------|-------|
| **GTBench** | Strategic reasoning via board/card games | NeurIPS 2024 |
| **TMGBench** | ToM in strategic reasoning | 2025 |
| **Game Reasoning Arena** | OpenSpiel-based framework for LLM vs LLM/RL/random | 2025 |
| **GAMEBoT** | Decomposes game reasoning into modular subproblems | ACL 2025 |
| **GameArena** | Live computer games for reasoning eval | ICLR 2025 |
| **BALROG** | Agentic reasoning on games (spatial + strategic) | ICLR 2025 |

---

## 3. Theory of Mind in Card Games

This is the core differentiator. ToM in card games manifests as:

**1st-order ToM** — "What does my opponent have?"
- Card counting, range estimation, reading betting patterns
- Critical for cooperative games (Hanabi) — understanding partner's hint intent

**2nd-order ToM** — "What does my opponent think I have?"
- Bluffing effectiveness depends on this
- Slow-playing requires manipulating opponent beliefs

**Suspicion-Agent** (COLM 2024) showed GPT-4 displays strong high-order ToM:
- Can understand others' mental states and intentionally influence behavior
- Prompt engineering (not training) enables this
- Outperforms traditional imperfect-information algorithms without specialized training
- [Paper](https://arxiv.org/abs/2309.17277) | [GitHub](https://github.com/CR-Gjx/Suspicion-Agent)

---

## 4. What LLMs Struggle With

From the literature, consistent failure modes:

1. **Basic game-theoretic reasoning** — Even SOTA models fail at fundamental strategic calculations (TMGBench, ScienceDirect 2026)

2. **Logical inconsistencies** — Models contradict themselves across equivalent game states presented differently

3. **Long-horizon state management** — Lose track of game state over many turns in complex games like Guandan

4. **Coordination** — Per Nature Human Behaviour (2025): "LLMs perform well in self-interested games but struggle in games that require coordination"

5. **Probabilistic calculation** — Systematic errors in odds calculation, pot odds, expected value

6. **Data contamination** — Models may have memorized chess openings but can't apply strategic principles to novel games (hence TCG-Bench's approach)

7. **Vision-based play** — BALROG found models perform *worse* when given visual input vs text descriptions

---

## 5. LLMs vs Specialized Game AI

| Dimension | Specialized AI (DeepStack, AlphaZero) | LLMs |
|-----------|---------------------------------------|------|
| Poker superhuman play | Yes (since 2017) | No — ~53% on PokerBench |
| Generalization across games | Game-specific | Can play many games with prompting |
| Strategy quality | Near-equilibrium | Far from equilibrium |
| ToM/Deception | Engineered | Emergent (with prompting) |
| No training required | No — extensive RL needed | Yes — zero-shot possible |
| Adaptability | Fixed strategy | Can adapt via prompting/context |

The tradeoff: specialized AI plays perfectly but is game-specific; LLMs play poorly but generalize and require no training.

---

## 6. Frameworks & Tools

| Framework | Description | Link |
|-----------|-------------|------|
| **Game Reasoning Arena** | OpenSpiel-based, Gym-style API, LLM vs LLM/RL/random | [GitHub](https://github.com/SLAMPAI/game_reasoning_arena) |
| **PokerBench** | 11K poker scenarios, full evaluation pipeline | [GitHub](https://github.com/pokerllm/pokerbench) |
| **LLM-Hanabi** | Hanabi benchmark for ToM evaluation | [GitHub](https://github.com/HKUST-KnowComp/LLM-Hanabi) |
| **Suspicion-Agent** | ToM-aware GPT-4 for imperfect info games | [GitHub](https://github.com/CR-Gjx/Suspicion-Agent) |
| **LLM4CardGame** | 8 card games, SFT pipeline | [GitHub](https://github.com/THUDM/LLM4CardGame) |
| **TCG-Bench** | Contamination-free TCG benchmark | [Site](https://tcg-bench.com/) |
| **OpenSpiel** | Google's game framework (underlying many above) | [GitHub](https://github.com/deepmind/open_spiel) |

---

## 7. Emerging Trends (2025-2026)

1. **Agentic tool use** — Allowing LLMs to use solvers/calculators during poker (the "How Far Are LLMs from Pro Poker" paper)
2. **Reasoning model advantage** — Models like o1/o3 significantly outperform standard LLMs on ToM tasks in Hanabi
3. **Memory-equipped agents** — Persistent memory enables genuine strategic bluffing (not just random aggression)
4. **Contamination resistance** — Novel games like TCG-Bench to test genuine reasoning vs memorization
5. **Multi-agent self-play** — LLMs competing against each other as evaluation (CATArena, Husky Hold'em Bench)
6. **Behavioral game theory** — Moving beyond Nash Equilibrium to evaluate *how* models reason, not just outcomes

---

## 8. Key Takeaway

Card games expose a fundamental gap: LLMs can *talk about* strategy impressively but struggle to *execute* it reliably. Even GPT-4 manages only ~54% on poker decisions. The most promising directions are ToM-aware prompting (Suspicion-Agent), supervised fine-tuning on expert data (PokerBench), and leveraging reasoning models for deeper strategic inference. The field is moving fast — 2025-2026 has seen an explosion of benchmarks tackling different facets of this problem.
