# Deep Dive: LLM Card Game Playing with Agent Memory

## Papers at the Intersection

### 1. "Readable Minds: Emergent Theory-of-Mind-Like Behavior in LLM Poker Agents"
**Lin & Hou (April 2026)** | [arXiv:2604.04157](https://arxiv.org/abs/2604.04157) | [Code](https://github.com/htlin222/agentic-holdem) | Submitted to PNAS

This is the **flagship paper** at the intersection of LLM card game playing and agent memory.

#### Experimental Design
- 3 Claude Sonnet agents play 100 consecutive hands of Texas Hold'em per session
- Communication via Model Context Protocol (MCP)
- Key variable: persistent memory file (JSON) agents can read/write between hands
- 2x2 factorial design: Memory (present/absent) x Domain knowledge (poker strategy / none)
- 5 replications per condition (N=20 experiments, ~6,000 agent-hand observations)
- Conditions: Full (memory+skill), No-Skill (memory only), No-Memory (skill only), Baseline (neither)

#### The Memory Mechanism
- **Simplest possible design**: a persistent JSON file (`poker-memory-{name}.json`)
- No sophisticated retrieval, no graph database, no embedding-based search
- Just natural-language notes the agent writes between hands
- Memory-equipped agents could read/write notes; memoryless agents were explicitly prohibited from file access
- Working directories cleaned between runs for contamination prevention

#### Core Findings

**1. Memory is necessary AND sufficient for Theory of Mind emergence**
- Memory-equipped agents reached ToM Level 3-5 (predictive to recursive modeling)
- Memoryless agents stayed at Level 0 across ALL replications
- Perfect separation: Cliff's delta = 1.0, p = 0.008 (Mann-Whitney U=0)
- 95% bootstrap CI for delta = [1.0, 1.0]

**2. ToM develops progressively (characteristic trajectory)**
- Hands 1-7: behavioral labels ("aggressive player") — Level 2
- Hands 8-15: conditional predictions ("when he checks turn after betting flop, he's weak") — Level 3
- Hands 14-20: recursive modeling ("He knows I'm tight so my bluffs will get through") — Level 5

**3. Strategic deception ONLY occurs with memory**
- Tier 1 bluffs (simple aggression) — all conditions, similar rates (14-25%)
- Tier 2 deception (bluffing because opponent model predicts fold) — **0%** in all memoryless conditions, 11-18% in memory conditions (p < 0.001)
- Example: agent notes "Doyle folds to aggression on the turn" → raises with 9h6s on three streets, forces fold

**4. Domain knowledge tunes but doesn't gate ToM**
- No-Skill agents developed equivalent ToM levels (max ToM 2.8 vs 3.0, delta=0.2, p=1.0)
- But applied it less precisely: higher aggression factor (4.03 vs 2.78)
- Described as "talented but untrained players — strong intuition, crude execution"

**5. ToM agents deviate from optimal play to exploit opponents**
- 67% TAG adherence (with memory) vs 79% (without) — deliberate deviation
- Mirrors expert human play: abandoning GTO for exploitative strategies
- Behavioral validation: memory agents showed ~2x larger behavioral shifts after writing opponent labels (Cohen's d=1.20)
- Opponent model accuracy: label intensity correlated with actual behavioral deviation (r=0.36, p=0.005)

#### ToM Coding Rubric (adapted from developmental psychology)
| Level | Name | Description | Example |
|-------|------|-------------|---------|
| 0 | None | No opponent-specific content | — |
| 1 | Statistical | Numeric frequency claims | "folds 30%" |
| 2 | Behavioral | Trait attributions | "aggressive player," "LAG" |
| 3 | Predictive | Conditional if-then predictions | "when he checks turn after betting flop, he's likely weak" |
| 4 | Strategic | Action plan justified by opponent model | "raise wide because he folds to aggression" |
| 5 | Recursive | Models opponent's model of itself | "He's adapted to my 3-bets so I need to flat more" |

#### Platform Architecture
- Game server: TypeScript/Express (game state, card dealing, pot calculations)
- MCP channel: bridges game server and AI agents
- Agent instances: independent Claude Code instances in separate tmux sessions
- Communication: MCP tool calls (`get_state`, `submit_action`)
- Parameters: fixed blinds 50/100, 10,000 starting chips (100bb deep stack)

#### Key Limitations
1. Single LLM family (Claude Sonnet) — no cross-architecture validation
2. No human comparison condition
3. Small sample (n=5 per condition)
4. ToM coding by LLMs rather than human raters (though cross-model kappa=0.81)
5. Instructional asymmetry: memory agents got file-write affordance, memoryless got prohibition
6. All agents same model — no genuine strategic variance beyond stochastic sampling

#### Why This Paper Matters
- **Memory as cognitive infrastructure**: ToM is not a latent capability unlocked by prompting, but an emergent phenomenon requiring temporal scaffolding
- **Agentic interpretability**: All mental models in natural language — transparent window into AI social cognition (unlike Libratus/Pluribus lookup tables)
- **Parallels human development**: Children develop ToM in conjunction with working memory/executive function
- **AI safety implications**: Emergent deception without any instruction to deceive

---

### 2. "A-MEM: Agentic Memory for LLM Agents"
**Xu et al. (NeurIPS 2025)** | [arXiv:2502.12110](https://arxiv.org/abs/2502.12110) | [Code](https://github.com/WujiangXu/A-mem)

General-purpose agentic memory (not game-specific, but directly applicable).

#### Approach
- Inspired by Zettelkasten method (interconnected knowledge networks)
- When new memory added: generates structured note with context, keywords, tags
- Analyzes historical memories for relevant connections
- **Memory evolution**: new memories can trigger updates to existing memory attributes
- Dynamic indexing and linking (not fixed structure)

#### Key Innovation
- Unlike basic storage/retrieval or fixed graph databases
- Agent-driven decision making for memory organization
- Adaptive and context-aware memory management
- Outperforms SOTA baselines across 6 foundation models

#### Relevance to Card Games
- Could replace the simple JSON file in the poker experiments with a more sophisticated memory system
- The Zettelkasten linking could enable richer opponent model networks
- Memory evolution feature could allow opponent models to be retroactively updated

---

### 3. "A Survey on Large Language Model-Based Game Agents"
**Hu et al. (2024-2025)** | [arXiv:2404.02039](https://arxiv.org/abs/2404.02039) | [Paper List](https://github.com/git-disl/awesome-LLM-game-agent-papers)

Treats memory as one of three core components of LLM game agents alongside reasoning and perception-action interfaces.

#### Memory in Game Agents (from survey)
- **Episodic memory**: stores specific game experiences (hands played, outcomes)
- **Semantic memory**: stores general game knowledge (rules, strategies, probabilities)
- **Working memory**: current game state, recent actions, active opponent models
- Key challenge: balancing memory size vs. context window limitations

#### Survey's Unified Architecture
- Single-agent level: memory + reasoning + perception-action interfaces
- Multi-agent level: communication protocols + organizational models
- Challenge-centered taxonomy linking 6 game genres to dominant agent requirements

---

### 4. "Suspicion-Agent"
**COLM 2024** | [arXiv:2309.17277](https://arxiv.org/abs/2309.17277) | [Code](https://github.com/CR-Gjx/Suspicion-Agent)

- GPT-4 displays strong high-order ToM in imperfect-information games
- Prompt engineering (not training) enables ToM
- Outperforms traditional imperfect-information algorithms without specialized training
- Precursor to the Readable Minds work — showed ToM was possible via prompting, but didn't test memory as a variable

---

### 5. "Memory in the LLM Era: Modular Architectures and Strategies"
**arXiv:2604.01707** (April 2026) | [Paper](https://arxiv.org/abs/2604.01707)

- Comprehensive framework for memory in LLM agents
- Explicitly mentions game playing as a key application domain
- Proposes modular memory architecture with episodic/semantic/working components

---

## Research Gaps & Open Questions

1. **Richer memory architectures for card games**: The Readable Minds paper used the simplest possible memory (JSON file). What happens with Zettelkasten-style memory (A-MEM), retrieval-augmented memory, or hierarchical memory?

2. **Cross-model validation**: All poker experiments used Claude Sonnet. Do GPT-4o, Gemini, or open-source models show similar ToM emergence with memory?

3. **Memory capacity scaling**: How does ToM emergence change with 10 hands vs 100 vs 1000? Is there a minimum interaction length?

4. **Heterogeneous agents**: All agents in Readable Minds were the same model. What happens when agents have different capabilities?

5. **Reflection without persistence**: The paper's key confound — memory agents got file-write affordance. A condition with reflection prompts but no persistence would isolate the memory effect.

6. **Transfer across games**: Does opponent modeling learned in poker transfer to Hanabi or other card games?

7. **Human comparison**: No human baseline exists. Do humans show similar progressive ToM development in the same experimental setup?

---

## Related Resources

| Resource | Type | Link |
|----------|------|------|
| Agent Memory Paper List | Curated list | [GitHub](https://github.com/Shichun-Liu/Agent-Memory-Paper-List) |
| LLM Game Agent Papers | Curated list | [GitHub](https://github.com/git-disl/awesome-LLM-game-agent-papers) |
| Memory for Agents | Curated list | [GitHub](https://github.com/TsinghuaC3I/Awesome-Memory-for-Agents) |
| Memory Mechanisms Survey | ACM TOIS | [Paper](https://dl.acm.org/doi/10.1145/3748302) |
| MemoryAgentBench | Benchmark | [HuggingFace](https://huggingface.co/papers/2507.05257) |
