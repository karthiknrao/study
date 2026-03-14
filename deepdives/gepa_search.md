# GEPA-type Methods for Training LLM Agents in RL Settings

## What is GEPA?

**GEPA (Genetic-Pareto Reflective Prompt Evolution)** is a method from UC Berkeley/Stanford (ICLR 2026 Oral) that optimizes LLM agents through **reflective prompt evolution** instead of traditional RL. Key ideas:

1. **Language as Learning Signal**: Instead of sparse scalar rewards (success/fail), GEPA uses rich natural language feedback from execution traces (reasoning, tool calls, errors)
2. **Genetic Prompt Evolution**: Treats prompts as evolvable "genes" - mutates and combines them across generations
3. **Reflective Mutation**: After failures, the LLM reflects on what went wrong and proposes improved prompts
4. **Pareto-based Selection**: Maintains diverse top performers to avoid local optima

**Results**: Up to 35x fewer rollouts than GRPO, outperforms RL by 6-20% on tasks like HotpotQA, AIME-2025

---

## Related Approaches for LLM Agents in Games

| Method | Description |
|--------|-------------|
| **GRPO** | Group Relative Policy Optimization - standard RL for LLMs (DeepSeek-R1) |
| **PPO/DAPO** | Proximal Policy Optimization for LLM fine-tuning |
| **ArCHer** | Hierarchical multi-turn RL for LLM agents |
| **LGRL** | LLM-Guided RL - decomposes goals into subgoals for RL agents |
| **AgentGym-RL** | Unified framework for multi-turn RL training across diverse environments |
| **StarPO** | State-Thinking-Actions-Reward Policy Optimization |

---

## Applying GEPA to Game-Playing Agents

For games specifically, GEPA can be adapted by:

1. **Multi-module prompts**: Each game agent component (reasoning, action selection, opponent modeling) gets its own prompt
2. **Execution traces**: Serialize game state, reasoning, actions taken, outcomes → feed back as natural language
3. **Reflective improvement**: LLM analyzes "why did we lose this match?" and updates prompts
4. **Evolutionary search**: Try different strategies via genetic mutation, keep Pareto-optimal ones

**Advantages over traditional RL**:
- Works with API-only models (no weight access needed)
- More sample efficient
- Interpretable - you can read what the agent "learned"
- Handles sparse rewards better via rich language feedback

---

## Papers & Code

### Core GEPA Paper & Code

| Resource | Link |
|---------|------|
| **GEPA Paper (ICLR 2026 Oral)** | https://arxiv.org/abs/2507.19457 |
| **Official Code** | https://github.com/gepa-ai/gepa |
| **DSPy Integration** | https://dspy.ai/tutorials/gepa_ai_program/ |

### Related Reflexive Methods (precursors to GEPA)

| Paper | Description | Code |
|-------|-------------|------|
| **Reflexion** (NeurIPS 2023) | Verbal reinforcement learning - agents reflect on failures in language | https://github.com/noahshinn/reflexion |
| **ReAct** (ICLR 2023) | Synergizes reasoning + acting in LLM agents | Various implementations |
| **ReCon** (ACL 2024) | Recursive contemplation for deception handling in Avalon game | https://github.com/Shenzhi-Wang/recon |

### Game-Specific Implementations

| Project | Description | Link |
|---------|-------------|------|
| **Agent-Pro** | LLM agents for poker games (Blackjack, Texas Hold'em) via RLCard | https://github.com/zwq2018/Agent-Pro |
| **Text Game LLM Improver** | Self-play in text-based games (evolves LLM through self-play) | https://github.com/emartin59/text-game-llm-improver |
| **Werewolf Game RL** | LLM agents with RL for Werewolf social deduction game | https://arxiv.org/abs/2310.18940 |

### Comprehensive Surveys/Collections

| Resource | Description |
|----------|-------------|
| **Awesome LLM Game Agent Papers** | https://github.com/git-disl/awesome-LLM-game-agent-papers (586 stars) |
| **Awesome Agent Papers** | https://github.com/luo-junyu/Awesome-Agent-Papers |

---

## Key Takeaway

GEPA itself hasn't been widely applied to games yet (published July 2025). The closest implementations are **Reflexion** and **ReAct** - which use the same core idea of reflective self-improvement. For game-playing agents, **Agent-Pro** and the **Werewolf RL** paper are the most relevant examples combining LLM + RL for strategic games.
