# PolicyEvol Agent - Code Flow Documentation

## Overview

This is a **Policy Evolution (PE) agent system** for playing **Leduc Hold'em poker** (a simplified 2-player poker game) using LLMs. The core idea: an LLM-based agent plays against a baseline opponent (CFR, NFSP, DQN, DMC, or another LLM agent), and after each game, it reflects on its play, analyzes opponent patterns, and evolves its strategy over time.

---

## File Breakdown

| File | Role |
|------|------|
| `main.py` | Entry point. Game loop, training/validation orchestration, policy tracking |
| `agent.py` | **PEAgent** -- the main LLM-powered agent with belief, planning, pattern analysis |
| `suspicion_agent.py` | **SuspicionAgent** -- an alternative LLM agent (used as opponent when `--rule_model suspicion-agent`) |
| `agent_model.py` | Agent initialization (`agi_init`). Loads LLM based on settings |
| `setting.py` | Pydantic config classes for different LLM providers (OpenAI, DeepSeek, Qwen, Llama) |
| `util.py` | JSON loading, logging utilities, LLM loading |
| `context.py` | Console/web context wrapper for printing and user prompts |
| `game_config/leduc_limit.json` | Game rules and observation format description |
| `person_config/*.json` | Agent personality configs (name, age, personality, memories) |

---

## Main Code Flow

### 1. Entry Point: `main.py::run()`

```
run(args)
|-- Load settings & LLM config
|-- Load agent configs (player1, player2)
|-- Load game config (rules, observation format)
|-- agi_init() --> create PEAgents
|-- get_rule_model() --> load baseline opponent
\-- Training loop: for game_idx in range(100):
    \-- play_game() --> returns updated policy + score
```

**Key args:**
- `--user_index=1` -> player at index 1 is the baseline opponent
- `--rule_model` -> opponent type: `cfr`, `nfsp`, `dqn`, `dmc`, `suspicion-agent`
- `--mode` -> `normal`, `first_tom`, `second_tom`, `automatic`
- `--train` -> runs 100 training games

---

### 2. Game Loop: `main.py::play_game()`

This runs **one full poker game** (multiple betting rounds until showdown/fold).

```python
while not env.is_over():
    idx = env.get_player_id()  # whose turn?

    if idx == args.user_index:  # BASELINE OPPONENT'S TURN
        if rule_model == "suspicion-agent":
            act, comm = suspicion_agent.make_act(...)
        else:
            act = rule_model.eval_step(state)  # CFR/NFSP/DQN/DMC
    else:  # LLM AGENT'S TURN
        act, comm, bot_short_memory, bot_long_memory, new_policy =
            pe_agent.make_act(...)

    env.step(act)
```

**Memory structures maintained per game:**
- `bot_short_memory[player_idx]` -- round-by-round history for each player
- `bot_long_memory[player_idx]` -- game history (can exclude opponent observations via `--no_hindsight_obs`)
- `oppo_bot_short_memory` -- opponent's perspective of history

After the game ends:
- Computes payoff / winner
- Builds `long_memory` (interleaved turn-by-turn history + win message)
- **`get_summarization()`** -- LLM reflects on the game and generates a memory summary
- **`add_long_memory()`** -- saves summary to agent's memory

---

### 3. Agent Decision Pipeline: `agent.py::PEAgent.make_act()`

This is the **core cognitive loop**. Every time it's the LLM agent's turn:

```
make_act(observation, opponent_name, ...)
|
|-- 1. convert_obs()          --> LLM converts raw dict obs to readable text
|-- 2. get_short_memory_summary() --> LLM summarizes game history so far
|-- 3. get_opponent_pattern()     --> LLM analyzes opponent's behavior pattern
|-- 4. get_self_pattern()         --> LLM reflects on own strategy
|-- 5. get_opponent_belief()      --> LLM infers opponent's cards & goals
|-- 6. get_self_belief()          --> LLM forms belief about own situation
|-- 7. planning_module()          --> LLM makes plans, estimates win rates
|-- 8. action_decision()          --> LLM picks action + says something
|
\-- Returns: action, comment, updated memories, new_policy
```

**The "policy"** is a string concatenation: `opponent_pattern ||| self_pattern`. It evolves game-by-game.

---

### 4. Theory of Mind (ToM) Modes

The `--mode` flag controls how deeply the agent reasons about the opponent:

| Mode | Description |
|------|-------------|
| `normal` | No belief modeling. Just plans based on own cards + history |
| `first_tom` | Agent reasons about opponent's cards and actions |
| `second_tom` | Agent reasons about what opponent thinks agent has |
| `automatic` | Runs both 1st and 2nd ToM, then LLM-evaluates which is better |

Each mode uses **different prompt templates** in `planning_module()`, `get_self_belief()`, `get_opponent_belief()`, `get_self_pattern()`, `get_opponent_pattern()`.

---

### 5. Policy Evolution: Pattern & Reflection Methods

After each game, the agent's memory grows. Patterns are derived from recent memories (`self.memory[-last_k:]`):

- **`get_opponent_pattern(old_pattern, ...)`**
  - Inputs: last 20 game memories + old opponent pattern
  - LLM infers: opponent's tendencies per card, opponent's guess of agent's pattern, opponent's character (radical/conservative/neutral/flexible)

- **`get_self_pattern(old_self_pattern, opponent_pattern, ...)`**
  - Inputs: last 20 memories + old self-pattern + current opponent pattern
  - LLM does: reflection on right/wrong actions, strategy improvement

- **`get_summarization(...)`** (called after game ends in `main.py`)
  - Summarizes full game history
  - Infers opponent's cards with probability
  - Reflects on winning/losing strategy

---

### 6. Prompt Engineering Structure

All reasoning is done via **LangChain `LLMChain`** with large handcrafted prompts. Example prompt structure in `planning_module()` (2nd ToM):

```
You are the objective player behind NPC {name}, playing {game_name} vs {opponent}.
Game rule: {rule}
Your observation: {observation}
Recent history: {short_memory_summary}
Your pattern: {pattern}
Your belief: {belief}

Tasks:
1. Make Reasonable Plans (for each valid action)
2. Estimate opponent's actions & Win/Lose/Draw rates per plan
3. Calculate Expected Chips Gain per plan
4. Select best plan
```

---

### 7. Configuration & LLM Loading

**`setting.py`** defines preset configs:
- `OpenAIGPT4Settings`, `OpenAIGPT3_5TurboSettings`, `DeepSeekSettings`, `OpenAIQwenSettings`, `OpenAILlamaSettings`

**`agent_model.py::load_llm_from_config()`** instantiates the LLM:
- `chatopenai` -> `ChatOpenAI`
- `Qwen` -> `Tongyi` (DashScope)
- `DeepSeek` -> `init_chat_model("deepseek-chat", api_base="https://api.deepseek.com/")`
- `Llama` -> `ChatOpenAI` (with custom base URL)

---

### 8. Logging

`util.py::get_logging()` writes JSON-structured logs to `./log/{name}.json`. Logs include:
- Observations (raw + readable)
- Short memory summaries
- Beliefs, plans, patterns
- Actions and "talk sentences"
- Long memory and memory summaries per game

---

## Data Flow Diagram (Simplified)

```
main.py
  |
  |---> agi_init() ---> PEAgent (name, personality, llm, memory=[])
  |
  |---> get_rule_model() ---> CFR/NFSP/DQN/DMC/SuspicionAgent
  |
  \---> play_game() x100
          |
          |---> env.get_state() ---> raw_obs dict
          |
          |---> PEAgent.make_act(raw_obs)
          |       |
          |       |---> convert_obs() ---> readable text
          |       |---> get_short_memory_summary() ---> game state summary
          |       |---> get_opponent_pattern() ---> updated opponent model
          |       |---> get_self_pattern() ---> updated self strategy
          |       |---> get_opponent_belief() ---> what do they have?
          |       |---> get_self_belief() ---> what's my situation?
          |       |---> planning_module() ---> plans + expected values
          |       \---> action_decision() ---> action + comment
          |
          |---> env.step(action)
          |
          \---> (after game) get_summarization() ---> add_long_memory()
```

---

## Key Design Patterns

1. **All state is textual** -- The agent has no explicit neural policy network. Its "policy" is a natural language string describing patterns and strategies, passed via prompts to the LLM.

2. **Memory as narrative** -- Both short-term (current game) and long-term (past games) memories are stored as natural language sentences, not vectors or tensors.

3. **Explicit belief modeling** -- The agent maintains `self_belief` and `opponent_belief` strings, updated every turn.

4. **Prompt-level ToM** -- Theory of mind isn't a learned model; it's a prompt template that asks the LLM to reason from the opponent's perspective.

5. **Sleep delays** -- `time.sleep()` calls (1-2s) between LLM calls, likely to avoid rate limiting.
