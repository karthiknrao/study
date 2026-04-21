# Deep Dive: Using LLMs for Negotiation and Bargaining

## 1. Academic Research Landscape

### Key Papers

| Paper | Venue | Contribution |
|---|---|---|
| **NegotiationArena** (Bianchi et al.) | ICML 2024 | Foundational benchmark; found LLMs can boost outcomes by 20% using behavioral tactics |
| **LLM-Deliberation** (Abdelnabi et al.) | NeurIPS 2024 | Multi-agent, multi-issue negotiation games; cooperation/competition/maliciousness |
| **AgenticPay** | 2026 | Multi-agent buyer-seller framework with Gymnasium-like API |
| **PACT** | 2025 | 20-round bargaining benchmark with Glicko-2 skill ratings |
| **ASTRA** (Kwon et al.) | EMNLP 2025 | Tool-integrated action for adaptive strategic reasoning |
| **Game-theoretic LLM** | 2024 | Workflow-based approach improving rationality in strategic decisions |
| **RLVR for Negotiation** | 2026 | Teaching negotiation via reinforcement learning with verifiable rewards |
| **Richelieu** | NeurIPS 2024 | Self-evolving agents for AI Diplomacy |

### Benchmarks
- **NegotiationArena** - ultimatum games, trading games, price negotiations
- **LLM-Deliberation** - multi-agent, multi-issue scorable games
- **PACT** - multi-round buyer-seller bargaining
- **AgenticPay** - buyer-seller transaction simulation
- **NegMAS** - Python library for autonomous negotiation agents

### Evaluation Metrics
Success rate, individual/joint utility, Pareto efficiency, fairness, Glicko-2 ratings, deal conversion rate, concession patterns, rationality measures (deviation from game-theoretic optimum)

---

## 2. Frameworks & Platforms

| Framework | Type | Key Feature |
|---|---|---|
| **NegotiationArena** | Research benchmark | Flexible multi-scenario evaluation |
| **AgenticPay** | Simulation framework | Gymnasium API, private constraints |
| **NegMAS** | Python library | ANAC competition-grade, RL integration |
| **LLM-Deliberation** | Benchmark | NeurIPS 2024 dataset track |
| **PACT** | Benchmark/leaderboard | Glicko-2 rankings, visualization suite |

**Multi-agent orchestration**: LangGraph (graph-based stateful agents), AutoGen (Microsoft, 50k+ stars), CrewAI (role-based)

---

## 3. Key Techniques & Approaches

### Prompt Engineering Strategies

- **Role-playing prompts** — personality assignment, contextual framing ("desperate seller," "skeptical buyer")
- **Theory-of-Mind prompting** — explicit opponent modeling ("What does the other party want?")
- **Strategic reasoning chains** — Chain-of-Thought for BATNA analysis, offer planning, concession scheduling
- **Few-shot learning** — example negotiation dialogues demonstrating effective tactics

### Core Negotiation Concepts Implemented via LLMs

**BATNA Awareness** — explicit walk-away point calculation, dynamic updating

**Anchoring** — initial offer positioning, reference point establishment, competitive anchoring

**Concession-Making** — gradual/reciprocal concessions, conditional trades ("I'll concede X if you concede Y")

**Issue Linking** — multi-issue coordination, logrolling (trading across issues), package deals vs. item-by-item

### Multi-Turn Dialogue Management
- Full conversation history tracking + key information extraction
- State tracking: current offers, remaining bargaining space, time pressure, trust levels
- Strategic turn-taking: when to offer vs. ask questions vs. stay silent

### Game Theory Integration
- Nash equilibrium reasoning hints fed to LLM
- Mechanism design constraints on outputs
- Coalition formation (Shapley values) for multi-party scenarios
- The **Game-theoretic LLM** paper showed workflow-based approaches significantly improve rationality

---

## 4. Limitations & Challenges

### Sycophancy & Agreeability Bias
LLMs tend to be **too cooperative** — difficulty maintaining tough positions, over-conceding, "yes-man" behavior. Institutional guardrails and explicit non-cooperative instructions help but don't fully resolve this.

### Strategic Reasoning Gaps
- **Limited forward planning** — myopic focus on immediate exchanges
- **Weak backward induction** — poor multi-turn strategy
- **Vulnerability to exploitation** — especially in incomplete information games
- Models converge on similar negotiation patterns, limited strategic diversity

### Context Window Limitations
Extended negotiations exceed context, causing loss of early history and consistency issues. Memory summarization helps but loses nuance.

### Evaluation Challenges
No universal framework; difficulty comparing across studies; trade-offs between fairness, efficiency, relationship value.

### Ethical Concerns
- **Deception** — papers show LLMs can learn strategic dishonesty
- **Price discrimination** — AI-driven segmentation by buyer profile
- **Manipulation** — emotional exploitation, vulnerability targeting
- **Transparency** — should AI agents disclose they're not human?

---

## 5. Real-World Applications

| Domain | Use Case | Maturity |
|---|---|---|
| **E-commerce** | Price negotiation bots, dynamic discounting | Active |
| **Customer Service** | Dispute resolution, refund negotiation, de-escalation | Active |
| **B2B Contracts** | Supplier negotiation, SLA terms, clause-by-clause analysis | Emerging |
| **Salary/Hiring** | Negotiation copilots, compensation optimization, bias mitigation | Emerging |
| **Diplomacy** | UN Security Council simulations, climate negotiation modeling | Research |
| **Supply Chain** | Multi-tier procurement, logistics contracts | Emerging |
| **Real Estate** | Property price/lease negotiation | Early |
| **Legal** | Settlement negotiation, debt restructuring | Early |

---

## 6. Recent Breakthroughs (2024-2026)

- **RLVR for Negotiation** (2026) — RL with rule-based graders produces objectively better strategic play
- **Self-evolving agents** (Richelieu) — agents that reflect, revise memory, and self-improve (21% → 34% performance jump)
- **ASTRA** (EMNLP 2025) — tool-integrated actions for real-time offer optimization, competitive with Claude/Gemini
- **Game-theoretic workflows** — near-optimal allocations when equilibrium hints are provided to LLMs
- **Emotion-aware negotiation** — EQ-Negotiator integrates emotional intelligence into credit dialogues
- **Concordia Contest** (NeurIPS 2024) — standardized multi-agent negotiation competition

---

## 7. Practical Implementation Patterns

### Agent Architecture

```
┌─────────────────────────────────┐
│        System Prompt Layer       │
│  Role, goals, BATNA, ZOPA,      │
│  ethical boundaries              │
├─────────────────────────────────┤
│        Memory Module             │
│  Conversation history,           │
│  commitments, opponent model     │
├─────────────────────────────────┤
│        Strategy Module           │
│  Opening offers, concession      │
│  schedule, multi-issue coord     │
├─────────────────────────────────┤
│        Reasoning Engine          │
│  Theory of Mind, CoT,           │
│  offer evaluation, risk assess   │
├─────────────────────────────────┤
│      Communication Interface     │
│  Tone, turn-taking, formatting   │
└─────────────────────────────────┘
```

### Key Prompt Pattern

```
Before each response, reason through:
1. What does the other party want? (Theory of Mind)
2. What is my best strategic move? (Game Theory)
3. What concessions can I afford? (BATNA-aware)
4. How does this advance my goals? (Forward planning)
5. What should I reveal vs. conceal? (Info management)
```

### Knowledge Representation

Negotiation domains are typically structured as:
- **Issues** with types (continuous/discrete), ranges, importance weights, and utility functions
- **Constraints** (budget limits, deadlines)
- **BATNA** with estimated value and success probability
- **Opponent models** with estimated preferences, confidence levels, and behavioral patterns

---

## Key Sources

- [NegotiationArena](https://arxiv.org/abs/2402.05863) — Bianchi et al., ICML 2024
- [LLM-Deliberation](https://arxiv.org/abs/2309.17234) — Abdelnabi et al., NeurIPS 2024
- [AgenticPay](https://arxiv.org/abs/2602.06008) — Multi-Agent Negotiation System, 2026
- [PACT Benchmark](https://github.com/lechmazur/pact) — Multi-Round Buyer-Seller Bargaining
- [Game-theoretic LLM](https://arxiv.org/abs/2411.05990) — Agent Workflow for Negotiation Games
- [ASTRA](https://arxiv.org/abs/2503.07129) — Adaptive Strategic Reasoning Agent, EMNLP 2025
- [RLVR for Negotiation](https://arxiv.org/abs/2604.09855) — 2026
- [NegMAS](https://negmas.readthedocs.io/) — Negotiation Multi-Agent System Library
- [Richelieu](https://arxiv.org/abs/2409.15258) — Self-Evolving LLM Agents for Diplomacy, NeurIPS 2024

---

## Bottom Line

The field is maturing rapidly. LLMs are **surprisingly capable** negotiators when given proper strategic scaffolding, but they suffer from **agreeability bias, limited long-horizon planning, and ethical risks**. The most effective systems combine LLM reasoning with game-theoretic analysis and structured domain knowledge rather than relying on pure language model intuition.

The biggest open problems: overcoming sycophancy, improving multi-turn strategic reasoning, and establishing ethical norms for AI-driven negotiation.
