# LLM Agents for Wargaming: A Deep Dive

## The Two Anchor Papers

### 1. WarAgent: War and Peace (Nov 2023)

**Paper:** [arXiv:2311.17227](https://arxiv.org/abs/2311.17227) | **Code:** [github.com/agiresearch/WarAgent](https://github.com/agiresearch/WarAgent)
**Authors:** Hua, Fan, Li, Mei, Ji, Ge, Hemphill, Zhang (Rutgers / UMich)

**What it does:** LLM-powered multi-agent simulation of **international-scale conflicts** (WWI, WWII, Warring States China). Each agent represents a sovereign state.

**Architecture — four building blocks:**

| Component | Role |
|-----------|------|
| **Country Agents** | LLM agents with 6-dimension profiles (leadership, military, resources, history, policy, morale) and a guided 4-step reasoning chain |
| **Secretary Agents** | Verification/correction layer — up to 4 rounds of negotiation to ensure format compliance and logical consistency |
| **Board** | Shared data structure encoding international relationships as symbols: `x` (War), `&` (Alliance), `o` (Treaty), `~` (Peace) |
| **Stick** | Per-agent domestic state tracker: mobilization, internal stability, war readiness |

**Key design decisions:**
- **Board+Stick as state+memory compression** — avoids unbounded prompt growth across rounds. This is the core engineering insight that makes multi-round simulation tractable.
- **7-action space:** Wait, Mobilize, Declare War, Form Alliance, Non-Intervention Treaty, Peace Agreement, Send Message. Each with publicity (public/private) metadata.
- **Anonymization** — country names and historical details are deliberately altered to prevent LLM memorization from contaminating the simulation.
- **Simultaneous round-based** — all agents act in parallel each round.

**Results:**

| Metric | Accuracy |
|--------|----------|
| World War breakout | **100%** |
| Mobilization timing | **92.09%** |
| Alliance formation | **77.78%** |
| War declaration | **54.60%** |

**Critical finding — War Inevitability:** Even with null/empty triggering events, the system still produces cold-war tensions. Historical background (agent profiles) is the most influential factor, more than catalytic events. Structural conditions drive outcomes.

---

### 2. BattleAgent (Apr 2024)

**Paper:** [arXiv:2404.15532](https://arxiv.org/abs/2404.15532) | **Code:** [github.com/agiresearch/BattleAgent](https://github.com/agiresearch/BattleAgent) | **EMNLP 2024 Demo**
**Authors:** Lin, Hua, Li, Chang, Fan, Ji, Hua, Jin, Luo, Zhang (Rutgers / UMich / U Rochester)

**What it does:** Multi-modal (text + vision) simulation of **tactical-level historical battles** — medieval European battles (Crecy, Agincourt, Poitiers, Falkirk).

**Architecture — three pillars:**

1. **Enhanced 2D Realism** — Coordinate-based maps with terrain features (rivers, forests, villages). 15-minute time intervals. 10km sight range with line-of-sight. VLM processes visual map state.
2. **Hierarchical Multi-Agent System** — Commanding agents (army-level) + 30 individual soldier agents with rich personal profiles (name, age, family, personality, fears).
3. **Dynamic Agent Structure** — Agents can **fork** (split forces for scouting), **merge** (consolidate), or **prune** (remove when destroyed). This is their key architectural innovation.

**Action space:** 51 actions across 7 categories (Reposition, Preparation, Attack, Defense, Observation, Retreat, Dynamic Structure). Far more granular than WarAgent's 7 actions.

**Casualty evaluation:** An independent GPT-4 "observer" agent evaluates combat outcomes, factoring in force sizes, weapons, terrain, and distances.

**Results:**

| Model | Accuracy |
|-------|----------|
| GPT-4 / GPT-4V | Reasonably close to historical casualty figures; correct directionality in all cases |
| Claude-3-Opus | Consistently overestimated casualties (e.g., predicted ~8,000 vs historical ~2,000 at Falkirk) |

Soldier agents' generated narratives thematically matched their side's experience (English: "return", "duty"; French: "fear", "chaos").

---

## How WarAgent and BattleAgent Compare

| Dimension | WarAgent | BattleAgent |
|-----------|----------|-------------|
| **Scale** | International / grand strategy | Tactical / battlefield |
| **Time period** | Modern (WWI, WWII) + Ancient China | Medieval Europe |
| **Agent count** | 7–8 country agents | Commanders + 30 individual soldiers |
| **Modality** | Text-only | Text + Vision (VLM) |
| **Action space** | 7 diplomatic/military actions | 51 tactical actions |
| **State management** | Board (relationships) + Stick (domestic) | Coordinate-based map + agent profiles |
| **Key innovation** | Board/Stick state compression; Secretary agents for verification | Dynamic agent structure (fork/merge/prune); VLM terrain reasoning |
| **Research group** | Same lead (Yongfeng Zhang, Rutgers) | Same lead (Yongfeng Zhang, Rutgers) |

They are **complementary systems** from the same lab — WarAgent for the geopolitical/diplomatic layer, BattleAgent for the tactical/combat layer. Together they suggest a path toward a unified multi-scale wargaming simulation.

---

## The Broader Landscape

### Directly related wargaming systems

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| **"Human vs. Machine"** (Lamparth et al., [2403.03407](https://arxiv.org/abs/2403.03407)) | 2024 | Behavioral benchmark comparing expert human wargamers to LLMs — identifies systematic divergences |
| **"Escalation Risks from LMs"** (Rivera et al., [2401.03408](https://arxiv.org/abs/2401.03408)) | 2024 | **Critical finding:** LLMs exhibit pronounced escalation bias in military scenarios. FAccT 2024. |
| **"Snow Globe"** (Hogan & Brennen, [2404.11446](https://arxiv.org/abs/2404.11446)) | 2024 | Open-ended qualitative wargaming framework — challenges the quantitative paradigm |
| **"AI Arms and Influence"** (Kenneth Payne, [2602.14740](https://arxiv.org/abs/2602.14740)) | 2026 | Tests frontier models in **nuclear crisis** scenarios. Finds spontaneous deception, theory of mind, and that the nuclear taboo doesn't reliably prevent escalation |
| **"Red Lines and Grey Zones"** (Drinkall, [2510.03514](https://arxiv.org/abs/2510.03514)) | 2025 | Tests LLM compliance with International Humanitarian Law — documents significant failures |

### Diplomacy and negotiation

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| **Richelieu** (Guan et al., [2407.06813](https://arxiv.org/abs/2407.06813)) | 2024 | Self-evolving LLM agent for Diplomacy — learns negotiation strategies over time via experience reflection |
| **"Measuring Negotiation Tactics"** (Li et al., [2512.18292](https://arxiv.org/abs/2512.18292)) | 2025 | Fine-grained taxonomy of negotiation tactics (persuasion, deception, coalition-building) for benchmarking LLM vs human play |
| **Cicero** (Meta FAIR, 2022) | 2022 | The precursor — combined RL + NLP to play Diplomacy at human level. Not LLM-native, but established the viability of AI in strategic multi-player negotiation |

### Multi-agent social simulation

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| **"Artificial Leviathan"** (Dai et al., [2406.14373](https://arxiv.org/abs/2406.14373)) | 2024 | LLM agents spontaneously develop social contracts and governance — Hobbesian political philosophy via simulation |
| **"Dynamic Coalition Detection"** (Kulkarni et al., [2502.16339](https://arxiv.org/abs/2502.16339)) | 2025 | Real-time detection of coalition formation/shifts from agent dialogue |
| **"LLMs as Strategic Actors"** (Solopova et al., [2603.02128](https://arxiv.org/abs/2603.02128)) | 2026 | Rigorous evaluation framework for LLM strategic competence in adversarial game-theoretic settings |

---

## Key Themes and Open Problems

### 1. Escalation Bias
The Rivera et al. (FAccT 2024) finding is alarming: LLMs default to escalation. This is arguably the most important safety finding in the field. Kenneth Payne's 2026 nuclear crisis paper reinforces it — the nuclear taboo doesn't reliably constrain LLM behavior.

### 2. Scale Mismatch
WarAgent operates at the nation-state level; BattleAgent at the battlefield level. No system yet bridges these scales convincingly. A unified simulation that handles grand strategy through tactical execution remains an open problem.

### 3. Fidelity vs. tractability
BattleAgent's 51-action space produces richer behavior but is harder to evaluate. WarAgent's 7-action space is more tractable but misses tactical nuance. The Board/Stick state compression in WarAgent is an elegant solution to context window management, but it necessarily loses information.

### 4. Historical validity
Both systems anonymize scenarios to prevent memorization, but the effectiveness of this approach is not rigorously validated. The gap between Claude-3 and GPT-4 on BattleAgent (one overestimates casualties by 4x) shows model-dependent accuracy is a serious concern.

### 5. Missing capabilities both papers acknowledge
- No espionage / covert operations
- No asynchronous actions (everything is round-based)
- No communication time lags
- No bidirectional command influence (soldiers can't affect commander decisions)
- No systematic stopping criteria
- Casualty evaluation relies on LLM rather than a dedicated combat model

### 6. The evaluation problem
How do you evaluate a wargame simulation? Historical accuracy is one metric (WarAgent: 55–100% depending on metric), but counterfactual validity is arguably more important and much harder to measure. No consensus framework exists.

---

## Suggested Reading Order

1. **WarAgent** — start here for the multi-agent architecture patterns
2. **BattleAgent** — for the tactical layer and dynamic agent structures
3. **Rivera et al. "Escalation Risks"** (FAccT 2024) — essential safety context
4. **Kenneth Payne "AI Arms and Influence"** (2026) — cutting-edge nuclear crisis work
5. **Snow Globe** — methodological counterpoint (qualitative > quantitative)
6. **Richelieu** — for self-evolving agent strategies in Diplomacy

All papers are open-access on arXiv. Both WarAgent and BattleAgent have public GitHub repos with runnable code.

---

## Sources

- [WarAgent - arXiv:2311.17227](https://arxiv.org/abs/2311.17227)
- [BattleAgent - arXiv:2404.15532](https://arxiv.org/abs/2404.15532)
- [Escalation Risks - arXiv:2401.03408](https://arxiv.org/abs/2401.03408)
- [Snow Globe - arXiv:2404.11446](https://arxiv.org/abs/2404.11446)
- [AI Arms and Influence - arXiv:2602.14740](https://arxiv.org/abs/2602.14740)
- [Red Lines and Grey Zones - arXiv:2510.03514](https://arxiv.org/abs/2510.03514)
- [Human vs Machine - arXiv:2403.03407](https://arxiv.org/abs/2403.03407)
- [Richelieu - arXiv:2407.06813](https://arxiv.org/abs/2407.06813)
- [Artificial Leviathan - arXiv:2406.14373](https://arxiv.org/abs/2406.14373)
- [LLMs as Strategic Actors - arXiv:2603.02128](https://arxiv.org/abs/2603.02128)
- [Dynamic Coalition Detection - arXiv:2502.16339](https://arxiv.org/abs/2502.16339)
- [Measuring Negotiation Tactics - arXiv:2512.18292](https://arxiv.org/abs/2512.18292)
