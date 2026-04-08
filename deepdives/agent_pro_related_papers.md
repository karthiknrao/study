# Papers Related to Agent-Pro: Learning to Evolve via Policy-Level Reflection and Optimization

**Agent-Pro** (ACL 2024) — arXiv: [2402.17574](https://arxiv.org/abs/2402.17574)
LLM-based agent with policy-level reflection and optimization that learns from interactive experiences and progressively elevates its behavioral policy. Evaluated on Blackjack and Texas Hold'em.

---

## Direct Successors & Extensions

1. **PolicyEvol-Agent: Evolving Policy via Environment Perception and Self-Awareness with Theory of Mind** (Apr 2025)
   - arXiv: [2504.15313](https://arxiv.org/abs/2504.15313)
   - Directly extends Agent-Pro's policy evolution idea with Theory of Mind. LLM agent evolves biased policies in incomplete-information games (Leduc Hold'em) via psychological state perception and reflective expertise patterns.

2. **Readable Minds: Emergent Theory-of-Mind-Like Behavior in LLM Poker Agents** (Apr 2026)
   - arXiv: [2604.04157](https://arxiv.org/abs/2604.04157)
   - Investigates ToM-like behavior emerging in LLM poker agents through memory systems — showing ToM requires cognitive infrastructure, not just prompting. Very aligned with Agent-Pro's belief-based approach.

---

## Policy-Level Reflection & Self-Improvement

3. **Reflexion: Language Agents with Verbal Reinforcement Learning** (NeurIPS 2023)
   - arXiv: [2303.11366](https://arxiv.org/abs/2303.11366)
   - The foundational predecessor. Agents improve via verbal self-reflection stored in episodic memory, without weight updates. Agent-Pro explicitly extends this from action-level to policy-level reflection.

4. **ExpeL: LLM Agents Are Experiential Learners** (AAAI 2024)
   - arXiv: [2308.10144](https://arxiv.org/abs/2308.10144)
   - Agents autonomously gather experiences and extract reusable insights from success/failure trajectories. At inference, recalls insights to inform decisions. Closely parallels Agent-Pro's experience-to-policy pipeline.

5. **Agent-R: Training Language Model Agents to Reflect via Iterative Self-Training** (Jan 2025)
   - arXiv: [2501.11425](https://arxiv.org/abs/2501.11425)
   - Uses MCTS to construct training data that recovers correct trajectories from erroneous ones. Iterative self-training framework for on-the-fly reflection in interactive environments.

6. **Gödel Agent: A Self-Referential Agent Framework for Recursive Self-Improvement** (ACL 2025)
   - arXiv: [2410.04444](https://arxiv.org/abs/2410.04444)
   - Takes self-improvement further: agents can recursively rewrite their own code to improve. Inspired by Gödel machines. Goes beyond policy-level reflection to full self-modification.

7. **Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents** (May 2025)
   - arXiv: [2505.22954](https://arxiv.org/abs/2505.22954)
   - Sakana AI's extension of self-referential improvement with open-ended evolution. Agents build on prior innovations and recursively self-improve.

---

## LLM Card Game / Poker Playing

8. **PokerBench: Training LLMs to Become Professional Poker Players** (Jan 2025)
   - arXiv: [2501.08328](https://arxiv.org/abs/2501.08328)
   - Comprehensive benchmark for evaluating LLM poker ability. Identifies limitations of supervised fine-tuning for game-playing, relevant to Agent-Pro's card game evaluation.

9. **How Far Are LLMs from Professional Poker Players? Revisiting Game-Theoretic Reasoning** (Feb 2026)
   - arXiv: [2602.00528](https://arxiv.org/abs/2602.00528)
   - Systematic study identifying three core LLM flaws in poker: heuristic bias, factual misunderstandings, and "knowing-doing" gaps. Proposes ToolPoker with external solvers.

10. **Can Large Language Models Master Complex Card Games?** (Sep 2025)
    - arXiv: [2509.01328](https://arxiv.org/abs/2509.01328)
    - Evaluates LLM decision-making across card games including Texas Hold'em and Blackjack — directly comparable to Agent-Pro's evaluation domains.

11. **Learning to Play Blackjack: A Curriculum Learning Perspective** (Mar 2026)
    - arXiv: [2604.00076](https://arxiv.org/abs/2604.00076)
    - Uses LLMs to design curriculum learning for Blackjack, where the LLM creates multi-stage training paths for RL agents.

---

## Surveys & Broader Context

12. **A Survey on Large Language Model-Based Game Agents** (Apr 2024)
    - arXiv: [2404.02039](https://arxiv.org/abs/2404.02039)
    - Comprehensive overview of LLM game agents, covering learning, reasoning, and adaptation methods.

13. **A Survey on the Optimization of Large Language Model-based Agents** (Mar 2025)
    - arXiv: [2503.12434](https://arxiv.org/abs/2503.12434)
    - Surveys optimization approaches for LLM agents including reflection, memory, and policy evolution methods.

14. **Self-Improving LLM Agents at Test-Time** (Oct 2025)
    - arXiv: [2510.07841](https://arxiv.org/abs/2510.07841)
    - Addresses how LLM agents can improve without parameter updates during test-time interaction.
