OK, got a really rich picture. Let me lay it out for you — first a quick recap of what ReBeL is, then the two questions.

## Quick context on ReBeL

It's a 2020 paper by Noam Brown et al. (Facebook AI). The core trick: redefine a game "state" as a **public belief state** — the common-knowledge info + the probability distribution over private info. Then do self-play RL + search on that augmented state space. Provably converges to Nash in 2-player zero-sum games. Beats humans at HUNL poker with way less domain knowledge than Libratus/Pluribus.

The bigger idea: this is the paper that generalized AlphaZero's "search + RL" recipe to **imperfect information** settings.

---

## Q1: Latest similar papers

A few clear threads going on since 2020:

**Direct ReBeL extensions / speedups**
- **TurboReBeL** (ICLR 2026 submission, Sept 2025) — Li & Huang. 250× faster belief learning, matches ReBeL on Turn Endgame Hold'em with 0.4% of the training cost. Probably the most direct successor.
- **RL-CFR** (ICML 2024, arXiv:2403.04344) — uses RL to *learn* the action abstraction dynamically. Beats ReBeL's replication by 64 mbb/hand.
- **Efficient Online Pruning and Abstraction for IIEFGs** (ICLR 2025) — Li & Huang again.
- **SePoT / Multi-Agent Search with Policy Transformations** (IJCAI 2024, arXiv:2312.15220) — adds test-time search on top of any policy-gradient method, no search during training. Tested on Leduc hold'em, Goofspiel, Battleships.

**Bigger scope: new domains**
- **Obscuro — Fog of War Chess** (ICLR 2026 submission) — superhuman in the most popular imperfect-info chess variant. General subgame solving without common knowledge.
- **DouDiZhu / GuanDan surveys** (IEEE Trans. Games 2024–2025) — pushing ReBeL-style methods to trickier multi-player Chinese card games. Still not superhuman.
- **Diplomacy → CICERO** (Science 2022, Meta) — by the same Noam Brown team. This is the bridge to your second question, see below.

**Theory / convergence**
- **Reevaluating Policy Gradient Methods for IIGs** (ICLR 2026 submission, MIT/CFAR) — bombshell result: 7000 runs showing generic PG beats NFSP/PSRO/ESCHER/R-NaD. Worth knowing because it pushes back on the ReBeL-is-always-best narrative.
- **Policy Gradient Methods Converge Globally in IIGs** (NeurIPS 2025) — first global last-iterate convergence for PG in zero-sum EFGs.
- **A Policy-Gradient Approach with Best-Iterate Convergence** (Liu/Farina/Ozdaglar, arXiv 2024/2025).

So the field is fragmenting into three lanes: (a) scale ReBeL up, (b) replace CFR+search with policy gradients that converge nicely, (c) drag it to new games (chess variants, Chinese card games, Diplomacy). The hottest recent work is **TurboReBeL** and that ICLR 2026 policy-gradient reevaluation.

---

## Q2: Using this with LLMs — yes, big time

This is literally the most active research question in the ReBeL lineage. Noam Brown himself went to OpenAI and the **o1 / o3** story is the most direct answer. Three flavors worth distinguishing:

### Flavor 1: o1 / o3 style — "search at inference time, learned via RL"

This is the closest spiritual cousin to ReBeL applied to LLMs. ReBeL's idea was: learn a value function over **belief states**, then do search on top. OpenAI's o1 is essentially: learn a *chain of thought* via RL, where the CoT is the "search" happening inside the model's output. Performance scales with both **train-time RL compute** and **test-time thinking compute** — a new scaling axis, exactly what ReBeL pioneered for games.

The connection is very explicit: Noam Brown has said in interviews that pre-o1, the dominant inference-compute method was MCTS (worked great for Go, but failed for poker and language), and o1 is a learned-search approach that works for language. Same paradigm, new domain.

Relevant: OpenAI's "Learning to Reason with LLMs" post, Noam Brown's Simons talk, DeepSeek-R1 (replicated the recipe open-source with GRPO + RLVR).

### Flavor 2: Self-play for LLM alignment (the literal ReBeL→LLM mapping)

These are exactly the math from ReBeL — two-player zero-sum game, find the Nash equilibrium, no Bradley-Terry assumption needed:

- **SPPO** (Self-Play Preference Optimization, ICLR 2025) — Wu et al. Two-player constant-sum game, exponential weight updates, provably approximates Nash. SOTA on AlpacaEval 2.0 against GPT-4-Turbo with just 60k prompts.
- **MPO** (Magnetic Preference Optimization, ICLR 2025) — last-iterate convergence to the *original* (unregularized) Nash, not the regularized version everyone else converges to.
- **RSPO** (Regularized Self-Play Policy Optimization, ICLR 2025) — unified framework for all the regularizer choices.
- **COMAL** — convergent meta-algorithm; only one that actually converges to Nash on a 3×3 synthetic game in their experiments.
- **TANPO / SADPO** (ICLR 2025) — two-agent Nash Policy Optimization with provable sample efficiency.
- **Direct Nash Optimization (DNO)**, **REBEL**, **DRO** — siblings in this family.

The thesis: standard RLHF assumes Bradley-Terry (transitive preferences) which is wrong for human prefs (Condorcet paradox etc.). Modeling alignment as a **two-player constant-sum game** and finding Nash fixes this. SPPO is probably the most pragmatic starting point if you want to actually run this.

### Flavor 3: Multi-agent debate / self-play for reasoning (MARS, MARSHAL, SPIRAL)

These use self-play across *strategic games* to train LLMs, then the skills transfer to non-game reasoning tasks:
- **MARS** (arXiv:2510.15414, Oct 2025) — Qwen3-4B trained via self-play on cooperative+competitive games. Gains on AIME +10%, GPQA-Diamond +12.5% when plugged into MAD/AutoGen systems.
- **MARSHAL** / **SPIRAL** (2025) — similar self-play-in-zero-sum-games-for-reasoning idea.
- **ECON** — replaces round-by-round multi-agent debate with Bayesian Nash Equilibrium, ~11% better than MAD with much less token overhead.
- **LLM-Nash** (arXiv:2507.08208) — formal framework where LLMs *are* the strategy space (their reasoning prompts are the moves).
- **K-Level Reasoning** — hierarchical ToM-style recursive reasoning for LLMs.

---

## So, what's the actual answer to "can I use this with LLMs"?

Yes, and there are three doors depending on what you mean:

| Your goal | Closest analog | Where to start |
|---|---|---|
| Make an LLM reason better by giving it more "thinking" | ReBeL's value-net + search | **o1/R1 paradigm** — RL with verifiable rewards (RLVR), GRPO |
| Align an LLM without BT assumption | ReBeL's Nash-equilibrium framing | **SPPO** or **MPO** — drop-in for DPO |
| Multi-agent system that converges instead of looping | ReBeL's equilibrium search | **ECON** or **MARS** |

My recommendation if you want to play with this yourself: start with **SPPO** as a baseline (small, clean, well-implemented), and if you want to push state-of-the-art, look at **MPO** (better convergence theory) or **MARS** (if you care about general reasoning transfer).

---

Where do you want to go from here? A few directions I can dig into:
1. **Deep dive on one paper** — I can pull and summarize any of these in detail
2. **Implementation guide** — code-level walkthrough of SPPO or MARS
3. **Survey view** — write up a proper comparison table of these methods
4. **Connected to your specific use case** — if you tell me what you're actually trying to do (alignment? reasoning? multi-agent? a specific domain?), I can point to the most relevant subset

What sounds useful?