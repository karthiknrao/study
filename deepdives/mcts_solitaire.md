# Using MCTS to Play Klondike / Solitaire

Yes — with caveats. MCTS is one of the strongest classical approaches to Klondike and has been applied to it directly. But naïve MCTS doesn't work; you need variants that handle the three things that make solitaire nasty: **partial observability**, **stochasticity**, and **long horizons with sparse reward**.

## The three obstacles

| Obstacle | Why naïve MCTS fails | Fix |
|---|---|---|
| **Hidden cards** (Klondike stock/waste) | MCTS assumes the state is fully known; can't branch over hidden info | **Information Set MCTS** (Cowling 2012) or **POMCP** (Silver & Veness 2010) — branch over a *belief distribution* or a *determinization* |
| **Stochasticity** (stock draws) | Each draw has multiple possible outcomes | Explicit **chance nodes** in the tree; MCTS samples one outcome per visit |
| **Sparse reward** (win/lose only at end) | Random rollouts from a Klondike state win ~0% of deals → no signal | **Informed rollouts** (heuristic-biased) or replace rollouts with a **learned value network** |

Get those three right and MCTS becomes competitive.

## The canonical MCTS-for-Klondike work

The reference point is **[Bjarnason, Tadepalli, Fern 2007/2009](https://eprints.whiterose.ac.uk/id/eprint/75050/1/EnsDetMagic.pdf)** — they applied a hybrid of **Hindsight Optimization (HOP) + UCT** to Klondike. Two key ideas:

1. **HOP**: simulate *many* random completions of the deal (sampling hidden cards consistently with what you can see), play each one out greedily, and pick the move that's best in expectation across samples.
2. **UCT tree search on top**: spend more samples on promising branches.

They also did an earlier version on **Thoughtful Solitaire** (the perfect-information variant) using Nested Monte-Carlo Search — this is the cleanest testbed, since you remove the partial-info headache and isolate the search problem.

Closely related:

- **[Cowling, Powley, Whitehouse 2012 — Information Set MCTS](https://eprints.whiterose.ac.uk/id/eprint/75048/1/CowlingPowleyWhitehouse2012.pdf)** — generalizes MCTS to imperfect-info games by treating each node as an *information set* (the set of states consistent with what the player has seen). Applied to Klondike as the canonical case. This is now the standard formulation for partial-info MCTS.
- **[Cazenave 2009 — Nested Monte-Carlo Search](https://www.ijcai.org/Proceedings/09/Papers/083.pdf)** — used on Thoughtful Solitaire.
- **[TU Berlin BA thesis 2025 — "Playing Klondike Solitaire variants with stochastic simulations"](https://doc.neuro.tu-berlin.de/bachelor/2025-BA-DominikMilas.pdf)** — recent MCTS implementation with explicit chance nodes, modern best practices.

## Does MCTS actually beat the alternatives on Klondike?

Honestly, no — search-based solvers win more. The dominant non-LM approach is **Solvitaire** ([long-running search project, arXiv 1906.12314](https://arxiv.org/html/1906.12314v5)), which uses **IDA\* and BFS with handcrafted heuristics** to *prove* deals winnable. MCTS gives you a strong policy; IDA\*/A\* give you a proof. For a research question like "what fraction of Klondike deals are winnable?", proof is what you need, and search-based methods win.

But MCTS is the right tool when:

- You want a *fast online policy* with bounded compute.
- You don't need optimality — just "play this deal well in 30 seconds."
- The state space is too large or heuristic design is hard (Klondike's hidden-info makes heuristics brittle).

## How to make MCTS strong on Klondike (practical recipe)

If you actually want to build this, the stack is:

1. **Information Set MCTS** (ISMCTS) or **POMCP** for the partial-observability part. Maintain either a particle filter of possible worlds or use *determinization* — sample a fully-specified state consistent with what's observed, run MCTS on it as if it were perfect info, repeat many times, average.
2. **Explicit chance nodes** for stock/waste draws. Each draw event is a node with branches for the possible cards.
3. **Informed rollouts** — random rollouts win nothing. Use a biased rollout policy: e.g., greedy "move to foundation if possible" + a tableau heuristic. Or skip rollouts entirely and use a **learned value network** (AlphaZero-style).
4. **Good simulator** — Klondike rules are simple; PySolitaire / BVS Solitaire Collection / pysol_cards exist. The simulator is the bottleneck for rollout speed.
5. **Heuristic for stuck states** — detect dead ends early (no legal moves that reveal new info), backtrack. ISMCTS handles this via statistics; vanilla MCTS needs a hand-coded loop.

With all of this, modern MCTS implementations win ~25–35% of Klondike deals depending on draw mode — competitive with strong heuristic players, well above random (~0%) and well below search-proven optimality (~82% winnable).

## Why MCTS is interesting for MCTS+LLM research

The combination of MCTS + LLM is what makes the current research exciting, and Klondike is a perfect testbed:

- **MCTS gives you the search scaffold** (chance nodes, backprop, value estimates).
- **An LLM can serve as the rollout policy** (instead of biased random) or as a **value function** (instead of full rollouts).
- **The simulator stays as ground truth** for state transitions — you don't need the LLM to hallucinate next states, you just need it to *score* them or *propose* them.

This is exactly the LATS / RAP recipe, just with a card-game simulator instead of a math reasoning task. It's also what Game-RL (ICLR 2026) implicitly does for FreeCell/Spider/Klondike: VLM proposes, simulator evaluates, MCTS-style aggregation picks.

## TL;DR

- **Yes**, MCTS is a viable, well-studied approach to Klondike (Bjarnason 2007/2009, Cowling 2012, ongoing work).
- **Not vanilla MCTS** — you need Information Set MCTS / POMCP for hidden info, chance nodes for draws, and informed rollouts or a value network.
- **Stronger than random**, **weaker than IDA\*** — but IDA\* gives proofs, MCTS gives policies. Different tools, different goals.
- **Best current frontier**: MCTS with a learned value network (AlphaZero-style), or MCTS with an LLM as the proposal/value module (LATS-style), with a real Klondike simulator anchoring transitions.

## Key references

- [Bjarnason, Tadepalli, Fern — Lower Bounding Klondike Solitaire with Monte-Carlo Planning](https://eprints.whiterose.ac.uk/id/eprint/75050/1/EnsDetMagic.pdf) (2007/2009)
- [Cowling, Powley, Whitehouse — Information Set Monte Carlo Tree Search](https://eprints.whiterose.ac.uk/id/eprint/75048/1/CowlingPowleyWhitehouse2012.pdf) (2012)
- [Cazenave — Nested Monte-Carlo Search (Thoughtful Solitaire)](https://www.ijcai.org/Proceedings/09/Papers/083.pdf) (2009)
- [Milas — Playing Klondike Solitaire variants with stochastic simulations (TU Berlin BA thesis)](https://doc.neuro.tu-berlin.de/bachelor/2025-BA-DominikMilas.pdf) (2025)
- [Silver & Veness — Monte-Carlo Planning in Large POMDPs (POMCP)](https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/monte-carlo-planning-in-large-pomdps.pdf) (2010)
- [Sutton — A Survey of Monte Carlo Tree Search Methods](http://www.incompleteideas.net/609%20dropbox/other%20readings%20and%20resources/MCTS-survey.pdf)
- [Solvitaire — The Winnability of Klondike Solitaire and Many Other Patience Games](https://arxiv.org/html/1906.12314v5)
- [Baier — Monte-Carlo Tree Search Enhancements for One-Player and Two-Player Domains](https://project.dke.maastrichtuniversity.nl/games/files/phd/Baier_thesis.pdf)
