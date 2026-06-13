# Deep Dive: Reinforcement Learning for Klondike & Patience Games

Klondike is one of the most interesting open problems in single-player game AI. Unlike chess, Go, or poker, it combines **partial observability + stochasticity + sparse reward + huge horizon** in a setting where brute-force search is hopeless. The "embarrassment of applied mathematics" framing (Gent) is real — the optimal win rate of standard Klondike is still unknown.

---

## 1. Why Klondike is uniquely hard for RL

| Property | Consequence |
|---|---|
| **Partial observability** — 21 face-down cards initially | Markov property fails; state must include hidden-info beliefs (information sets) |
| **Stochastic draws from stock** (Draw 1 vs Draw 3) | Opponent is RNG; non-adversarial but non-deterministic |
| **Sparse reward** — win/lose only at end | Credit assignment over 100s of moves |
| **Huge horizon** — typical game 200–500 moves | Tree explodes; bootstrap learning signals are weak |
| **Massive state space** — ~10^68 states (est.) | No value table; function approximation required |
| **Many "draw-1" turns where only one legal move** | Forced moves waste search budget; classic search techniques need adaptation |
| **Optimal win-rate unknown** | Hard to define "good" baselines / strong opponents |

The combination — **POMDP, stochastic, sparse, long-horizon, single-player** — is a *distinct* problem class from multi-agent imperfect-info games (poker) or perfect-info games (Go). Most RL success stories don't apply cleanly.

---

## 2. The three landmark results (the canonical reading list)

**Yan, Diaconis, Rusmevichientong, Van Roy (2005)** — *Solitaire: Man vs Machine* (Stanford) — the foundational paper.
- Introduced "Thoughtful Klondike" (variant where player knows full card locations).
- Used **level-k rollouts** (a simulation-based policy-improvement scheme — close cousin of MC planning).
- Established an **upper bound ≈ 82%** for Thoughtful Klondike winnability.
- "Human Monte Carlo" estimate gave a lower bound of ~43%.

  [web.stanford.edu/~bvr/pubs/solitaire.pdf](https://web.stanford.edu/~bvr/pubs/solitaire.pdf)

**Bjarnason, Fern, Tadepalli (2009)** — *Lower Bounding Klondike Solitaire with Monte-Carlo Planning* (ICAPS).
- Combined **sparse sampling + UCT** to give the first rigorous empirical lower bound on an *optimal* standard-Klondike policy.
- Result: **wins > 35%** of standard (non-thoughtful) Klondike games.
- The 35%–82% gap is still the open territory.

  [web.engr.oregonstate.edu/~afern/papers/klondike.pdf](https://web.engr.oregonstate.edu/~afern/papers/klondike.pdf)

**Bjarnason, Tadepalli, Fern, Niedner (2009)** — *Searching Solitaire in Real Time*.
- Tightened bounds: **82% ≤ optimal ≤ 91.44%** winnable.
- Real-time search with learned heuristics.

**Blake & Gent (2019, JAIR 2026)** — *The Winnability of Klondike Solitaire and Many Other Patience Games* — **the modern state of the art for winnability numbers**.
- **Solvitaire**: a single general-purpose depth-first solver that handles 73 variants of 35 patience games.
- Reports **81.945% ± 0.084%** for standard Draw-3 Thoughtful Klondike (a 30x tighter CI than Yan).
- FreeCell standard: **99.998881%** ± 0.0002% — essentially solved.
- Techniques: transposition tables, symmetry breaking, dominances (with proofs), streamliners (e.g., always-move-to-foundation, suit symmetry).
- Draw-size curve for Klondike: Draw 1 = 90.48%, Draw 3 = 81.95%, Draw 5 = 53.43%, Draw 7 = 23.78%, Draw 13 = 0.6% (deck is fully known → no hidden info helps).

  [arxiv.org/abs/1906.12314](https://arxiv.org/abs/1906.12314)

> The 82.8% MCTS winrate you may have seen referenced (e.g. on ResearchGate figures of a Klondike study) is reported as an *empirical* result from a tuned MCTS agent, possibly on a different metric or variant. It's a near-upper-bound figure and worth treating with caution — the published rigorous upper bound is 81.945% ± 0.084% on thoughtful Draw-3.

---

## 3. Algorithmic families applied to patience

| Family | Used on Klondike? | Notes |
|---|---|---|
| **Heuristic A*** | FreeCell (very well) | FreeCell is essentially solved; default `freecell-solver` ships heuristic-A*. |
| **Depth-first backtracking + dominances** | Yes (Solvitaire) | Best current for *winnability* estimation; not for play. |
| **Real-time search (Bjarnason)** | Yes | Tightest known bounds on optimal play. |
| **Level-k rollouts / MC planning** | Yes (Yan) | Policy improvement by simulating lower-level play. |
| **MCTS / UCT with determinization** | Yes (recent BA/M.Sc. work) | See Milas & Fricke 2025 (TU Berlin), Godlewski & Sawicki. |
| **DQN / PPO on latent state** | **Almost no rigorous published work on Klondike proper.** | This is the open frontier. |
| **AlphaZero / MuZero style** | No published result on Klondike | MuZero's learned-dynamics approach is *theoretically* well-suited (handles stochasticity + hidden state), but no published Klondike result I'm aware of. |
| **CFR / NFSP** | Not applicable | Multi-agent regret-minimization; Klondike has one player, no opponent to exploit. |
| **Imitation learning from human play** | Possible but no strong baseline | Human play is highly variable; no large-scale Klondike dataset like there is for Go. |

**Survey pointers**: Browne et al. (2012) "A Survey of Monte Carlo Tree Search Methods" §7.7 treats Klondike as the canonical single-player stochastic benchmark. Whitehouse (2014) PhD "MCTS for games with hidden information and uncertainty" extends the framework.

---

## 4. State representation — the design choice that makes or breaks it

Three plausible encodings, none clearly dominant:

**a) Card-locations vector** — 52 cards × ~16 locations (tableau piles, foundations, stock, waste, free cells). Simple, but destroys pile-ordering information. Works for shallow nets.

**b) Pile-of-piles encoding** — 7 tableau piles, each up to 19 deep. Feed as 2D grid (rank/suit × position) with masking. Preserves order. Used in some FreeCell NN work.

**c) Information-set representation** — for the unknown cards, marginalize over a particle-filter of possible stock orderings. Most principled, much heavier. This is what makes Klondike closer to a POMDP than an MDP.

**Reward shaping** matters more than people admit: pure +1/0/-1 on win/loss/draw produces a signal so weak that self-play won't converge. Practical systems use shaped rewards like "cards moved to foundation," "uncovered a face-down card," "legal-move reduction." Solvitaire's streamliners (always-move-to-foundation) hint that *forcing* good behavior can be a substitute for learning it.

---

## 5. Practical starting points (the things you can run)

**Toolkits:**
- **[RLCard](https://rlcard.org/)** (Rice/TAMU, MIT) — `pip install rlcard`. Bundles DQN, NFSP, DMC, CFR. Supports blackjack, Leduc hold'em, limit/no-limit hold'em, Dou Dizhu, Mahjong, UNO, Gin Rummy, Bridge. **Does not include Klondike** — you'd add it as a custom env.
- **OpenSpiel** (DeepMind) — extensive game framework, mostly multi-agent. No built-in Klondike.
- **PyPokerEngine**, **RLCard-Showdown** for GUI.

**Klondike-specific code:**
- `vuonghy2442/lonelybot` (Rust, 13 stars, May 2026) — engine for both Thoughtful and Random Klondike.
- `ShootMe/MinimalKlondike` (C#, 27 stars) — minimal solver.
- `122left/aisol` (C++) — single-file SDL2 Klondike meant to be driven by an RL agent.
- `macroxue/triple-klondike` (C++) — Triple Klondike contest solver.

**Implementers' reference:** the Bjarnason / Yan / Solvitaire papers describe the state-move encoding and move-generation logic you'll need regardless of which learning algorithm you pick.

---

## 6. Open problems & research opportunities (2026)

This is what makes the area worth a deep dive right now:

1. **Close the 35%–82% gap empirically.** A modern deep RL agent (PPO/A3C/SAC with recurrent policies + belief states) *should* comfortably exceed 35% on standard Klondike. The question is *how close* to the 82% upper bound it gets — and what the ceiling is in practice.

2. **MuZero-style self-play on Klondike.** MuZero learns its own dynamics model, which is exactly the right tool for unknown stock orderings. No published result; would be a strong contribution.

3. **Belief-state POMDP formulation.** Treat Klondike as POMDP, track distributions over hidden cards (particle filter or learned), apply point-based value iteration or DQN-with-belief. Mostly unexplored.

4. **Transfer across variants.** Solvitaire solved 35 games with one solver; can a single RL agent transfer between Klondike / FreeCell / Spider / Canfield? Architectural choice (graph nets? transformers over card positions?) is the question.

5. **Curriculum from Thoughtful to Random.** Train first on full-information Klondike, then progressively hide more cards. Standard curriculum-learning idea, untested on patience.

6. **Interpretability.** What features does a learned value function attend to? "Cards in foundation," "moves left in stock," "buried-card density" are obvious, but the interesting question is whether a network discovers a heuristic like Bjarnason's *safe-moves-to-foundations* dominance.

---

## 7. Reading order suggestion

1. Blake & Gent (2026, JAIR) — for the modern landscape and winnability numbers.
2. Bjarnason et al. (2009, ICAPS) — for the lower-bound methodology and the real POMDP framing.
3. Yan et al. (2005) — for level-k rollouts and the historical baseline.
4. Browne et al. (2012) survey — for the MCTS framing.
5. Milas & Fricke (2025) — for the most recent empirical MCTS-on-Klondike work.
