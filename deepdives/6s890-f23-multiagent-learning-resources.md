# MIT 6.S890 (Fall 2023) — Topics in Multiagent Learning

Gabriele Farina & Constantinos Daskalakis — MIT

Source: <https://www.mit.edu/~gfarina/6S890_f23/>

This file collects all reading links and lecture-material (slides/notes) links from the schedule, organized by lecture. All course-hosted files are under the base URL `https://www.mit.edu/~gfarina/6S890_f23/`.

---

## Part I — Normal-form games

### Lecture 1 — 9/7 — Introduction to the course and logistics
- **Material (Slides):** <https://www.mit.edu/~gfarina/6S890_f23/lecture1.pdf>

### Lecture 2 — 9/12 — Setting and equilibria: Nash equilibrium
- **Topic:** Definition of normal-form games. Solution concepts and Nash equilibrium. Nash equilibrium existence theorem. Brouwer's fixed point theorem.
- **Material (Slides):** <https://www.mit.edu/~gfarina/6S890_f23/lecture2.pdf>

### Lecture 3 — 9/14 — Setting and equilibria: Correlated equilibrium
- **Topic:** Definition of Correlated and coarse correlated equilibria. Their relationships with Nash equilibria in two-player zero-sum games. Linear programming formulations.
- **Material (Slides):** <https://www.mit.edu/~gfarina/6S890_f23/lecture3.pdf>

### Lecture 4 — 9/19 — Learning in games: Foundations
- **Topic:** Regret and hindsight rationality. Phi-regret minimization and special cases. Connections with equilibrium computation and saddle-point optimization.
- **Reading:** Gordon, Greenwald & Marks, *ICML* 2008 (GGM08) — <https://www.cs.cmu.edu/~ggordon/gordon-greenwald-marks-icml-phi-regret.pdf>
- **Material (Notes):** <https://www.mit.edu/~gfarina/6S890_f23/lecture4.pdf>

### Lecture 5 — 9/21 — Learning in games: Algorithms
- **Topic:** Regret matching, regret matching plus, FTRL and multiplicative weights update.
- **Reading:** Blackwell appendix — <https://www.mit.edu/~gfarina/6S890_f23/L05_appendix.pdf>
- **Material (Notes):** <https://www.mit.edu/~gfarina/6S890_f23/lecture5.pdf>

## Part Ib — Complexity of equilibrium

### Lecture 6 — 9/26 — Nash equilibrium and PPAD complexity
- **Topic:** Sperner's lemma, Brouwer's fixed point, and the PPAD complexity class.
- **Material (Slides):** <https://www.mit.edu/~gfarina/6S890_f23/lecture6.pdf>

### Lecture 7 — 9/28 — PPAD-completeness of Nash equilibria, and open problems (Part I)
- **Material (Slides):** <https://www.mit.edu/~gfarina/6S890_f23/lecture7.pdf>

### # — 10/3 — Project Brainstorming
- See Canvas for a list of project ideas. (No material)

### Lecture 8 — 10/5 — PPAD-completeness of Nash equilibria, and open problems (Part II)
- **Topic:** Arithmetic Circuit SAT, and the PLS, PPP, and PPA complexity classes.
- **Material (Slides):** <https://www.mit.edu/~gfarina/6S890_f23/lecture8.pdf>

## Part II — Stochastic games

### Lecture 9 — 10/12 — Stochastic games
- **Topic:** Minimax theorem, and existence of equilibrium. Stationary Markov Nash equilibria.
- **Reading (Proof notes):** <https://www.mit.edu/~gfarina/6S890_f23/L9_notes.pdf>
- **Material (Slides):** <https://www.mit.edu/~gfarina/6S890_f23/lecture9.pdf>

### Lecture 10 — 10/17 — Computation and learning of equilibria in stochastic games (Part I)
- **Topic:** Upper bounds. (Guest lecture by Noah Golowich)
- **Material (Notes):** <https://www.mit.edu/~gfarina/6S890_f23/lecture10.pdf>

### Lecture 11 — 10/19 — Computation and learning of equilibria in stochastic games (Part II)
- **Topic:** Lower bounds. (Guest lecture by Noah Golowich)
- **Material (Notes):** <https://www.mit.edu/~gfarina/6S890_f23/lecture11.pdf>

## Part III — Imperfect-information games

### Lecture 12 — 10/24 — Foundations of imperfect-information extensive-form games
- **Topic:** Complete versus imperfect information. Kuhn's theorem. Normal-form and sequence-form strategies. Similarities and differences with normal-form games.
- **Reading:** Farina et al., *ICML* 2022 (Kernelized MWU) — <https://proceedings.mlr.press/v162/farina22a/farina22a.pdf>
- **Material (Slides):** <https://www.mit.edu/~gfarina/6S890_f23/lecture12.pdf>

### Lecture 13 — 10/26 — Linear programming for Nash equilibrium in two-player zero-sum extensive-form games
- **Topic:** Formulation and implementation details.
- **Material (Slides):** <https://www.mit.edu/~gfarina/6S890_f23/lecture13.pdf>

### Lecture 14 — 10/31 — Learning in imperfect-information extensive-form games (Part I)
- **Topic:** Construction of learning algorithms for extensive-form games.
- **Reading (CFR appendix):** <https://www.mit.edu/~gfarina/6S890_f23/L14_cfr.pdf>
- **Material (Slides):** <https://www.mit.edu/~gfarina/6S890_f23/lecture14.pdf>

### Lecture 15 — 11/2 — Learning in imperfect-information extensive-form games (Part II) and sequential irrationality
- **Topic:** Proof of Counterfactual Regret Minimization (CFR). Introduction to sequential irrationality.
- **Material (Slides):** <https://www.mit.edu/~gfarina/6S890_f23/lecture15.pdf>

### Lecture 16 — 11/7 — Equilibrium refinements and team coordination
- **Topic:** Extensive-form perfect equilibria and quasi-perfect equilibrium.
- **Material (Slides):** <https://www.mit.edu/~gfarina/6S890_f23/lecture16.pdf>

### Lecture 17 — 11/9 — Scalability-enhancing techniques
- **Material (Slides):** <https://www.mit.edu/~gfarina/6S890_f23/lecture17.pdf>

### B1 — 11/14 — Project break (no class)
### B2 — 11/16 — Project break (no class)

## Part IV — Nonconcave games

### Lecture 18 — 11/21 — Aspects of nonconcave games
- **Topic:** Overview, challenges, and local Nash equilibria.
- **Material (Slides):** <https://www.mit.edu/~gfarina/6S890_f23/lecture18.pdf>

### Lecture 19 — 11/28 — Aspects of nonconcave games
- **Topic:** Randomized Nash equilibria. Infinite games, threshold dimension and Littlestone dimension.
- **Material (Slides):** <https://www.mit.edu/~gfarina/6S890_f23/lecture19.pdf>

### Lecture 20 — 11/30 — Aspects of nonconcave games
- **Topic:** Infinite games. Double-oracle algorithm for infinite games.
- **Material (Slides):** <https://www.mit.edu/~gfarina/6S890_f23/lecture20.pdf>

### P1 — 12/5 — Project presentations
### P2 — 12/7 — Project presentations

---

## Summary counts
- **External papers (readings):** 2
  - Gordon, Greenwald & Marks, *ICML* 2008 (Lecture 4)
  - Farina et al., *ICML* 2022 — Kernelized MWU (Lecture 12)
- **Course-hosted reading appendices/notes:** 3 (Lectures 5, 9, 14)
- **Lecture material (slides/notes PDFs):** 20 (Lectures 1–20)

## Notes
- The course page URL-encodes underscores as `%5F` (e.g. `L05%5Fappendix.pdf`); decoded forms (`L05_appendix.pdf`) resolve to the same files and are used above.
- Unlike the CMU course, most "material" here is hosted locally as `lectureN.pdf`; the link text is either *Slides* or *Notes* (noted per lecture).
- Only Lectures 4 and 12 reference external research papers; the other readings are short course-hosted appendices/proof notes.
