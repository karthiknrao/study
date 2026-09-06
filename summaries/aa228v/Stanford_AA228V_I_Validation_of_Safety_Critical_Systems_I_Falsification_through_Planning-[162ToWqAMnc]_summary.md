### 1. Executive Summary & Core Concepts

**Lecture Overview:**
This lecture transitions from optimization-based falsification (searching for failures via gradient-based methods) to **planning-based falsification** using Tree Search algorithms. The core thesis is that while optimization is powerful, it often struggles with high-dimensional spaces and black-box systems. Instead, tree search methods iteratively build trajectories to systematically explore the state space. We examine **Heuristic Search (RRT)** and **Monte Carlo Tree Search (MCTS)**, highlighting how they balance exploration and exploitation to find likely failure modes in dynamic systems.

**Key Concepts Highlight:**
*   **Likelihood-Weighted Objective:** A hybrid objective function that minimizes robustness while maximizing the likelihood of the trajectory (or minimizing negative log-likelihood). This prevents the algorithm from finding "impossible" failures that require extremely low-probability disturbances.
*   **Local Descent vs. Population Methods:** Local descent (e.g., Gradient Descent) moves a single trajectory toward a minimum, often getting stuck in local minima. Population methods maintain a collection of trajectories to explore the space more broadly, often finding multiple failure modes.
*   **Tree Search Framework:** A unified algorithmic structure consisting of two steps: **Select** (choose which node to expand) and **Extend** (add a new node/trajectory segment). All tree search algorithms (RRT, MCTS, A*) are variations of how these two steps are implemented.
*   **Heuristic Search / RRT:** A method that samples random goal states and extends the tree toward them. It uses simple heuristics (like Euclidean distance) to select nodes. It is effective for exploring the space but lacks guarantees without specific conditions.
*   **Monte Carlo Tree Search (MCTS):** An algorithm that balances exploration and exploitation using a **Lower Confidence Bound (LCB)** value. It maintains a value estimate ($Q$) and visit count ($n$) for each node to decide whether to explore new branches or exploit known promising paths.
*   **Admissible Heuristics:** A heuristic is "admissible" if it never overestimates the cost to reach the goal. When used with discrete state/disturbance spaces, this guarantees the algorithm (specifically A*) will find the optimal path.
*   **Progressive Widening:** A technique in MCTS used for continuous systems. It limits the number of children a node can have based on the number of times the parent node has been visited, preventing infinite branching in continuous disturbance spaces.

---

### 2. Deep Dive: Expanded Lecture Notes

#### Concept 1: Incorporating Likelihood into Falsification
*   **Detailed Explanation:** In previous lectures, minimizing robustness alone could lead to trajectories that are physically impossible or statistically unlikely (e.g., a pendulum requiring a massive, low-probability disturbance to fall). To fix this, we define the likelihood of a trajectory $p(\tau)$ as the product of the probability density of the initial state and the probability densities of all sampled disturbances. We then use a **piecewise objective function**:
    *   If $\tau$ is a failure: Minimize $-p(\tau)$ (i.e., maximize the likelihood of the failure).
    *   If $\tau$ is *not* a failure: Minimize Robustness (move toward failure).
    *   *Practical Note:* In practice, we often combine these into a single weighted objective: $\text{Robustness} + \lambda \cdot (-\log p(\tau))$, where $\lambda$ tunes the trade-off.
*   **Context & Nuance:** This connects to the broader theme of "finding *likely* failures." Pure robustness minimization ignores probability; this approach ensures the found failure is actually probable in the real world.
*   **Analogy:** Imagine searching for a mistake in a game. Pure robustness minimization is like finding a move that wins the game but requires the opponent to make a move that has a 0.01% chance of happening. The likelihood-weighted approach finds a move where the opponent is *likely* to make a mistake.
*   **Key Takeaway:** To find meaningful failures, we must penalize trajectories that rely on statistically improbable sequences of events.

#### Concept 2: Optimization Methodologies (Gradient vs. Direct)
*   **Detailed Explanation:**
    *   **Local Descent (Gradient-based):** Requires computing the gradient of the objective function. This requires a differentiable model (white-box). Examples: Gradient Descent, L-BFGS. These are powerful but fail if the system is a "black box" (where you can simulate but not differentiate).
    *   **Zero-Order/Direct Methods:** Do not require gradients. They only need to evaluate the objective function for a given input. Examples: Nelder-Mead, Hooke-Jeeves. These work for black-box simulators.
*   **Context & Nuance:** The choice of optimizer dictates the system requirements. If you have a complex, opaque simulator, you *must* use direct methods. If you have a simple mathematical model, gradient methods are faster and more precise.
*   **Analogy:** Gradient descent is like driving downhill with a GPS showing the exact slope; Direct methods are like driving downhill by guessing the slope based on where you ended up after a step.
*   **Key Takeaway:** Gradient methods are faster but require differentiability; Direct methods are slower but work on any system where you can run a simulation.

#### Concept 3: The Tree Search Framework (Select & Extend)
*   **Detailed Explanation:** Tree search unifies various algorithms into a loop:
    1.  **Initialize:** Start with a root node (initial state).
    2.  **Select:** Choose a node in the current tree to expand based on a specific rule (e.g., closest to a goal, highest value).
    3.  **Extend:** Sample a disturbance, simulate the next state, and add it to the tree.
    4.  **Repeat:** Continue until a stopping criterion is met.
*   **Context & Nuance:** This framework allows us to compare algorithms like RRT and MCTS easily. The "intelligence" of the algorithm lies entirely in *how* it selects the node and *how* it chooses the disturbance during the extend step.
*   **Analogy:** Think of building a map. "Select" is choosing which unexplored area to look at next; "Extend" is actually walking into that area and marking it on the map.
*   **Key Takeaway:** All tree search algorithms are variations of a "Select and Extend" loop; the specific rules define the algorithm's behavior.

#### Concept 4: Heuristic Search (RRT)
*   **Detailed Explanation:** Rapidly Exploring Random Trees (RRT) is a simple exploration method.
    *   **Select:** Sample a random goal state (or a state in the failure region) and select the node in the tree closest to it (using Euclidean distance).
    *   **Extend:** Sample a random disturbance from the nominal distribution and add the resulting state to the tree.
    *   **Improvements:** To find failures specifically, we can sample goal states *only* from the failure region and choose disturbances that steer the agent *toward* the goal state (rather than purely random disturbances).
*   **Context & Nuance:** Vanilla RRT is great for exploring the space but may miss failures if it doesn't steer toward them. By biasing the goal sampling toward the failure region, we make the search more efficient.
*   **Analogy:** RRT is like a hiker who wants to reach a campsite. They pick a random spot on the map and walk toward it. If we bias the goal to "the cliff edge," they are more likely to find the dangerous spot.
*   **Key Takeaway:** RRT is a flexible exploration tool; its effectiveness depends heavily on how we define the "goal" and the heuristic for selecting nodes.

#### Concept 5: Monte Carlo Tree Search (MCTS)
*   **Detailed Explanation:** MCTS is more sophisticated than RRT because it learns value estimates.
    *   **State:** Each node has a visit count $n$ and a value estimate $Q$ (lower is better for falsification).
    *   **Select (Progressive Widening & LCB):**
        *   First, check if the node has enough children. If $n_{children} \le \sqrt{K \cdot n_{parent}}$, we must **Extend** (create a new child).
        *   If it has enough children, we traverse deeper. We pick the child that minimizes: $Q_{child} + \sqrt{\frac{C}{n_{child}}}$.
        *   The term $\sqrt{\frac{C}{n_{child}}}$ is the **Exploration Bonus**. If a node hasn't been visited much ($n$ is low), this bonus is high, encouraging exploration.
    *   **Extend:** Sample a disturbance, add the node, initialize $Q$ (often via rollouts or heuristics), and propagate values back up the tree (updating $Q$ as a moving average).
*   **Context & Nuance:** MCTS explicitly balances **Exploitation** (following known good paths, low $Q$) and **Exploration** (trying unvisited paths). This is crucial for continuous systems where we can't just "try all options."
*   **Analogy:** MCTS is like a casino player. They track which slots (nodes) pay out well (low cost). They mostly play the good slots (exploitation) but occasionally try a new slot (exploration) to see if it pays even better.
*   **Key Takeaway:** MCTS uses a Lower Confidence Bound (LCB) to decide whether to explore new branches or exploit known promising ones, balancing novelty and known quality.

#### Concept 6: A* and Admissible Heuristics
*   **Detailed Explanation:** If the state and disturbance spaces are **discrete** and the heuristic is **admissible** (never overestimates the cost to goal), the algorithm becomes **A***.
    *   **Cost Function:** $\text{Total Cost} = \text{Current Cost} + \text{Heuristic Cost}$.
    *   **Admissibility:** Example: In a grid world, Euclidean distance is admissible because the straight-line distance is always less than or equal to the actual path length (due to grid constraints).
*   **Context & Nuance:** A* guarantees the optimal path (shortest or most likely) because it systematically explores nodes in order of their estimated total cost. This is a significant upgrade from RRT, which has no such guarantee.
*   **Analogy:** A* is like a GPS with a "traffic prediction" that is never overly optimistic. It guarantees you take the fastest route because it never underestimates the time to reach the destination.
*   **Key Takeaway:** In discrete, admissible scenarios, A* guarantees finding the optimal failure path, whereas RRT/MCTS are heuristic searches without such strict guarantees.

---

### 3. Pathways for Further Exploration

1.  **Topic: The Curse of Dimensionality in Optimization**
    *   **Why it Matters:** The lecture noted that optimization over high-dimensional spaces (initial state + disturbances) is difficult. Understanding *why* this happens is key to appreciating why tree search is often preferred.
    *   **Search/Study Direction:** Look into "Curse of Dimensionality" in the context of numerical optimization and how it affects Gradient Desconvergence.

2.  **Topic: A* Search Algorithm Implementation**
    *   **Why it Matters:** The lecture touched on A* as a special case of heuristic search. Implementing it helps solidify the concepts of admissible heuristics and open/closed lists.
    *   **Search/Study Direction:** Study the specific implementation of A* in grid-worlds, focusing on the priority queue management and the proof of optimality.

3.  **Topic: Progressive Widening in Continuous MCTS**
    *   **Why it Matters:** Standard MCTS is defined for discrete actions. The lecture introduced "Progressive Widening" to handle continuous disturbances. This is a critical adaptation for robotics.
    *   **Search/Study Direction:** Research "Monte Carlo Tree Search for Continuous Action Spaces" and specifically the paper "Progressive Neural Networks" or similar works on continuous control via MCTS.

4.  **Topic: Zero-Order Optimization Methods**
    *   **Why it Matters:** Since gradient methods fail on black-box systems, understanding direct methods (like Nelder-Mead) is essential for practical falsification.
    *   **Search/Study Direction:** Compare the convergence rates of Nelder-Mead vs. Gradient Descent on non-linear, non-convex functions.

5.  **Topic: Coverage Metrics (Discrepancy)**
    *   **Why it Matters:** The lecture mentioned using "star discrepancy" to decide when to stop tree search. This is a vital metric for determining if the search has "covered" enough of the space.
    *   **Search/Study Direction:** Look into "Krylov Subspace" or "Quasi-Monte Carlo methods" to understand how to measure the coverage quality of a set of points in a high-dimensional space.

6.  **Topic: Safety-Critical Systems & Formal Verification**
    *   **Why it Matters:** Falsification is a core component of safety verification. Understanding how this connects to formal methods (like Signal Temporal Logic) provides the "specification" side of the equation.
    *   **Search/Study Direction:** Explore "Reachability Analysis" and "Counter-Example Generation" in the context of control systems.

---

### 4. Comprehension & Review Questions

**Recall & Understanding**
1.  What is the primary reason we incorporate likelihood into the falsification objective function instead of minimizing robustness alone?
2.  What is the difference between a "local descent" method and a "population" method in the context of optimization?
3.  In the Tree Search framework, what are the two fundamental steps that define any tree search algorithm?
4.  What is "Progressive Widening" in the context of Monte Carlo Tree Search, and why is it necessary?
5.  What does it mean for a heuristic to be "admissible"?

**Application & Analysis**
6.  You are given a black-box simulator for a drone that cannot be differentiated mathematically. Which category of optimization method (Gradient-based vs. Direct) should you use, and why?
7.  In RRT, you are sampling goal states uniformly from the entire state space. How would you modify the "Select" step to specifically target failure modes?
8.  In MCTS, you have a node with a low $Q$ value but a very low visit count $n$. How does the Lower Confidence Bound (LCB) formula handle this node during the Select step?
9.  If you apply A* search to a continuous disturbance space (like the continuum world) without discretizing it, what guarantee do you lose compared to the discrete grid world case?
10.  Consider the objective function $J = \text{Robustness} + \lambda (-\log p(\tau))$. If $\lambda$ is set to 0, what type of failures will the algorithm prioritize?

**Critical Thinking & Evaluation**
11.  The lecture states that pure robustness minimization can lead to "very unlikely trajectories." Critique this approach: In what scenarios might finding an "unlikely" failure actually be desirable or necessary for safety certification?
12.  Compare RRT and MCTS. RRT uses a simple distance heuristic, while MCTS learns value estimates. Argue why MCTS is generally more efficient than RRT for finding *likely* failures, despite its higher computational overhead per step.
13.  The lecture relies on "domain expertise" to design heuristics (e.g., Euclidean distance). Discuss the limitations of this approach. Why is it difficult to design an admissible heuristic for complex, high-dimensional systems?

***

**Answer Key & Explanations**

**Recall & Understanding**
1.  **Answer:** To prevent the algorithm from finding failures that require statistically improbable disturbances. We want to find failures that are *likely* to occur in the real world.
2.  **Answer:** Local descent moves a single trajectory toward a minimum (prone to local minima). Population methods maintain multiple trajectories to explore the space more broadly and find multiple failure modes.
3.  **Answer:** The **Select** step (choosing which node to expand) and the **Extend** step (adding a new node/trajectory segment).
4.  **Answer:** Progressive Widening limits the number of children a node can have based on the visit count of the parent ($n_{children} \le \sqrt{K \cdot n_{parent}}$). It is necessary because in continuous systems, a node could theoretically have infinite children, which is computationally impossible.
5.  **Answer:** A heuristic is admissible if it **never overestimates** the cost to reach the goal state.

**Application & Analysis**
6.  **Answer:** You should use **Direct (Zero-Order) methods** (like Nelder-Mead) because they do not require gradients, only function evaluations. Gradient methods require differentiability, which is not possible for a black-box simulator.
7.  **Answer:** Instead of sampling goal states uniformly from the entire state space, you should sample goal states **only from the failure region** (or a region defined by the failure specification).
8.  **Answer:** The LCB formula is $Q + \sqrt{C/n}$. Because $n$ is low, the term $\sqrt{C/n}$ (the exploration bonus) will be **high**. This makes the total value large, so the algorithm is *less* likely to select this node immediately unless the $Q$ value is exceptionally low, balancing the urge to explore versus exploit.
9.  **Answer:** You lose the **guarantee of optimality**. A* guarantees the optimal path only if the state and disturbance spaces are discrete and the heuristic is admissible. In continuous spaces, "optimal" is ill-defined without discretization, and the search space is infinite.
10. **Answer:** If $\lambda = 0$, the algorithm ignores likelihood and prioritizes **robustness** alone. It will find the failure that is "closest" to the nominal trajectory in terms of robustness, regardless of how improbable the disturbances are.

**Critical Thinking & Evaluation**
11.  **Answer:** In safety-critical systems (e.g., nuclear reactors, autonomous cars), we often care about **worst-case** scenarios, not just likely ones. An "unlikely" failure (e.g., a sensor glitch combined with a wind gust) might be catastrophic. Therefore, we may intentionally seek low-probability, high-impact failures to ensure the system is robust against *all* possible disturbances, not just likely ones.
12.  **Answer:** MCTS learns $Q$ values based on actual simulation results (rollouts), allowing it to steer toward *promising* areas of the failure space. RRT relies on static heuristics (like distance) which don't account for the dynamics of the system. MCTS can adaptively focus computational resources on paths that are actually leading to failure, whereas RRT might waste time exploring irrelevant parts of the state space.
13.  **Answer:** Designing admissible heuristics requires deep domain knowledge and understanding of the system's constraints. In high-dimensional, coupled systems, "distance" may not be a meaningful metric for "cost." Furthermore, proving that a heuristic never overestimates cost is mathematically difficult and often impossible for complex, non-linear dynamics, leading to the loss of optimality guarantees.
