### 1. Executive Summary & Core Concepts

**Lecture Overview:**
This lecture bridges the gap between classical symbolic AI and modern machine learning by introducing **search** as a fundamental component of intelligence, distinct from reflexive pattern recognition. It formalizes problem-solving via "search problems" (defined by states, successors, and termination conditions) and compares exact algorithms (Exhaustive Search, Dynamic Programming) against heuristic, approximate methods (Best-of-N, Beam Search). The lecture argues that while deep learning handles perception and reflexive mapping, search handles reasoning and planning, and the most robust AI systems combine learned cost functions (via ML) with search algorithms to optimize outcomes.

**Key Concepts Highlight:**
*   **The Bitter Lesson:** A philosophical stance (popularized by Rich Sutton) asserting that general methods leveraging computation (search and learning) outperform hand-crafted heuristics. It justifies why we still study search algorithms in the era of deep learning.
*   **Search Problem Formalization:** The abstraction of a problem into three core components: a **Start State**, a **Successors** function (defining actions, costs, and resulting states), and an **is_end** function. This structure allows general algorithms to solve specific, complex problems.
*   **State Space & Constraints:** The "State" must contain *all* information necessary to evaluate future actions. If constraints exist (e.g., "cannot take tram twice"), the state must explicitly track that history (e.g., a boolean flag for the last action), otherwise the algorithm will "cheat" or fail.
*   **Future Cost (Recurrence):** The core mathematical definition for exact search. $FutureCost(State) = \min_{s \in Successors} (Cost(Action) + FutureCost(NextState))$. This recursive decomposition is the foundation of both Exhaustive Search and Dynamic Programming.
*   **Dynamic Programming (DP):** "Exhaustive Search plus Caching." It avoids re-exploring states by storing previously computed solutions. It offers exponential speedups but requires that the total number of states fits in memory.
*   **Best-of-N (Rollouts):** A stochastic, parallelizable heuristic where you run $N$ independent rollouts using a policy (often a language model) and select the best result. It is simple, parallelizable, and converges to the optimal solution as $N \to \infty$ (assuming the policy has positive probability on optimal actions).
*   **Beam Search:** A deterministic heuristic that keeps only the top $K$ (beam width) most promising partial solutions at each step. It prunes less-likely paths early to save memory, trading optimality guarantees for efficiency.

### 2. Deep Dive: Expanded Lecture Notes (The "Teaching" Section)

#### 1. The Bitter Lesson & The Role of Search
*   **Detailed Explanation:** In the first lecture, we established that intelligence involves perception, reasoning, action, and learning. Machine learning (specifically neural networks) excels at "learning" and mapping inputs to outputs reflexively. However, "reasoning" and "planning" require **search**. The lecture cites Rich Sutton’s "The Bitter Lesson," which argues that hand-crafted priors (human-designed rules) are fragile. Instead, general methods that leverage raw computation—specifically **search** and **learning**—are the only scalable approaches.
*   **Context & Nuance:** Why is this relevant in 2025? Because we now have powerful learned models (like LLMs) that can define the "costs" or "probabilities" of actions, but we still need a search mechanism to organize that computation at test time. Search provides the structure; Learning provides the weights.
*   **Analogy:** Think of a chess engine. A neural network might predict "this move looks good" based on pattern recognition (Learning). But to find the *best* move, the engine must search thousands of potential future positions to see if that move leads to a win (Search).
*   **Key Takeaway:** Search is not obsolete; it is the engine that allows learned models to reason through complex, multi-step problems.

#### 2. Formalizing the Search Problem
*   **Detailed Explanation:** To solve problems generally, we must translate natural language descriptions into code. A **Search Problem** is parameterized by:
    1.  **Start State:** The initial configuration.
    2.  **Successors:** A function taking a state and returning a list of `(action, cost, next_state)` tuples.
    3.  **is_end:** A boolean check for termination.
    *Example:* In the "Magic Tram" problem, walking costs 1 minute, and the tram costs 2 minutes but doubles your location index.
*   **Context & Nuance:** The **State** is the most critical component. It must be "Markov"—containing all relevant history.
    *   *Constraint Example:* If the rule is "You can't take the tram twice in a row," a simple state `(location)` is insufficient because it doesn't remember the last action. You must expand the state to `(location, last_action_was_tram)`.
    *   *Trade-off:* Including *too much* history (e.g., the entire path) creates a massive state space, making algorithms like Dynamic Programming intractable. You want the *minimal* state that captures validity.
*   **Analogy:** The State is your "memory." If your memory is missing a detail (like "I just used the tram"), you might make an invalid move. If your memory is too bloated (remembering every step ever taken), you run out of RAM.
*   **Key Takeaway:** Correctly defining the State is the hardest part of modeling. If the state doesn't capture the constraints, the algorithm will "hack" the system (e.g., going into negative tickets).

#### 3. Exhaustive Search & Future Cost
*   **Detailed Explanation:** Exhaustive search tries *all* possible sequences of actions. It relies on the **Future Cost** recurrence:
    $$F(S) = \min_{a \in Actions} (Cost(a) + F(NextState(a)))$$
    *   Base Case: If `is_end(State)` is true, Future Cost is 0.
    *   Recursion: Otherwise, pick the action that minimizes the immediate cost plus the future cost of the resulting state.
*   **Context & Nuance:** This is mathematically elegant but computationally expensive. It explores states multiple times (e.g., reaching state 2 via walk, then state 2 via tram, then state 2 again via a different path). It has exponential time complexity but linear memory complexity (stack depth).
*   **Analogy:** Imagine a hiker trying to find the shortest path through a dense forest. Exhaustive search is the hiker who physically walks every single branch, dead-end included, before deciding which path was shortest.
*   **Key Takeaway:** Exhaustive search guarantees the optimal solution but is generally intractable for large problems due to exponential time complexity.

#### 4. Dynamic Programming (DP)
*   **Detailed Explanation:** DP is "Exhaustive Search + Caching." It uses a lookup table (memoization). Before computing the future cost for a state, it checks if that state has already been solved.
    *   *Time Complexity:* Linear in the number of states (plus cost of transitions).
    *   *Memory Complexity:* Linear in the number of states.
*   **Context & Nuance:** DP is only faster if there is **overlap** (many paths lead to the same state). If every action leads to a unique new state (like generating a unique sentence), DP offers no speedup.
*   **Analogy:** Instead of walking every branch in the forest, you put a sign on a tree saying "I already checked this branch; the path forward is 5 miles." When you encounter that tree again, you just read the sign.
*   **Key Takeaway:** DP is the go-to for problems with a manageable number of states and high redundancy (overlap). If the state space is too large (e.g., billions of states), you cannot store the cache in memory.

#### 5. Best-of-N (Rollouts)
*   **Detailed Explanation:** A heuristic approach where you don't solve the problem exactly. Instead, you define a **Policy** (a function mapping states to actions, often probabilistic). You run $N$ independent rollouts (trajectories) from start to end, and pick the best result.
*   **Context & Nuance:**
    *   *Parallelism:* Because rollouts are independent, this is "embarrassingly parallel." You can throw 1,000 GPUs at it.
    *   *Convergence:* As $N \to \infty$, Best-of-N converges to the optimal solution (assuming the policy has non-zero probability for the optimal actions).
    *   *Use Case:* Highly effective when you have a good learned policy (like an LLM) that is already "informed" about good moves.
*   **Analogy:** Instead of calculating the perfect route, you ask 100 random drivers to drive to work and pick the one who got there fastest.
*   **Key Takeaway:** Best-of-N is simple, parallelizable, and works surprisingly well when combined with powerful learned models (LLMs).

#### 6. Beam Search
*   **Detailed Explanation:** A deterministic heuristic that keeps a "beam" of width $K$ (e.g., 10). At each step, it expands all current candidates, evaluates them, and keeps only the top $K$ lowest-cost partial solutions.
    *   *Beam Width 1:* This is **Greedy Search** (always pick the best immediate move).
    *   *Beam Width $\infty$:** This approaches Exhaustive Search.
*   **Context & Nuance:** Beam search is deterministic (unlike Best-of-N). It is used when you want to control the memory usage strictly. It can get stuck in local optima if the beam width is too small, but it is generally more stable than random rollouts.
*   **Analogy:** Driving a car in the dark with a headlight that only illuminates 10 cars ahead. You don't look at the whole road, just the 10 most promising cars.
*   **Key Takeaway:** Beam search trades optimality guarantees for memory efficiency. It is the standard for decoding sequences in NLP.

#### 7. Test-Time Compute & Language Models
*   **Detailed Explanation:** The lecture applies these concepts to LLMs.
    *   **State:** The prompt + generated text so far.
    *   **Action:** Predict next token.
    *   **Cost:** Negative log-likelihood of the token (we want to minimize cost, so we maximize probability).
    *   **Verifier:** A bonus (negative cost) is applied if the output passes a check (e.g., contains a number).
    *   **Method:** Use Best-of-N sampling. Generate many solutions, let the verifier filter them, and pick the best.
*   **Context & Nuance:** This connects "Symbolic AI" (search structure) with "Statistical AI" (learned probabilities). The LLM provides the "weights" or "costs," and the search algorithm organizes the computation at test time.
*   **Analogy:** A writer drafting 50 versions of a paragraph, checking each against a rubric, and picking the best one.
*   **Key Takeaway:** Modern AI systems often use "Test-Time Compute" (spending more time/compute on inference) to solve harder problems, leveraging search algorithms to orchestrate the LLM's generation.

### 3. Pathways for Further Exploration

1.  **The Topic/Concept:** Markov Decision Processes (MDPs)
    *   **Why it Matters:** The lecture explicitly states that search is a special case of MDPs. MDPs are the formal framework for Reinforcement Learning, where the "world" is stochastic rather than deterministic.
    *   **Search/Study Direction:** Study the Bellman Equation and how it differs from the deterministic search recurrence. Look into how "value iteration" handles cycles (which exhaustive search cannot).

2.  **The Topic/Concept:** Uniform Cost Search (UCS)
    *   **Why it Matters:** The lecture mentions UCS as a way to handle cycles and find optimal paths without exhaustive recursion. It is the "A* without heuristics" algorithm.
    *   **Search/Study Direction:** Compare UCS vs. Dijkstra’s algorithm. Understand the priority queue implementation and how it differs from the recursive DP approach.

3.  **The Topic/Concept:** A* Search and Heuristics
    *   **Why it Matters:** The lecture discusses "exact" and "heuristic" search. A* is the bridge between them, using a heuristic function $h(n)$ to guide the search while still guaranteeing optimality (if the heuristic is admissible).
    *   **Search/Study Direction:** Study the "admissibility" and "consistency" of heuristics. How does A* reduce the search space compared to Beam Search?

4.  **The Topic/Concept:** Particle Filtering
    *   **Why it Matters:** The lecture notes that Beam Search is related to particle filtering. Particle filtering is a probabilistic version of beam search used in tracking and robotics.
    *   **Search/Study Direction:** Look into how particle filters handle stochastic environments and how they differ from the deterministic pruning of Beam Search.

5.  **The Topic/Concept:** LLM Decoding Strategies (Nucleus Sampling vs. Beam Search)
    *   **Why it Matters:** The lecture uses LLMs as a primary example. In practice, LLMs rarely use pure Beam Search; they often use Nucleus (top-k) sampling.
    *   **Search/Study Direction:** Investigate the trade-offs between diversity (Nucleus) and consistency (Beam Search) in LLM generation. Why is "Best-of-N" becoming more popular in LLM inference?

### 4. Comprehension & Review Questions

**Recall & Understanding**
1.  What are the three core components required to formally define a Search Problem?
2.  In the context of the "Magic Tram" problem, why is it necessary to include "tickets" or "last action" in the state definition rather than just the location?
3.  What is the mathematical definition of "Future Cost" in exhaustive search?
4.  How does Dynamic Programming differ from Exhaustive Search in terms of code implementation?
5.  What is the relationship between Beam Width and Greedy Search?

**Application & Analysis**
6.  You are designing a search algorithm for a game where the state space is $10^{100}$ (astronomically large) but the number of steps to a solution is short (e.g., 20 moves). Which algorithm (DP, Exhaustive, Best-of-N, Beam) is most appropriate, and why?
7.  In the LLM example, the cost is defined as the negative log-likelihood of the next token. Why is this specific cost function chosen, and how does it relate to the goal of "maximizing probability"?
8.  Suppose you have a problem where every action leads to a unique, never-before-seen state (no overlap). Why would Dynamic Programming provide no significant speedup over Exhaustive Search in this specific scenario?
9.  How does the "parallelism" advantage of Best-of-N differ from the "determinism" advantage of Beam Search? Which is better for a distributed computing cluster?

**Critical Thinking & Evaluation**
10. The lecture cites "The Bitter Lesson" to argue against hand-crafted priors. Critique this view: Are there scenarios where hand-crafted priors (symbolic rules) are still superior to general search + learning?
11. Beam Search is deterministic, but it can still fail to find the global optimum if the beam width is too small. How does this failure mode differ from the failure mode of Exhaustive Search (which is intractability)?
12. Consider the "Test-Time Compute" trend. If we can simply increase the number of rollouts (Best-of-N) to infinity, do we still need to optimize the training of the underlying model? Or does search completely subsume learning?

***

**Answer Key & Explanations**

**Recall & Understanding**
1.  **Start State**, **Successors** (actions, costs, next states), and **is_end**.
2.  Because the state must contain *all* information needed to evaluate future actions. If the constraint is "can't take tram twice," the state must track the last action. If it only tracks location, the algorithm doesn't know if it *can* take the tram, leading to invalid solutions.
3.  $F(S) = \min_{a \in Actions} (Cost(a) + F(NextState(a)))$. It is the minimum cost to go from the current state to the end state.
4.  Dynamic Programming adds a **cache** (lookup table). It checks if a state has been solved before; if so, it returns the cached result instead of recursing.
5.  If Beam Width = 1, Beam Search becomes **Greedy Search** (it only keeps the single best partial solution at each step).

**Application & Analysis**
6.  **Best-of-N** or **Beam Search**. DP and Exhaustive Search are impossible because the state space is too large to store or explore. Since the path length is short, rollouts are cheap. Best-of-N is ideal if you have a good policy; Beam is ideal if you want deterministic results.
7.  We want to *minimize* cost. Log-likelihood is high for probable tokens. Negative log-likelihood is low for probable tokens. Minimizing negative log-likelihood is mathematically equivalent to maximizing the probability of the sequence.
8.  DP relies on **overlap** (reusing sub-solutions). If every state is unique, there is no overlap, so the cache is never hit. The time complexity remains exponential (or high), offering no speedup.
9.  Best-of-N is parallel (independent rollouts) but stochastic (random). Beam Search is sequential (step-by-step) but deterministic. For a distributed cluster, Best-of-N is superior because you can distribute the rollouts across many machines.

**Critical Thinking & Evaluation**
10.  *Sample Answer:* In safety-critical systems (e.g., medical dosing, nuclear safety), hand-crafted priors might be preferred because general search can produce "black box" reasoning that is hard to audit. Also, in small, well-defined domains, symbolic rules can be more efficient than training a massive model.
11.  Exhaustive Search fails due to **resource limits** (time/memory) and cannot guarantee completion. Beam Search fails due to **pruning**; it *can* complete, but it might discard the path to the global optimum because it looked locally bad early on.
12.  *Sample Answer:* Search does not subsume learning. The quality of the "policy" (the learned model) dictates how much search is needed. A bad model requires infinite search to find a good answer. A good model requires less search. They are synergistic, not mutually exclusive.
