Here is your comprehensive study guide, synthesized from the lecture transcript into a structured masterclass format.

### 1. Executive Summary & Core Concepts

**Lecture Overview:**
This lecture completes the theoretical foundation of Model Predictive Control (MPC) by proving **persistent feasibility** and **stability** for the MPC control law. It establishes that by selecting a terminal set ($X_F$) that is "controlled invariant," we guarantee the system remains feasible at every time step. Furthermore, by leveraging **Lyapunov stability theory**, we demonstrate that the optimal cost function of the MPC problem acts as a valid Lyapunov function, ensuring the closed-loop system converges to the origin. The lecture concludes with practical implementation details, including the use of explicit MPC for offline computation and specific formulation changes required for **trajectory tracking** to avoid oscillatory behavior.

**Key Concepts Highlight:**
*   **Persistent Feasibility:** The property that the MPC optimization problem has at least one feasible solution at every time step $k$, ensuring the controller never gets "stuck" or fails to produce a control action.
*   **Controlled Invariant Set:** A set of states $X$ such that for every state $x \in X$, there exists a control input $u$ that keeps the next state within $X$. If the terminal set $X_F$ is controlled invariant, persistent feasibility is guaranteed.
*   **Lyapunov Stability Function:** A scalar function $V(x)$ (often the optimal cost $J^*$) that is positive definite and decreases along the system trajectories. If such a function exists, the system is asymptotically stable.
*   **Terminal Set ($X_F$):** A constraint set imposed on the final state of the MPC horizon. It is a "tuning knob" used to guarantee feasibility and stability. The origin is a trivial choice but often suboptimal due to high control effort.
*   **Explicit MPC (Offline Optimization):** A method where the optimization is solved offline to create a piecewise-affine mapping of the state space. Online, the controller simply looks up the region of the current state and applies the pre-computed gain, avoiding real-time optimization.
*   **Trajectory Tracking Formulation:** A modified MPC setup where the optimization variables are **delta controls** ($\Delta u = u_k - u_{k-1}$) rather than absolute controls, ensuring that constant inputs (like thrust) are handled correctly to prevent oscillations.

---

### 2. Deep Dive: Expanded Lecture Notes (The "Teaching" Section)

#### Concept 1: Persistent Feasibility and the Feasibility Lemma
*   **Detailed Explanation:** Persistent feasibility is the baseline requirement for an MPC loop to function. The core logical step is proving that if the system starts in a feasible state, the next state will also be feasible. The lecture defines the **one-step controllable set** (or pre-image) of a set $X_1$ as the set of states $x$ for which there exists a control $u$ such that $Ax + Bu \in X_1$.
*   **Context & Nuance:** The lecture proves a lemma: If the set $X_1$ (the feasible states starting from step 1 of the horizon) is a **controlled invariant set**, then the MPC problem is persistently feasible. The proof relies on "pattern matching" definitions: since $X_1$ is controlled invariant, any state in $X_1$ can stay in $X_1$. Therefore, $X_1$ is a subset of the feasibility set $X_0$. If we start in $X_0$, the MPC law moves us to a new state that is still in $X_0$, preserving feasibility indefinitely.
*   **Analogy:** Imagine a game of "Tag" where the "safe zone" is a specific circle. The lemma proves that if the rules of the game ensure that anyone in the circle can always stay in the circle, then as long as you start in the circle, you will never be "tagged out" (infeasible).
*   **Key Takeaway:** To guarantee the MPC never fails, the terminal set must be a controlled invariant set.

#### Concept 2: The Stability Theorem via Lyapunov Theory
*   **Detailed Explanation:** We transition from "can we solve the problem?" (feasibility) to "does the solution work?" (stability). The lecture introduces **Lyapunov stability**: a system is stable if there exists a function $V(x)$ (energy) that is zero at the origin, positive elsewhere, and strictly decreases over time.
*   **Context & Nuance:** The celebrated theorem states that for a linear system with quadratic costs, the **optimal cost function** $J^*(x)$ of the MPC problem is itself a valid Lyapunov function. The proof shows that $J^*(x_{k+1}) < J^*(x_k)$ by comparing the optimal cost at step $k$ with a "shifted" candidate sequence at step $k+1$. This relies on **Assumption 3**: there must exist a control $v$ such that the change in the terminal cost plus the stage cost is negative (or zero).
*   **Analogy:** Think of a ball rolling down a hill. The "Lyapunov function" is the height of the ball. If we can prove the ball always moves to a lower height (energy dissipates), it must eventually rest at the bottom (the origin). We don't need to solve the differential equation of the roll; we just need to prove the energy is dropping.
*   **Key Takeaway:** The optimal cost $J^*$ is the Lyapunov function. If the terminal cost $P$ is chosen correctly, the MPC law asymptotically stabilizes the system to the origin.

#### Concept 3: Tuning the Terminal Set ($X_F$) and Cost ($P$)
*   **Detailed Explanation:** The lecture provides two specific, actionable recipes for choosing $X_F$ and the terminal cost matrix $P$:
    1.  **Stable Dynamics ($A$ is stable):** Let $X_F$ be the **maximal positive invariant set** of the uncontrolled dynamics ($x_{k+1} = Ax_k$). Choose $P$ as the solution to the **Lyapunov Equation** ($-x^TPx + x^TQx + x^TA^TPAx \leq 0$).
    2.  **Unstable Dynamics ($A$ is unstable):** Use the infinite-horizon LQR gain $F_\infty$. Let $X_F$ be the maximal positive invariant set of the **closed-loop LQR dynamics** ($x_{k+1} = (A-BF)x_k$). Choose $P$ as the solution to the **Riccati Equation**.
*   **Context & Nuance:** Choosing the origin as $X_F$ is mathematically valid (it is controlled invariant) but practically suboptimal. It forces the system to use maximum control effort to reach zero, potentially violating constraints or wasting energy. The "maximal" sets allow for more flexibility.
*   **Analogy:** If you are driving a car (stable dynamics), you can rely on the engine to coast to a stop (positive invariant set). If you are driving a rocket (unstable dynamics), you need active thrusters (LQR) to keep it stable, and your "terminal set" is the region where the thrusters can keep it stable without exploding.
*   **Key Takeaway:** To guarantee stability, $P$ must be derived from the Lyapunov or Riccati equation corresponding to the system's stability characteristics.

#### Concept 4: Explicit MPC (Offline Computation)
*   **Detailed Explanation:** Historically, solving the optimization online was slow. **Explicit MPC** partitions the state space into polyhedral regions. In each region, the optimal control law is a linear function ($u = F_j x + G_j$).
*   **Context & Nuance:** Instead of solving the QP online, you simply determine which region your current state is in and apply the pre-computed gain. This removes the need for a real-time solver, which is crucial for safety-critical systems (where solver bugs are unacceptable) and embedded hardware.
*   **Analogy:** Instead of calculating the route to a destination every time you turn a corner (online optimization), you have a pre-mapped grid where every block tells you exactly which way to turn (lookup table).
*   **Key Takeaway:** Explicit MPC trades offline computational complexity for online speed and reliability, though modern solvers have reduced the necessity of this approach.

#### Concept 5: Trajectory Tracking and Delta Controls
*   **Detailed Explanation:** Standard MPC minimizes tracking error and control effort. However, if the system requires a non-zero steady-state input (e.g., an aircraft needs constant thrust to maintain altitude), a naive formulation causes oscillations: the controller reduces thrust to lower cost, the plane drops, the tracking error increases, and the controller adds thrust again.
*   **Context & Nuance:** The solution is to optimize over **delta controls** ($\Delta u = u_k - u_{k-1}$). In perfect tracking, the tracking error is zero, and the control is constant, meaning $\Delta u = 0$. This allows the MPC to "hold" a constant value (thrust) without penalizing it, as the cost is only applied to *changes* in control.
*   **Analogy:** A thermostat (naive) might turn the heater off completely to save energy, causing the room to get cold. A delta-control system only adjusts the *change* in heat, allowing it to maintain a steady, non-zero output without "flickering."
*   **Key Takeaway:** For tracking problems, formulate the optimization in terms of $\Delta u$ to handle non-zero steady-state control inputs correctly.

---

### 3. Pathways for Further Exploration

1.  **Topic:** Computational Tools for Invariant Sets (MPT3 Toolbox)
    *   **Why it Matters:** The lecture mentions using a toolbox to compute maximal positive invariant sets. Understanding how to generate these sets numerically is critical for applying the theory to real systems.
    *   **Search/Study Direction:** Look into the **MPT3 (Multiparameter Toolbox)** documentation. Study how to define polyhedral constraints and compute "maximal invariant sets" for linear systems.

2.  **Topic:** Discrete-Time Riccati Equations
    *   **Why it Matters:** The lecture asserts that $P$ is the solution to the Riccati equation for unstable systems. Understanding the derivation and properties of this equation connects MPC to optimal control theory.
    *   **Search/Study Direction:** Study the **Discrete-Time Algebraic Riccati Equation (DTARE)**. Understand how it relates to the infinite-horizon LQR problem and why its solution is positive definite.

3.  **Topic:** Piecewise-Affine (PWA) Systems
    *   **Why it Matters:** This is the mathematical foundation of Explicit MPC.
    *   **Search/Study Direction:** Explore **MPC Partitioning** algorithms. Understand how the state space is partitioned into regions where the solution is affine, and how the number of regions grows with the horizon length.

4.  **Topic:** Lyapunov Functions for Nonlinear Systems
    *   **Why it Matters:** The lecture focuses on linear systems. Extending Lyapunov stability to nonlinear MPC is a major area of research.
    *   **Search/Study Direction:** Investigate **Sum-of-Squares (SOS) methods** for constructing Lyapunov functions for nonlinear systems, which are used in modern nonlinear MPC.

5.  **Topic:** Constrained LQR vs. MPC
    *   **Why it Matters:** The lecture positions MPC as a "constrained LQR." Understanding the differences clarifies why we need the finite horizon $N$ and terminal constraints.
    *   **Search/Study Direction:** Compare **Infinite-Horizon LQR** (which is always stable if $P$ is correct) with **Finite-Horizon MPC** (which requires $X_F$ and $P$ to ensure stability).

6.  **Topic:** Real-Time Implementation Challenges
    *   **Why it Matters:** The lecture mentions that modern solvers (like CVX) have made online optimization faster.
    *   **Search/Study Direction:** Look into **Predictable QP Solvers** and **Parallel MPC** architectures to understand how industry handles the "online" computation delays mentioned in the lecture.

---

### 4. Comprehension & Review Questions

**Recall & Understanding**
1.  What is the definition of a **controlled invariant set**?
2.  What is the primary purpose of the terminal set $X_F$ in the MPC formulation?
3.  In the context of the stability theorem, what function serves as the Lyapunov function for the closed-loop MPC system?
4.  Why is choosing the origin (0) as the terminal set $X_F$ generally considered suboptimal for practical control?
5.  What is the "one-step controllable set" (or pre-image) of a set $X$?

**Application & Analysis**
6.  If the system dynamics matrix $A$ is asymptotically stable, which equation must the terminal cost matrix $P$ satisfy to guarantee stability?
7.  If the system dynamics matrix $A$ is unstable, what specific controller gain should be used to define the terminal set $X_F$, and which equation defines $P$?
8.  In a trajectory tracking problem for an aircraft maintaining altitude, why does a naive MPC formulation (penalizing absolute control $u$) lead to oscillations?
9.  How does formulating the optimization in terms of **delta controls** ($\Delta u$) resolve the oscillation issue in tracking problems?
10.  If you use Explicit MPC, what are the two main advantages over online optimization in safety-critical applications?

**Critical Thinking & Evaluation**
11.  The lecture states that "if $X_F$ is controlled invariant, MPC is persistently feasible." Critique the practicality of this theorem: Why is the theorem itself not considered "useful" for tuning until we connect $X_F$ to the terminal constraint?
12.  Consider a scenario where computational resources are extremely limited (e.g., a microcontroller). Would you choose Explicit MPC or a fast online QP solver? Justify your choice based on the lecture's points regarding validation and solver reliability.
13.  The stability proof relies on "Assumption 3" (the existence of a control $v$ such that the cost decreases). How does this assumption relate to the physical intuition of "energy dissipation" in the system?

***

### Answer Key & Explanations

**Recall & Understanding**
1.  **Controlled Invariant Set:** A set $X$ is controlled invariant if for every state $x \in X$, there exists a control input $u$ such that the next state $Ax + Bu$ remains in $X$.
2.  **Purpose of $X_F$:** It is a "tuning knob" added as a constraint to the MPC problem to guarantee **persistent feasibility** (the problem always has a solution) and to help guarantee **stability**.
3.  **Lyapunov Function:** The **optimal cost function** $J^*(x)$ (the value of the objective function of the MPC problem) serves as the Lyapunov function.
4.  **Suboptimality of Origin:** Constraining the system to reach the origin in $N$ steps restricts the space of feasible control sequences. It often forces the system to use excessive control effort to hit zero, leading to suboptimal performance and potential constraint violations.
5.  **One-Step Controllable Set:** The set of states $x$ such that there exists a feasible control $u$ where $Ax + Bu$ transitions into the set $X$.

**Application & Analysis**
6.  **Stable A:** $P$ must satisfy the **Lyapunov Equation**: $-x^TPx + x^TQx + x^TA^TPAx \leq 0$ (specifically, $P$ is the unique positive definite solution to the Lyapunov equation associated with $A$ and $Q$).
7.  **Unstable A:** Use the infinite-horizon **LQR gain** ($F_\infty$). $X_F$ is the maximal positive invariant set of the closed-loop LQR dynamics ($A-BF$). $P$ is the solution to the **Riccati Equation**.
8.  **Oscillations:** In a naive formulation, when tracking error is zero, the cost is dominated by the control effort term ($u^TRu$). The optimizer minimizes $u$ to zero, causing the system to deviate from the reference (e.g., plane drops), which increases the tracking error cost. The controller then adds thrust to correct the error, causing a cycle of oscillation.
9.  **Delta Controls:** By optimizing over $\Delta u$, a constant control input (like constant thrust) results in $\Delta u = 0$. The cost penalizes *changes* in control, not the absolute value. Thus, holding a constant thrust is "free" in terms of cost, preventing the controller from turning off necessary steady-state inputs.
10. **Explicit MPC Advantages:**
    *   **Validation:** The control law is explicit and easier to validate for safety-critical systems (no reliance on a complex solver's internal logic).
    *   **Reliability/Speed:** It avoids the risk of solver failure or long computation times online; it is simply a lookup table.

**Critical Thinking & Evaluation**
11. **Critique:** The lemma proves that *if* $X_1$ is controlled invariant, feasibility holds. However, $X_1$ is not a variable we can easily tune; it is derived from the problem. The theorem becomes useful only when we realize we can *force* this condition by choosing our terminal constraint $X_F$ to be a controlled invariant set. Without linking the abstract lemma to the concrete tuning knob ($X_F$), we don't know *how* to design the MPC.
12. **Choice:** **Explicit MPC** is generally preferred for safety-critical microcontrollers. The lecture notes that in safety-critical situations (like landing rockets), you must trust the optimizer. Explicit MPC removes the complex solver from the critical path, leaving only simple arithmetic (lookup and gain application), which is far easier to verify and validate for strict safety standards.
13. **Physical Intuition:** Assumption 3 requires that there exists a control $v$ such that the change in cost is negative. Physically, this means the system can always "do something" (apply control $v$) to reduce its "energy" (cost). If no such control exists, the system might get stuck in a state where it cannot dissipate energy, leading to instability or infeasibility. It formalizes the idea that the system is "drained" of energy over time.
