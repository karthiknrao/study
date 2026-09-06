Here is your comprehensive study guide for Week 3 of AA203, synthesized from the lecture transcript.

### 1. Executive Summary & Core Concepts

**Lecture Overview:**
This lecture extends the theory of indirect optimal control methods from unbounded to **bounded controls**, introducing the **Pontryagin Minimum Principle (PMP)**. The core thesis is that when control inputs are constrained (e.g., actuator limits), the optimality conditions shift from requiring a stationary point in the Hamiltonian to requiring a **global minimum**. We apply this principle to three archetype problems—minimum time, minimum fuel, and minimum energy—to derive specific control structures (such as bang-bang and saturating controls) before introducing numerical solvers like shooting and collocation methods for practical implementation.

**Key Concepts Highlight:**
*   **Bounded Controls:** In practical systems, controls cannot be infinite. We assume bounds (e.g., $|u(t)| \le 1$), which fundamentally change how we determine the optimal input.
*   **Pontryagin Minimum Principle (PMP):** The strengthened optimality condition for bounded controls. Instead of $\frac{\partial H}{\partial u} = 0$, we require $u^*$ to be the **global minimizer** of the Hamiltonian $H$ with respect to $u$.
*   **Hamiltonian ($H$):** The central function in indirect methods, defined as the stage cost plus the costates (co-states) dot the dynamics. It serves as the analog to the Lagrangian in finite-dimensional optimization.
*   **Costates ($p$):** The vector of "shadow" variables (analogous to Lagrange multipliers) that tracks the sensitivity of the cost to the state. One costate exists for every state variable.
*   **Bang-Bang Control:** A control profile that switches between the maximum positive and maximum negative bounds (e.g., full throttle vs. full brake). This is the result of minimizing time in linear systems.
*   **Singular Arcs:** Situations where the optimality conditions fail to characterize the optimal control (specifically when the derivative of $H$ with respect to $u$ is identically zero). These require more sophisticated analysis than standard cases.
*   **Shooting vs. Collocation Methods:** Two primary numerical approaches for solving the Two-Point Boundary Value Problem (TPBVP) arising from indirect methods. Shooting approximates the problem as an initial value problem; collocation approximates dynamics using basis functions at specific points.

---

### 2. Deep Dive: Expanded Lecture Notes

#### 1. The Shift from Unbounded to Bounded Optimality
*   **Detailed Explanation:**
    In the previous lecture, we assumed controls were unbounded. The optimality condition for the control was that the Hamiltonian had a **stationary point** ($\frac{\partial H}{\partial u} = 0$). However, in reality, actuators have limits (e.g., a motor cannot produce infinite torque). When we introduce bounds, the "stationary point" condition is no longer sufficient because the optimal value might lie on the boundary of the admissible set.
    To derive the new condition, we look at the finite-dimensional analogy. In unconstrained optimization, a local minimum requires the gradient to be zero. In constrained optimization (where the variable is bounded), a local minimum requires that the function value increases for any *admissible* direction of movement.
    Translating this to infinite-dimensional calculus of variations: If the optimal control $u^*$ lies strictly within the bounds, variations can be in both directions ($+\delta u$ and $-\delta u$), so $\delta J = 0$ holds. However, if $u^*$ lies **on the boundary**, we can only vary in one direction (inward). Therefore, the condition strengthens: The variation $\delta J$ must be $\ge 0$. This leads to the **Pontryagin Minimum Principle**: The optimal control $u^*$ must be the **global minimizer** of the Hamiltonian $H(x, u, p)$ with respect to $u$.
*   **Context & Nuance:**
    This is the critical bridge between theoretical calculus of variations and practical engineering. The "jump" from stationarity to global minimization is the mathematical formalization of "saturation." If the unconstrained optimum falls outside the physical limits, the actual control saturates at the limit.
*   **Analogy:**
    Imagine minimizing a bowl-shaped function (a parabola).
    *   *Unbounded:* You slide down to the very bottom of the bowl.
    *   *Bounded:* Imagine the bottom of the bowl is at $x=5$, but you are only allowed to exist in the range $x \in [-1, 1]$. The "stationary" point (5) is not allowed. The "global minimum" within your allowed range is at the edge, $x=1$. You stop at the wall.
*   **Key Takeaway:**
    When controls are bounded, you do not just solve $\frac{\partial H}{\partial u} = 0$; you must verify that $u^*$ is the global minimum of $H$ within the admissible bounds.

#### 2. Archetype 1: Minimum Time Problems (Bang-Bang Control)
*   **Detailed Explanation:**
    We consider a control-affine system $\dot{x} = Ax + Bu$ where we want to reach the origin from an arbitrary state $x_0$ in the shortest time. The cost functional is $J = \int_{t_0}^{t_f} 1 \, dt$ (minimizing time).
    The Hamiltonian is $H = 1 + p^T(Ax + Bu) = 1 + p^T A x + \sum p^T B_i u_i$.
    To minimize $H$ with respect to $u$, we look at the terms involving $u$. Since the controls $u_i$ are independent, we minimize each term $\sum p^T B_i u_i$ individually.
    *   If $p^T B_i > 0$, the linear function is minimized by making $u_i$ as small (negative) as possible $\rightarrow u_i = m_i^-$ (lower bound).
    *   If $p^T B_i < 0$, the linear function is minimized by making $u_i$ as large (positive) as possible $\rightarrow u_i = m_i^+$ (upper bound).
    *   If $p^T B_i = 0$, the term is zero regardless of $u_i$. This is a **Singular Arc**.
    Because the control always operates at the extreme bounds ($m_i^+$ or $m_i^-$), this is called **Bang-Bang Control**.
*   **Context & Nuance:**
    This makes physical sense: To get somewhere as fast as possible, you never "coast." You either push the gas to the max or hit the brakes to the max. The "Singular Arc" is a critical edge case where the optimality conditions provide no information about $u$, requiring higher-order derivatives or other methods to resolve.
*   **Analogy:**
    Driving a car on a straight track. To minimize the time to the finish line, you are either at 100% throttle or 100% braking. You never drive at 50% speed unless you are stuck in traffic (singular case, though rare in pure time-minimization).
*   **Key Takeaway:**
    For minimum time problems with linear dynamics and bounded controls, the optimal control is "Bang-Bang," switching between the maximum positive and maximum negative actuator limits.

#### 3. Archetype 2: Minimum Fuel Problems
*   **Detailed Explanation:**
    Here, the objective is to minimize the effort (fuel) used, often represented by the $L_1$ norm (absolute value) of the control: $J = \int \sum C_i |u_i| dt$. The time is fixed.
    The Hamiltonian includes the stage cost $\sum C_i |u_i|$.
    Minimizing $H$ with respect to $u_i$ involves minimizing terms like $C_i |u_i| + p^T B_i u_i$.
    *   If $p^T B_i > C_i$: The linear term dominates. To minimize, we pick the most negative $u_i$ allowed, which is $u_i = m_i^-$.
    *   If $p^T B_i < -C_i$: We pick the most positive $u_i$, which is $u_i = m_i^+$.
    *   If $|p^T B_i| < C_i$: The cost of using control ($C_i |u_i|$) outweighs the benefit of changing the state. The optimal control is $u_i = 0$ (coasting).
    This results in a control profile that is either at the max/min bounds or exactly zero.
*   **Context & Nuance:**
    This contrasts with Minimum Time. In Minimum Time, you never coast. In Minimum Fuel, you *want* to coast as much as possible to save fuel, only using thrust when the "price" of thrust (coefficient $C_i$) is less than the "benefit" of changing the trajectory ($p^T B_i$).
*   **Analogy:**
    A spacecraft moving through space. To save fuel, it applies a small burn to change its trajectory, then turns off engines and coasts. It does not keep the engines at 50% power; it is either off or on full power (bang-bang in terms of on/off, but with a "dead zone" where it turns off).
*   **Key Takeaway:**
    Minimum fuel controls exhibit a "dead zone" where $u=0$ if the cost of actuation is high relative to the state sensitivity.

#### 4. Archetype 3: Minimum Energy Problems
*   **Detailed Explanation:**
    This is the case we started with, involving quadratic costs ($u^2$). The cost is $J = \int u^2 dt$.
    The Hamiltonian term is $\frac{1}{2}u^2 + p^T B u$.
    To find the global minimum, we take the derivative with respect to $u$ and set it to zero:
    $\frac{\partial}{\partial u} (\frac{1}{2}u^2 + p^T B u) = u + p^T B = 0 \implies u = -p^T B$.
    This is the unconstrained optimum. However, we must respect bounds $|u| \le 1$.
    *   If $-p^T B$ is within $[-1, 1]$, then $u^* = -p^T B$ (linear region).
    *   If $-p^T B > 1$, then $u^* = 1$ (saturation).
    *   If $-p^T B < -1$, then $u^* = -1$ (saturation).
    This creates a **Saturating Linear Control** profile.
*   **Context & Nuance:**
    This is the most common problem in robotics and stabilization because quadratic costs penalize large errors more heavily and are smooth. The "linear" part means the control is proportional to the costate, acting like a linear feedback law until the actuator hits its limit.
*   **Analogy:**
    A thermostat. If the room is slightly too cold, it turns the heater up a little (linear). If the room is freezing, it turns the heater to max (saturation).
*   **Key Takeaway:**
    For minimum energy problems, the optimal control is linear in the costates ($u = -p^T B$) but saturates at the physical limits.

#### 5. Computational Methods: Solving the TPBVP
*   **Detailed Explanation:**
    Indirect methods result in a Two-Point Boundary Value Problem (TPBVP): We have differential equations for $x$ and $p$, with initial conditions for $x(t_0)$ and boundary conditions involving $x(t_f)$ or $p(t_f)$.
    *   **Shooting Methods:** Treat the problem as an Initial Value Problem (IVP). You guess the initial costates $p(t_0)$, integrate forward, and check if the final boundary conditions are met. If not, you iteratively adjust the guess.
    *   **Collocation Methods:** Approximate the solution using basis functions (like polynomials) at specific "collocation points." This converts the differential equations into a large system of algebraic equations.
    *   **SciPy `solve_bvp`:** A practical implementation of collocation/shooting hybrids. It requires defining the dynamics function, the boundary condition residuals (which must equal zero), a time mesh, and an initial guess for the state/costate vector $Z$.
*   **Context & Nuance:**
    The lecture demonstrated `solve_bvp` from SciPy. The key is defining the residual for boundary conditions. For example, if $x(4) = -2$, the residual function is $x(4) + 2 = 0$.
*   **Analogy:**
    *   *Shooting:* Like throwing a dart. You throw it, see where it hits, and adjust your aim (guess) for the next throw until it hits the bullseye.
    *   *Collocation:* Like fitting a curve. You draw a curve through specific points and adjust the curve's shape until it fits the constraints at those points.
*   **Key Takeaway:**
    Solving indirect methods requires numerical solvers for TPBVPs. `solve_bvp` in SciPy is a standard tool, requiring you to define dynamics, boundary residuals, and time grids.

---

### 3. Pathways for Further Exploration

1.  **The Topic/Concept:** Singular Arcs and Switching Functions
    *   **Why it Matters:** The lecture noted that when $p^T B_i = 0$, the optimality conditions fail to define $u$. This is a critical gap in indirect methods for complex systems.
    *   **Search/Study Direction:** Look into "Singular Arcs in Optimal Control" and "Switching Functions." Study how to determine the duration of a singular arc by differentiating the switching function with respect to time until it is no longer zero.

2.  **The Topic/Concept:** Direct Methods vs. Indirect Methods
    *   **Why it Matters:** The lecture contrasts "optimize then discretize" (indirect) with "discretize then optimize" (direct). Next week covers Direct Methods.
    *   **Search/Study Direction:** Research "Direct Methods for Optimal Control," specifically "Multiple Shooting" and "Collocation-based Direct Methods." Understand why Direct Methods are often more robust for non-linear systems and constraints.

3.  **The Topic/Concept:** State Constraints vs. Control Constraints
    *   **Why it Matters:** The professor mentioned that indirect methods are poor for handling state constraints (like obstacle avoidance).
    *   **Search/Study Direction:** Investigate "Penalty Function Methods" or "Barrier Functions" used in Direct Methods to handle state constraints, contrasting them with the Lagrange multiplier approach used for control bounds in indirect methods.

4.  **The Topic/Concept:** Numerical Solvers for TPBVPs
    *   **Why it Matters:** We only scratched the surface of `solve_bvp`. Understanding the convergence criteria is vital for implementation.
    *   **Search/Study Direction:** Study the "Collocation Method" in depth. Look for resources on "Gauss-Legendre collocation" and how it ensures high accuracy by enforcing orthogonality conditions.

5.  **The Topic/Concept:** Pontryagin Maximum Principle (PMP) Variants
    *   **Why it Matters:** We used the Minimum Principle. There is also a Maximum Principle used in certain economic or resource-extraction models.
    *   **Search/Study Direction:** Review the "Maximum Principle" vs. "Minimum Principle." Understand that the sign convention depends on whether you are maximizing profit or minimizing cost.

6.  **The Topic/Concept:** LQR and LQG Control
    *   **Why it Matters:** The "Minimum Energy" problem with quadratic costs is the foundation of Linear Quadratic Regulator (LQR) control.
    *   **Search/Study Direction:** Connect the indirect derivation of $u = -p^T B$ to the algebraic solution of LQR ($u = -Kx$). Understand that LQR is the finite-horizon, linear-system special case of the Minimum Energy optimal control problem.

---

### 4. Comprehension & Review Questions

**Recall & Understanding**
1.  What is the primary mathematical difference between the optimality condition for unbounded controls and the Pontryagin Minimum Principle for bounded controls?
2.  Define the "Costate" ($p$) and explain its relationship to the state variables ($x$).
3.  In the context of Minimum Time problems, what is the specific structure of the optimal control known as "Bang-Bang" control?
4.  What is a "Singular Arc," and why is it problematic for standard optimality conditions?
5.  When using a numerical solver like `solve_bvp`, what are the three main components (arguments) you must define?

**Application & Analysis**
6.  Consider a Minimum Energy problem where the unconstrained optimal control is $u = -p^T B$. If $p^T B = 1.5$ and the control bound is $|u| \le 1$, what is the actual optimal control value?
7.  In a Minimum Fuel problem, the term $p^T B_i$ represents the "benefit" of control. If $|p^T B_i| < C_i$ (where $C_i$ is the fuel cost coefficient), what is the optimal control $u_i$?
8.  Analyze the Hamiltonian for a Minimum Time problem. Why does the term $1$ (the stage cost) not affect the optimization of $u$ directly, but still influence the system's trajectory?
9.  If you were to use a Shooting Method to solve a TPBVP, what variable would you typically "guess" to initiate the integration?
10.  Compare the control profiles of Minimum Time vs. Minimum Fuel. Which one involves "coasting" (zero control) and which one does not?

**Critical Thinking & Evaluation**
11.  The lecture states that indirect methods are not the "tool of choice" for handling state constraints (like obstacle avoidance). Based on the complexity of the optimality conditions, argue why Direct Methods might be preferred for problems with complex state constraints.
12.  Critique the "Bang-Bang" control solution for a Minimum Time problem. While mathematically optimal, what are the practical physical drawbacks (e.g., chattering, energy) that might make a Minimum Energy solution preferable in real-world robotics?
13.  In the derivation of the PMP, we moved from $\delta J = 0$ (unconstrained) to $\delta J \ge 0$ (constrained). Explain why this inequality is necessary when the control is at the boundary, referencing the "one-sided" nature of admissible variations.

---

### Answer Key & Explanations

**1. Primary Mathematical Difference:**
The unbounded condition requires the Hamiltonian to have a **stationary point** ($\frac{\partial H}{\partial u} = 0$). The PMP requires $u^*$ to be the **global minimizer** of the Hamiltonian with respect to $u$ within the admissible bounds.

**2. Definition of Costate:**
The costate $p(t)$ is a vector of functions (one for each state variable) that acts as the analog to Lagrange multipliers. It represents the sensitivity of the cost function to changes in the state.

**3. Bang-Bang Control Structure:**
The control switches between the maximum positive bound ($m_i^+$) and the maximum negative bound ($m_i^-$). It does not take intermediate values (unless in a singular arc, which is often ignored or treated separately).

**4. Singular Arc:**
A singular arc is a portion of the trajectory where the derivative of the Hamiltonian with respect to the control is identically zero (e.g., $p^T B = 0$). In these cases, the first-order optimality conditions provide no information about the value of $u$, requiring higher-order analysis.

**5. Components of `solve_bvp`:**
1.  **fun:** The function defining the differential equations (dynamics).
2.  **bc:** The function defining the boundary condition residuals (must equal zero).
3.  **t_eval/t_mesh:** The time grid/mesh.
4.  **z_guess:** The initial guess for the aggregated state/costate vector.

**6. Minimum Energy Control Value:**
The unconstrained optimum is $u = -1.5$. Since the bound is $|u| \le 1$, the control saturates. The actual optimal control is $u = -1$ (clipped to the lower bound).

**7. Minimum Fuel Control Value:**
If the cost of actuation ($C_i$) is higher than the benefit ($|p^T B_i|$), the optimal control is $u_i = 0$. The system "coasts" to save fuel.

**8. Role of the Stage Cost '1':**
The stage cost $1$ is constant and does not depend on $u$. Therefore, when minimizing $H$ with respect to $u$, this term vanishes. However, it affects the *value* of the Hamiltonian and the boundary conditions (specifically the transversality conditions involving $H$ at $t_f$), ensuring the time minimization is encoded in the system's energy/conservation laws.

**9. Shooting Method Guess:**
You typically guess the **initial costates** $p(t_0)$. You integrate the system forward with this guess and check if the final boundary conditions (e.g., $x(t_f)$) are satisfied.

**10. Comparison of Profiles:**
*   **Minimum Time:** No coasting. Always at max/min bounds (Bang-Bang).
*   **Minimum Fuel:** Includes a "dead zone" where $u=0$ (coasting) if the cost of thrust is high relative to the state sensitivity.

**11. Indirect vs. Direct for State Constraints:**
Indirect methods rely on smooth optimality conditions. State constraints (like obstacles) create non-smoothities or complex boundary layers that are difficult to handle with simple Lagrange multipliers in the indirect framework. Direct methods discretize the problem into a large finite-dimensional optimization problem (NLP), where state constraints can be handled directly as inequality constraints at each time step, making them more robust for complex geometric constraints.

**12. Critique of Bang-Bang:**
While Bang-Bang is fastest, it involves high-frequency switching (chattering) at the boundaries. This can cause wear on actuators, generate heat, and excite unmodeled high-frequency dynamics. Minimum Energy solutions are smoother and safer for physical hardware, trading a small amount of time for actuator longevity.

**13. Why $\delta J \ge 0$ is Necessary:**
In the unconstrained case, we can move in *any* direction, so the gradient must be zero to ensure we aren't moving "downhill" in any direction. In the bounded case, if we are at the boundary, we can only move *inward*. Therefore, we only require that moving inward increases the cost ($\delta J \ge 0$). We do not require the gradient to be zero because the "downhill" direction is blocked by the physical limit.
