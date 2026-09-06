### 1. Executive Summary & Core Concepts

**Lecture Overview:**
This lecture transitions from identifying single system failures to modeling the **full distribution of possible failures**. The core objective is to teach how to sample from this "failure distribution" when the normalizing constant (the total probability mass of failures) is difficult or impossible to compute analytically. The lecture introduces two primary algorithms for sampling from unnormalized densities: **Rejection Sampling** and **Markov Chain Monte Carlo (MCMC)**. It highlights the practical challenges of these methods, such as inefficient sampling for rare events (in Rejection Sampling) and the difficulty of exploring multiple distinct failure modes (in MCMC), and introduces techniques like "smoothing" to mitigate these issues.

**Key Concepts Highlight:**
*   **Failure Distribution:** The conditional probability distribution of trajectories given that they are failures. It is mathematically defined as the nominal trajectory distribution ($P(\tau)$) conditioned on the trajectory being a failure ($\tau \notin \psi$).
*   **Unnormalized Probability Density:** A function that has the correct *shape* of a target probability distribution but does not integrate to 1. It is denoted as $\bar{P}(\tau)$ and is crucial because we can often compute it even when we cannot compute the full normalized distribution.
*   **Rejection Sampling:** An algorithm that samples from a "proposal distribution" ($Q(\tau)$) and accepts/rejects samples based on a ratio involving the unnormalized target density. It requires a proposal distribution that can be easily sampled from and a scaling constant $C$ such that the proposal covers the target.
*   **Proposal Distribution ($Q(\tau)$):** A probability distribution from which we can easily draw samples. In Rejection Sampling, this acts as the "dartboard" or the source of candidate samples.
*   **Markov Chain Monte Carlo (MCMC):** A class of algorithms that generate a sequence of samples (a chain) where each new sample depends on the previous one. Unlike Rejection Sampling, MCMC does not require a separate proposal distribution to cover the entire space; it uses a "kernel" to propose local moves.
*   **Kernel Function ($G$):** The conditional distribution used in MCMC to propose a new sample based on the current sample. A common choice is a symmetric Gaussian distribution centered at the current state.
*   **Burn-in:** The initial phase of an MCMC chain where samples are not yet representative of the target distribution because the chain has not converged from its initial state. These samples are typically discarded.
*   **Smoothing:** A technique used to modify the unnormalized failure density by adding a small variance to the "distance to failure," allowing the sampler to move between distinct failure modes (e.g., left-tail and right-tail failures) that would otherwise be isolated by zero-probability regions.

---

### 2. Deep Dive: Expanded Lecture Notes (The "Teaching" Section)

#### 1. The Failure Distribution & Unnormalized Densities

*   **Detailed Explanation:**
    In previous lectures, we focused on finding *a* failure. Now, we want the *distribution* of failures. Let $\tau$ be a trajectory. The nominal trajectory distribution is $P(\tau)$. The failure distribution is $P(\tau | \tau \text{ is a failure})$.
    Mathematically, this is:
    $$ P(\tau | \text{failure}) = \frac{P(\tau) \cdot \mathbb{1}_{\tau \notin \psi}}{\int P(\tau') \cdot \mathbb{1}_{\tau' \notin \psi} d\tau'} $$
    The numerator is $P(\tau)$ if $\tau$ is a failure, and 0 otherwise. The denominator is an integral over all trajectories, representing the total probability of failure. This denominator is often computationally intractable because it requires evaluating every possible trajectory.
    However, the **numerator** is easy to compute: we just check if $\tau$ is a failure (using the indicator function) and multiply by $P(\tau)$. This numerator is called the **unnormalized probability density**, denoted $\bar{P}(\tau)$. It has the correct shape of the distribution but lacks the scaling factor (normalizing constant).
*   **Context & Nuance:**
    This connects to Bayesian estimation concepts discussed earlier. We often cannot compute the full posterior (normalized), but we can compute the likelihood (unnormalized). Algorithms like Rejection Sampling and MCMC are designed specifically to draw samples from a distribution given only its unnormalized density.
*   **Analogy or Real-World Example:**
    Imagine you know the *shape* of a mountain range (the unnormalized density) but you don't know the total *volume* of the mountain range (the normalizing constant). You can still map the peaks and valleys (the shape) even if you can't calculate the exact total area under the curve.
*   **Key Takeaway:** We can always compute the unnormalized failure density ($\bar{P}(\tau)$), and this is sufficient to apply sampling algorithms to generate samples from the true failure distribution.

#### 2. Rejection Sampling

*   **Detailed Explanation:**
    Rejection sampling is a "generate and test" method.
    1.  Define a **proposal distribution** $Q(\tau)$ from which we can easily sample.
    2.  Choose a constant $C$ such that $\bar{P}(\tau) \leq C \cdot Q(\tau)$ for all $\tau$. This means the proposal distribution (scaled by $C$) must "cover" or "bound" the target unnormalized density.
    3.  **Algorithm:**
        *   Sample $\tau \sim Q(\tau)$.
        *   Sample $r \sim \text{Uniform}(0, 1)$.
        *   If $r < \frac{\bar{P}(\tau)}{C \cdot Q(\tau)}$, accept $\tau$; otherwise, reject it.
    The accepted samples are distributed according to the target distribution.
*   **Context & Nuance:**
    The "Dartboard Analogy": Imagine a rectangular board. $Q(\tau)$ is the horizontal axis. The height of the board is determined by $C \cdot Q(\tau)$. We throw darts uniformly. If a dart lands *under* the curve $\bar{P}(\tau)$, we accept it. If it lands *above* the curve (in the empty space), we reject it.
    **Efficiency Issue:** If failures are rare, most samples from $Q(\tau)$ will be successes. We reject almost everything, making the process highly inefficient.
*   **Analogy or Real-World Example:**
    Think of a fishing net. If you use a net that is too big for the fish you want (a uniform distribution over a large space), you catch lots of water (rejected samples). If you use a net that is precisely sized to the fish population (a good proposal distribution), you catch more fish with less effort.
*   **Key Takeaway:** Rejection sampling is simple but inefficient for rare events; it requires a proposal distribution that closely matches the target to minimize wasted samples.

#### 3. Markov Chain Monte Carlo (MCMC)

*   **Detailed Explanation:**
    MCMC does not use a global proposal distribution. Instead, it builds a **chain** of samples.
    1.  Start with an initial sample $\tau$.
    2.  Propose a new sample $\tau'$ using a **kernel** $G(\tau' | \tau)$. A common kernel is a Gaussian distribution centered at $\tau$.
    3.  Accept $\tau'$ with probability:
        $$ \text{Prob} = \min\left(1, \frac{\bar{P}(\tau') G(\tau | \tau')}{\bar{P}(\tau) G(\tau' | \tau)}\right) $$
    If the kernel is **symmetric** ($G(\tau|\tau') = G(\tau'|\tau)$), this simplifies to:
        $$ \text{Prob} = \min\left(1, \frac{\bar{P}(\tau')}{\bar{P}(\tau)}\right) $$
    *   If $\bar{P}(\tau') > \bar{P}(\tau)$, accept with probability 1 (move to higher density).
    *   If $\bar{P}(\tau') < \bar{P}(\tau)$, accept with probability proportional to the ratio (move to lower density with some chance to escape local maxima).
*   **Context & Nuance:**
    MCMC is guaranteed to converge to the target distribution in the limit of infinite samples. However, in practice, we use finite samples. This leads to **correlated samples** (the chain moves slowly).
*   **Analogy or Real-World Example:**
    Imagine a blind person exploring a dark room with a flashlight that only shows a small circle around them. They take small steps (kernel). If the ground ahead is higher (higher probability), they step forward. If the ground is lower, they might step back, but occasionally they take a risk and step down. Over time, they map the room.
*   **Key Takeaway:** MCMC uses local moves (kernel) to explore the distribution. It is powerful but can get "stuck" if the distribution has multiple separated peaks (modes) and the kernel is too small to jump between them.

#### 4. Practical Challenges: Burn-in, Thinning, and Smoothing

*   **Detailed Explanation:**
    *   **Burn-in:** The initial samples in an MCMC chain are not representative of the target distribution because the chain hasn't converged from its arbitrary start. We discard the first $N$ samples (burn-in period).
    *   **Thinning:** Because consecutive samples in a chain are correlated, we might only keep every $k$-th sample (e.g., every 10th) to get independent samples for statistical analysis.
    *   **Smoothing:** If the failure distribution has multiple distinct modes (e.g., failure at $x < -1$ AND failure at $x > 1$), the probability in the middle is zero. A small kernel cannot jump across the zero-probability gap.
    *   **Solution:** Define a distance function $\delta(\tau)$ (e.g., robustness clipped at 0). Replace the hard indicator function $\mathbb{1}_{\tau \notin \psi}$ with a smooth function, like a Normal distribution $\mathcal{N}(0, \epsilon)$ applied to $\delta(\tau)$. This gives non-zero probability to "near-failure" states, creating a bridge between modes.
*   **Context & Nuance:**
    Smoothing is essentially a form of **Rejection Sampling** where the smoothed distribution is the proposal. We sample from the smoothed distribution, then reject samples that are not actual failures, leaving us with samples from the true failure distribution.
*   **Analogy or Real-World Example:**
    Imagine two islands (failure modes) separated by an ocean (zero probability). A boat (kernel) that only moves 1 meter cannot cross the ocean. Smoothing is like building a narrow bridge (non-zero probability) across the ocean, allowing the boat to cross, after which we only count the trips that land on the islands.
*   **Key Takeaway:** MCMC requires careful tuning. If modes are separated, use smoothing to create a "bridge" of non-zero probability so the chain can explore all failure modes.

---

### 3. Pathways for Further Exploration

1.  **The Topic/Concept:** **Metropolis-Hastings Algorithm**
    *   **Why it Matters:** The lecture described a specific case of MCMC using a symmetric kernel. Metropolis-Hastings is the general framework for MCMC that allows for *asymmetric* kernels, which is crucial for high-dimensional or complex systems.
    *   **Search/Study Direction:** Look into the mathematical proof for why the Metropolis acceptance ratio works for non-symmetric kernels. Study how this differs from the simplified symmetric case presented in the lecture.

2.  **The Topic/Concept:** **Multimodal Distributions and Mode-Seeking**
    *   **Why it Matters:** The lecture highlighted that standard MCMC can get stuck in one mode. Understanding "mode-seeking" is critical for complex systems with multiple failure types.
    *   **Search/Study Direction:** Explore "Parallel Tempering" or "MCMC with Adaptive Kernels" as advanced techniques to help chains jump between distant modes more effectively than simple smoothing.

3.  **The Topic/Concept:** **Efficiency of Rejection Sampling**
    *   **Why it Matters:** The lecture noted Rejection Sampling is inefficient for rare events. Quantifying this inefficiency is important for choosing the right algorithm.
    *   **Search/Study Direction:** Study the "Acceptance Rate" in Rejection Sampling. Derive the relationship between the probability of failure $P(\text{fail})$ and the expected number of samples required to get one valid failure sample.

4.  **The Topic/Concept:** **Distance Functions and Robustness**
    *   **Why it Matters:** The smoothing technique relies on a "distance to failure" metric. The quality of this metric determines the quality of the smoothing.
    *   **Search/Study Direction:** Investigate different metrics for "distance to failure" (e.g., minimum robustness, margin to constraint violation). How does the choice of metric affect the "bridge" created in the smoothed distribution?

5.  **The Topic/Concept:** **Convergence Diagnostics in MCMC**
    *   **Why it Matters:** The lecture mentioned "burn-in" but didn't detail how to *know* when burn-in is over. In practice, determining convergence is a major challenge.
    *   **Search/Study Direction:** Look into "Gelman-Rubin statistics" or "Autocorrelation time" estimation to determine how many samples to discard as burn-in and how long the chain needs to run.

6.  **The Topic/Concept:** **Hamiltonian Monte Carlo (HMC)**
    *   **Why it Matters:** The lecture used a simple Gaussian kernel. HMC is a more advanced MCMC method that uses physics-inspired momentum to move through high-dimensional spaces more efficiently.
    *   **Search/Study Direction:** Compare the "local" moves of the Gaussian kernel in the lecture to the "global" moves possible in HMC. Why is HMC often preferred in high-dimensional Bayesian statistics?

---

### 4. Comprehension & Review Questions

**Recall & Understanding**
1.  Define the "failure distribution" in terms of the nominal trajectory distribution $P(\tau)$ and the indicator function.
2.  What is the primary computational advantage of working with an "unnormalized probability density" ($\bar{P}(\tau)$) rather than the fully normalized distribution?
3.  In Rejection Sampling, what is the role of the constant $C$, and what constraint must it satisfy relative to $Q(\tau)$ and $\bar{P}(\tau)$?
4.  What is a "kernel" in the context of MCMC, and what is a common example of a symmetric kernel?
5.  Why is the "burn-in" period necessary in MCMC, and what do we do with samples generated during this period?

**Application & Analysis**
6.  Suppose you are using Rejection Sampling for a system where failures are extremely rare (e.g., $P(\text{failure}) = 0.001$). Describe the expected behavior of the algorithm in terms of sample efficiency.
7.  In the MCMC algorithm, if a proposed sample $\tau'$ has a lower unnormalized density than the current sample $\tau$ ($\bar{P}(\tau') < \bar{P}(\tau)$), how is the acceptance probability determined?
8.  Consider a system with two distinct failure modes (e.g., $x < -5$ and $x > 5$) separated by a region of high probability for successes. Why would a standard MCMC with a small-variance Gaussian kernel fail to sample both modes effectively?
9.  How does "smoothing" the failure distribution help the MCMC chain move between distinct failure modes? Specifically, what does it do to the probability in the "gap" between modes?
10.  If you use a symmetric Gaussian kernel for MCMC, how does the acceptance probability simplify? What does this imply about the algorithm's tendency to move toward higher or lower density regions?

**Critical Thinking & Evaluation**
11.  The lecture states that Rejection Sampling reduces to "direct sampling" when $Q(\tau) = P(\tau)$. Critique this method: Why is it mathematically valid, but practically suboptimal for rare failure events?
12.  Evaluate the trade-offs between Rejection Sampling and MCMC. Under what specific conditions would you prefer Rejection Sampling over MCMC, and vice versa?
13.  The lecture introduces "smoothing" using a Normal distribution with variance $\epsilon$. Discuss the risks of choosing $\epsilon$ to be too large versus too small in the context of exploring a multimodal failure distribution.

---

**Answer Key & Explanations**

**Recall & Understanding**
1.  **Answer:** The failure distribution is $P(\tau | \tau \notin \psi) = \frac{P(\tau) \cdot \mathbb{1}_{\tau \notin \psi}}{\text{Normalizing Constant}}$. It is the nominal distribution weighted by the indicator of failure.
2.  **Answer:** The unnormalized density $\bar{P}(\tau)$ is easy to compute (just check if it's a failure and evaluate $P(\tau)$). We avoid the difficult integral (the normalizing constant) which requires evaluating all possible trajectories.
3.  **Answer:** $C$ is a scaling constant. It must satisfy $C \cdot Q(\tau) \geq \bar{P}(\tau)$ for all $\tau$. It ensures the proposal distribution (scaled) fully covers the target density.
4.  **Answer:** A kernel $G(\tau'|\tau)$ is the conditional distribution used to propose a new sample based on the current one. A common symmetric kernel is a Gaussian distribution centered at the current sample $\tau$.
5.  **Answer:** Burn-in is the initial phase where samples are not representative of the target distribution because the chain hasn't converged from its arbitrary start. We discard these samples.

**Application & Analysis**
6.  **Answer:** The algorithm will reject ~99.9% of samples. It will be highly inefficient, requiring thousands of rollouts to find a single valid failure sample.
7.  **Answer:** The acceptance probability is the ratio $\frac{\bar{P}(\tau')}{\bar{P}(\tau)}$ (assuming symmetric kernel). We sample $r \sim \text{Uniform}(0,1)$ and accept if $r < \frac{\bar{P}(\tau')}{\bar{P}(\tau)}$.
8.  **Answer:** Because the kernel only proposes local moves, and the probability of being in the "gap" between modes is zero (or very low), the chain has no "bridge" to cross. It will get stuck in one mode and never reach the other.
9.  **Answer:** Smoothing replaces the hard indicator (0 or 1) with a smooth function (e.g., a Normal distribution). This assigns non-zero probability to the "gap" between modes, creating a "bridge" that allows the chain to move from one failure mode to another.
10. **Answer:** It simplifies to $\min(1, \frac{\bar{P}(\tau')}{\bar{P}(\tau)})$. This implies we always move to higher density regions, and move to lower density regions only with a probability proportional to how much lower they are.

**Critical Thinking & Evaluation**
11. **Answer:** It is valid because the ratio of probabilities cancels out, leaving the indicator function. However, it is suboptimal because it wastes computational resources generating samples from the nominal distribution that are guaranteed to be rejected if they are not failures.
12. **Answer:** Rejection Sampling is preferred when the proposal distribution is easy to define and covers the target well (high efficiency). MCMC is preferred when the target is complex, multimodal, or when a good global proposal distribution is hard to design.
13. **Answer:** If $\epsilon$ is too small, the bridge is too narrow, and the chain may still get stuck. If $\epsilon$ is too large, the "bridge" becomes too wide, and the chain may wander into regions of very low probability, wasting samples on "near-failures" that are not actual failures, reducing the efficiency of the final rejection step.
