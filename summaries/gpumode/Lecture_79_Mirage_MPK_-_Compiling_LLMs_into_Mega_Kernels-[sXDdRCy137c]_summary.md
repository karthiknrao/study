Here is your comprehensive study guide for the **Mirage** project (Multi-level Super Optimizer) and the **MPK** (Mirage Persistent Kernel) extension, based on the lecture transcripts provided.

---

### 1. Executive Summary & Core Concepts

**Lecture Overview:**
This lecture introduces **Mirage**, a multi-level super-optimizer developed at CMU that automates the generation of highly efficient CUDA kernels for machine learning workloads. Unlike traditional compilers that rely on fixed transformation rules or manual kernel writing, Mirage uses an exhaustive search approach guided by abstract expressions to explore a massive space of possible kernel configurations across the entire GPU memory hierarchy. The second part of the lecture details **MPK**, a system that extends this concept to compile entire Large Language Models (LLMs) into a single "mega-kernel," eliminating kernel launch overhead and enabling fine-grained synchronization and dynamic workload management.

**Key Concepts Highlight:**
*   **Mile Graph (IR):** A multi-level intermediate representation that mirrors the GPU hardware hierarchy. It consists of nested graphs: a *Kernel Graph* (device memory/SMs), a *Thread Block Graph* (shared memory/SMs), and a *Thread Graph* (register file/individual threads).
*   **Exhaustive Search vs. Transformation Rules:** Traditional compilers use sparse, predefined transformation rules. Mirage uses a dense, exhaustive search to generate *all* possible valid graphs up to a bounded size, allowing it to discover custom, non-standard optimizations that rule-based systems miss.
*   **Abstract Expression Pruning:** A technique to make exhaustive search feasible. Instead of reasoning about full tensor algebra (which is expensive), Mirage abstracts index details into scalar-like expressions. This allows an automated theorem prover to quickly prune search branches that are mathematically impossible or redundant.
*   **Probabilistic Equivalence Verification:** A correctness checking mechanism using random testing in finite fields. It verifies that a generated graph produces the same output as the input program with high probability, avoiding the need for complex symbolic proofs.
*   **Task Graph:** The core abstraction in MPK for the mega-kernel. It interleaves *Tasks* (units of work on a single SM, e.g., a matrix tile) and *Events* (synchronization points). This replaces static kernel barriers with dynamic, fine-grained dependencies.
*   **In-Kernel Runtime System:** A dynamic scheduling mechanism running *inside* the GPU kernel. It uses specific "scheduler" warps to manage task queues and event triggers, allowing the system to react to dynamic workloads (like varying batch sizes) without CPU intervention.
*   **Mega-Kernel Approach:** Compiling an entire LLM inference step into a single CUDA kernel. This removes inter-kernel barriers, allows overlapping computation and communication (e.g., AllReduce), and enables better load balancing across Streaming Multiprocessors (SMs).

---

### 2. Deep Dive: Expanded Lecture Notes

#### 1. The Mile Graph (Multi-Level IR)
*   **Detailed Explanation:** The Mile Graph is the structural backbone of Mirage. It is not a single graph but a hierarchy of graphs that map directly to hardware resources.
    *   **Kernel Graph:** Represents computation across all SMs on the GPU. Vertices are operators (e.g., MatMul, RMSNorm); edges represent data flowing through **Device Memory** (HBM).
    *   **Thread Block Graph:** Represents computation within a single SM. Vertices are operations executed by a thread block; edges represent data in **Shared Memory** (L1 Cache/L2).
    *   **Thread Graph:** Represents computation by a single thread. Vertices are scalar operations; edges represent data in **Register Files**.
    *   **Graph-Defined Operators:** Special operators where the "computation" is not a single instruction but a lower-level graph (e.g., a "Tiled MatMul" is defined by a Thread Block Graph).
*   **Context & Nuance:** This structure allows Mirage to optimize at multiple levels simultaneously. A standard compiler might optimize a kernel but fail to optimize how data moves between shared memory and registers. The Mile Graph forces the optimizer to consider these hierarchies explicitly.
*   **Analogy:** Think of it like a city map. The Kernel Graph is the map of the whole city (neighborhoods/SMs). The Thread Block Graph is the map of a single neighborhood (streets/Shared Memory). The Thread Graph is the map of a single house (rooms/Registers). To optimize traffic, you need to understand all three levels, not just the city layout.
*   **Key Takeaway:** The Mile Graph decouples the logical computation from the physical hardware mapping, allowing the optimizer to explore how data moves through different memory tiers.

#### 2. Exhaustive Search & Abstract Expression Pruning
*   **Detailed Explanation:** Generating all possible kernels is computationally impossible without pruning. Mirage uses **Abstract Expressions** to prune the search space.
    *   **Abstraction:** Instead of tracking exact indices (e.g., $C_{ij} = \sum A_{ik}B_{kj}$), it abstracts to scalar counts (e.g., "Sum of 64 elements").
    *   **Pruning Logic:** The system derives a "desired expression" from the input. During search, if a partial graph generates an expression that *cannot* be a sub-expression of the desired one (based on algebraic axioms like commutativity and associativity), that branch is pruned.
    *   **Solver Integration:** This logic is handled by an automated theorem prover using First-Order Logic axioms.
*   **Context & Nuance:** This is the critical innovation that makes "super-optimization" feasible. Without this, the search space is infinite. With it, the search remains tractable even for complex operations like Attention.
*   **Analogy:** Imagine searching for a specific word in a dictionary. Instead of checking every letter combination (exhaustive), you use the prefix "App..." to instantly discard words starting with "B" or "C". Abstract expressions act as that prefix filter, discarding mathematically impossible structures early.
*   **Key Takeaway:** Abstract Expression Pruning is the "brake" that prevents the "engine" of exhaustive search from overheating, ensuring the system completes in minutes rather than years.

#### 3. Probabilistic Equivalence Verification
*   **Detailed Explanation:** After generating candidate graphs, Mirage must ensure they are correct. It uses **random testing in finite fields**.
    *   It generates random input tensors.
    *   It runs the original program and the new candidate graph.
    *   It compares outputs.
    *   **Finite Fields:** This avoids floating-point precision errors (where $1.0 + 1.0$ might not equal $2.0$ due to rounding), ensuring that if the math is wrong, the outputs *will* differ.
    *   **Guarantee:** There is a theoretical probability bound for false positives. If a graph is equivalent, outputs always match. If not equivalent, they match only with a small, calculable probability.
*   **Context & Nuance:** This is a pragmatic trade-off. Formal symbolic proof is too slow. Random testing is fast but usually unreliable for floating-point math. Finite-field random testing bridges this gap, providing high confidence with low overhead.
*   **Analogy:** Instead of proving a bridge won't collapse by calculating every atom's stress (impossible), you drop a known weight on it. If it doesn't break, it’s probably safe. Finite fields ensure the "weight" (numbers) behaves predictably so the test is valid.
*   **Key Takeaway:** Mirage uses probabilistic correctness checking (random testing in finite fields) to verify kernels, balancing speed and reliability.

#### 4. The MPK Task Graph & In-Kernel Runtime
*   **Detailed Explanation:** For the mega-kernel, the compiler generates a **Task Graph**.
    *   **Tasks:** The smallest unit of work (e.g., one tile of a matrix multiplication).
    *   **Events:** Synchronization primitives. An event is "triggered" when all tasks it depends on finish.
    *   **Runtime:** Inside the single CUDA kernel, specific **Scheduler Warps** (1-2% of SMs) monitor event queues. When an event triggers, the scheduler dispatches dependent tasks to **Worker SMs**.
    *   **Dynamic Handling:** Because the scheduler runs on the GPU, it can react to dynamic data (e.g., a shorter sequence finishes early) without returning to the CPU.
*   **Context & Nuance:** Traditional CUDA uses a "kernel barrier"—you launch Kernel A, wait for it to finish, then launch Kernel B. MPK removes this barrier. The scheduler inside the kernel decides *when* to run the next piece of work based on fine-grained dependencies.
*   **Analogy:** In a traditional factory, the boss (CPU) says, "Do Step 1, stop, tell me when it's done, then do Step 2." In MPK, the boss says, "Here is the assembly line rules. If Part A is done, immediately start Part B on the next machine." The line keeps moving without the boss checking in.
*   **Key Takeaway:** The Task Graph and In-Kernel Scheduler allow for fine-grained, dynamic scheduling within a single kernel, eliminating static barriers.

#### 5. Performance & Limitations
*   **Detailed Explanation:**
    *   **Performance:** Mirage outperforms vendor libraries by up to 3.3x. MPK outperforms standard serving systems by 1.2–6.7x.
    *   **Bottlenecks:** The primary bottleneck is **parameter enumeration** (e.g., trying all possible grid/block dimensions). This is currently an exponential cost.
    *   **Register Spillage:** Fusing too much code can exceed the register limit of an SM. MPK addresses this by separating schedulers and workers into different logical contexts (or using Green Contexts) to manage register allocation separately.
*   **Context & Nuance:** The system is "holistic" but not yet perfect. It relies on heuristics for parameter tuning and currently lacks a full visualizer for the search process.
*   **Key Takeaway:** While highly performant, the system's main computational cost is exploring hardware parameters (tiling sizes, grid dims), and register management is a critical constraint in mega-kernels.

---

### 3. Pathways for Further Exploration

1.  **Topic:** **Super-Optimizer Theory (TASL/PAT)**
    *   **Why it Matters:** To understand why Mirage is different, you must understand what it replaces. TASL (Tensor Algebra Super-Optimizer) and PAT (Partial Algebraic Transformation) are the predecessors.
    *   **Search/Study Direction:** Study the papers on **TASL** and **PAT**. Look for how they handle "algebraic transformations" vs. Mirage's "graph generation." Understand the limitations of rule-based systems.

2.  **Topic:** **Finite Field Arithmetic in Verification**
    *   **Why it Matters:** This is the key to Mirage's correctness checking. Understanding *why* finite fields prevent false negatives in random testing is crucial for compiler verification.
    *   **Search/Study Direction:** Look into **Galois Fields (GF)** and their application in **probabilistic equivalence checking** for floating-point code. Search for "random testing in finite fields for compiler verification."

3.  **Topic:** **CUDA Graphs vs. Task Graphs**
    *   **Why it Matters:** The lecture contrasts MPK's Task Graph with CUDA Graphs. Understanding the difference between static kernel-level dependencies and dynamic task-level dependencies is vital.
    *   **Search/Study Direction:** Compare **NVIDIA CUDA Graphs** (static, kernel-level) with **task-based runtime systems** in other frameworks (like **TVM** or **MLIR**). Look for how "in-kernel scheduling" compares to CPU-side scheduling.

4.  **Topic:** **Register Allocation & Spillage**
    *   **Why it Matters:** The lecture mentioned register spillage as a challenge in mega-kernels. This is a hardware constraint that dictates how much code can be fused.
    *   **Search/Study Direction:** Study **GPU Register Allocation** strategies. Look into how "Green Contexts" (CUDA feature) allow for finer-grained resource management than standard kernels.

5.  **Topic:** **Abstract Expression Systems**
    *   **Why it Matters:** The pruning mechanism is the engine of Mirage.
    *   **Search/Study Direction:** Explore **Abstract Interpretation** in compiler theory. Look for how "abstract domains" are used to approximate program behavior without executing it.

6.  **Topic:** **LLM Serving Dynamics (Continuous Batching)**
    *   **Why it Matters:** MPK is designed for LLMs, which have dynamic workloads (variable sequence lengths, pre-fill vs. decode phases).
    *   **Search/Study Direction:** Study **Continuous Batching** and **Speculative Decoding** in LLM inference. Understand why static kernel-per-layer approaches fail to optimize these dynamic patterns.

---

### 4. Comprehension & Review Questions

**Recall & Understanding**
1.  What are the three levels of the "Mile Graph" and which hardware memory level does each correspond to?
2.  What is the primary difference between traditional "transformation-based" compilers and Mirage's "exhaustive search" approach?
3.  Why does Mirage use "abstract expressions" instead of full tensor algebraic expressions for pruning?
4.  What is the role of the "Scheduler" warps in the MPK runtime system?
5.  How does the "Task Graph" differ from a standard "CUDA Graph"?

**Application & Analysis**
6.  Suppose you are compiling a simple `MatMul` followed by `Softmax`. How does Mirage use abstract expressions to prune invalid search branches during the generation of the Thread Block Graph?
7.  In the context of MPK, how does the removal of "kernel barriers" allow for the overlap of computation and communication (e.g., AllReduce)?
8.  If a generated candidate graph in Mirage passes the "probabilistic equivalence verification," what does that guarantee about its correctness? What is the risk?
9.  Why is "register spillage" a specific concern when moving from a "kernel-per-layer" approach to a "mega-kernel" approach?
10.  How does the MPK system handle dynamic workloads (like varying batch sizes) that would traditionally require a CPU-side restart?

**Critical Thinking & Evaluation**
11.  Critique the use of "exhaustive search" in Mirage. What is the fundamental computational cost of this approach, and how does the lecture suggest it is mitigated?
12.  The lecture states that MPK reduces engineering effort but introduces complexity in "register management." Evaluate the trade-off between the *performance* gained from kernel fusion and the *complexity* introduced by the in-kernel runtime.
13.  If you were to port Mirage to a new GPU architecture (e.g., AMD or a new NVIDIA H100 feature), which components of the Mile Graph and search process would require the most significant re-implementation, and why?

---

### Answer Key & Explanations

**Recall & Understanding**
1.  **Levels:** Kernel Graph (Device Memory/HBM), Thread Block Graph (Shared Memory/L1/L2), Thread Graph (Register File).
2.  **Difference:** Transformation systems use fixed, sparse rules (limited patterns). Mirage uses dense, exhaustive search to explore *all* possible valid graphs, allowing it to find custom, non-standard optimizations.
3.  **Reason:** Full tensor algebra is too complex and slow for solvers. Abstract expressions (scalar granularity) capture the necessary algebraic properties (commutativity, etc.) while being cheap to reason about, allowing fast pruning.
4.  **Role:** Scheduler warps monitor "Event" queues. When an event is fully triggered (all dependent tasks done), the scheduler dispatches the next dependent tasks to Worker SMs.
5.  **Difference:** CUDA Graph nodes are *kernels* (static, GPU-wide). Task Graph nodes are *tasks* (dynamic, SM-wide). Task Graphs allow fine-grained, dynamic dependencies within a single kernel launch.

**Application & Analysis**
6.  **Pruning Example:** If the desired expression is $Z = \sum W \cdot Norm(X)$, and a search branch computes $E^X$ (exponentiation), the abstract expression for that branch will contain an exponentiation operator. If the desired expression has no exponentiation, the theorem prover identifies this mismatch and prunes the branch immediately.
7.  **Overlap:** In a mega-kernel, an AllReduce task for one tile of data can start as soon as *that specific tile* of MatMul is done, rather than waiting for the *entire* MatMul kernel to finish. This allows SMs to communicate (NVLink) while other SMs are still computing.
8.  **Guarantee:** It guarantees that *if* the graphs are equivalent, outputs always match. The risk is a "false positive" (graphs are different, but outputs matched by chance), which has a small, calculable probability.
9.  **Register Spillage:** Fusing many operations into one kernel increases the number of variables live at any time. If this exceeds the SM's register limit, the compiler must "spill" variables to memory, causing a massive performance drop.
10.  **Dynamic Handling:** The in-kernel scheduler reads the actual data lengths (e.g., sequence length) at runtime. It can dispatch tasks dynamically based on what is actually present, rather than relying on a pre-compiled static schedule.

**Critical Thinking & Evaluation**
11.  **Critique:** The fundamental cost is exponential search space. It is mitigated by "Abstract Expression Pruning" (removing mathematically impossible branches) and "Parameter Enumeration" limits (though this is noted as a current bottleneck).
12.  **Evaluation:** The trade-off is significant. You gain massive performance (1.2–6.7x) and dynamic flexibility. However, you lose the simplicity of independent kernels. You must now manage register allocation carefully (using Green Contexts or separate schedulers) to avoid performance cliffs from register spillage.
13.  **Porting:** The **Operator Library** (the primitives at Kernel/Thread Block/Thread levels) must be re-implemented for the new hardware. The **Parameter Heuristics** (grid/block dimensions) must be tuned for the new architecture's specific memory hierarchy and core counts. The core search logic (Mile Graph/Pruning) remains largely agnostic.
