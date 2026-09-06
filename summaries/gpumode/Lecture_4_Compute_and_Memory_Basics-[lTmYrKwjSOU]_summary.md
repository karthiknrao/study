### 1. Executive Summary & Core Concepts

**Lecture Overview:**
This lecture, part of a reading group on *Programming Massively Parallel Processors*, dissects the hardware architecture of NVIDIA GPUs (specifically the RTX 3090) to explain how computation and memory interact. The session moves from high-level hardware structures (Streaming Multiprocessors, Warps) to the critical performance bottlenecks defined by memory bandwidth and compute limits. It demonstrates that maximizing GPU performance requires balancing thread occupancy, avoiding warp divergence, and minimizing global memory accesses through techniques like kernel fusion and shared memory tiling.

**Key Concepts Highlight:**
*   **Streaming Multiprocessors (SMs):** The fundamental compute units of a GPU. Unlike CPUs with a few complex cores, GPUs have many SMs (e.g., 82 on an RTX 3090), each containing multiple Arithmetic Logic Units (ALUs) and shared resources like register files and L1 cache/shared memory.
*   **Warp Divergence:** A performance penalty that occurs when threads within a single 32-thread warp execute different branches of code (e.g., `if/else`). Since older architectures executed instructions step-by-step for all threads, inactive threads sat idle, reducing throughput.
*   **Occupancy:** The metric describing the ratio of active warps to the maximum possible warps on an SM. High occupancy allows the GPU to hide memory latency by switching between warps while others wait for data.
*   **Kernel Fusion:** The process of combining multiple separate operations (like loading, computing, and storing) into a single GPU kernel to reduce global memory traffic. This is a core principle behind modern frameworks like PyTorch’s inductor and Flash Attention.
*   **Roofline Model:** A performance model that visualizes the theoretical limits of a kernel based on two constraints: memory bandwidth (diagonal slope) and peak compute throughput (horizontal ceiling). It helps determine if a kernel is "memory-bound" or "compute-bound."
*   **Computational Intensity:** The ratio of Floating Point Operations (FLOPs) to bytes of memory transferred. Low intensity indicates a memory-bound kernel; high intensity indicates a compute-bound kernel.
*   **Shared Memory Tiling:** A technique to reduce global memory reads by loading data blocks into the fast, on-chip shared memory. This allows threads within a block to reuse data without repeatedly accessing the slow global memory.

---

### 2. Deep Dive: Expanded Lecture Notes (The "Teaching" Section)

#### 1. Streaming Multiprocessors (SMs) & Hardware Architecture
*   **Detailed Explanation:** The GPU architecture is fundamentally different from a CPU. A CPU core has a heavy "front end" (fetch, decode) and a single ALU. In contrast, an SM is a "streaming" unit designed for massive parallelism. An SM contains multiple ALUs (e.g., 32 FP32 units, 16 INT32 units) and a shared register file. The most critical distinction is the **context sharing**: instead of saving/restoring all registers when switching tasks, the GPU keeps registers allocated in the register file and switches between *warps* (groups of 32 threads). This allows near-instantaneous context switching.
*   **Context & Nuance:** The lecture highlights a hardware nuance: consumer GPUs (like the RTX 3090) have very limited FP64 (64-bit floating-point) capability compared to FP32. The RTX 3090 has a 1:64 ratio of FP64 to FP32 performance. If you accidentally use 64-bit floats (e.g., using `int64` for indexing), your kernel will be drastically slower because the hardware lacks dedicated high-speed FP64 units.
*   **Analogy or Real-World Example:** Think of a CPU as a single, highly skilled chef who can handle complex ingredients one by one. An SM is like a factory line with 32 identical workers who can only do simple tasks (add, multiply) but do them in perfect unison. If one worker gets stuck waiting for an ingredient, the whole line slows down unless the manager (the scheduler) can switch to another group of 32 workers (a new warp) instantly.
*   **Key Takeaway:** To keep the GPU busy, you must understand that it is not "cores" but "SMs" that are the unit of parallelism, and you must manage resource limits (registers, shared memory) to maximize the number of active warps.

#### 2. Warps, Scheduling, and Divergence
*   **Detailed Explanation:** Threads are grouped into **warps** of 32 threads. Ideally, all 32 threads execute the same instruction at the same time. **Divergence** occurs when code paths differ (e.g., `if (thread_id < 16)`). In older architectures, the GPU would execute one path while masking the other threads, then switch. Modern GPUs (Volta and later) introduced per-thread program counters, allowing the warp to split into smaller groups that execute in parallel. However, this complicates synchronization.
*   **Context & Nuance:** Divergence is a silent killer of performance. Even if the code looks parallel, if 16 threads take one path and 16 take another, the hardware efficiency drops. The lecture notes that while modern hardware mitigates this via interleaving, it introduces complexity in synchronization (requiring explicit `syncwarp` commands).
*   **Analogy or Real-World Example:** Imagine a row of 32 students taking a quiz. If the question is "Add 1 to 2," everyone answers instantly. If the question is "If you are a boy, add 1 to 2; if you are a girl, add 2 to 4," the teacher has to manage two groups. If the hardware can't handle two groups simultaneously, one group waits while the other works. This is divergence.
*   **Key Takeaway:** Avoid branching code within a warp where possible. If divergence is unavoidable, be aware that it reduces throughput and complicates memory access patterns.

#### 3. Occupancy and Resource Balancing
*   **Detailed Explanation:** **Occupancy** is the percentage of the maximum possible warps that are actually active on an SM. It is not a single metric but a balance of constraints:
    1.  **Register Usage:** If each thread uses too many registers, fewer warps can fit in the register file.
    2.  **Shared Memory:** If a block uses too much shared memory, fewer blocks can reside on an SM.
    3.  **Block Size:** Block sizes should ideally divide the maximum threads per SM (e.g., 1536 on RTX 3090). A block size of 256 or 512 is often optimal to ensure enough warps are available to hide latency.
*   **Context & Nuance:** High occupancy is crucial for **hiding memory latency**. If one warp is waiting for data from global memory, the scheduler can instantly switch to another warp that is ready to compute. If occupancy is low, the GPU sits idle while waiting for memory.
*   **Analogy or Real-World Example:** Think of occupancy like a restaurant. If you have 100 seats (max warps) but only 20 people are eating (low occupancy), and the kitchen (memory) is slow, the restaurant is empty. If you have 90 people eating, and the kitchen is slow, the manager can keep 90 people busy chatting or looking at menus (computing) while the food is being prepared, so the table turnover is fast.
*   **Key Takeaway:** You must actively manage register and shared memory usage to maintain high occupancy. Use tools like `torch.cuda.get_device_properties` or profiling tools to check if you are hitting resource limits.

#### 4. Kernel Fusion and Memory Hierarchy
*   **Detailed Explanation:** The lecture emphasizes that **global memory is the bottleneck**. In eager PyTorch, every operation (e.g., `x * 2`, `x + 1`) is a separate kernel launch, meaning data is written to global memory, then read back for the next operation. **Kernel Fusion** combines these operations into a single kernel, keeping intermediate results in registers or shared memory, thus eliminating redundant global memory reads/writes.
*   **Context & Nuance:** This is the core innovation behind **Flash Attention** and modern compilers (Triton, Inductor). By fusing operations, you move from a "read-compute-write" pattern to a "read-compute-compute-...-write" pattern. The lecture demonstrated this with a GELU approximation, showing that a fused custom kernel was significantly faster than the unfused PyTorch chain.
*   **Analogy or Real-World Example:** Without fusion, it’s like a factory where every step requires shipping the product to a warehouse (global memory) and back. With fusion, the product stays on the assembly line (registers/shared memory) until the final step, drastically reducing shipping costs (memory bandwidth).
*   **Key Takeaway:** Reducing global memory accesses is the primary driver of performance in GPU computing. Always ask: "Can I keep this data in registers or shared memory instead of writing it to global memory?"

#### 5. The Roofline Model and Computational Intensity
*   **Detailed Explanation:** The **Roofline Model** is a diagnostic tool. It plots **Computational Intensity** (FLOPs per Byte) against performance.
    *   **Memory-Bound:** If your intensity is low (diagonal part of the roofline), your speed is limited by bandwidth. Adding more compute units won't help.
    *   **Compute-Bound:** If your intensity is high (flat part of the roofline), your speed is limited by the ALUs' speed.
    *   **The "Roof":** The maximum possible performance is the *minimum* of the bandwidth limit and the compute limit.
*   **Context & Nuance:** The lecture used an RGB-to-Gray conversion as an example. It had 5 FLOPs per 4 bytes of data, yielding an intensity of 1.25. This is low, meaning it is memory-bound. The theoretical "speed of light" was calculated based on bandwidth, and the measured time was close to this limit, confirming it was memory-bound.
*   **Analogy or Real-World Example:** Imagine a highway. If there are few cars (low intensity), the speed is limited by the road capacity (bandwidth). If there are millions of cars (high intensity), the speed is limited by the car engines (compute). The Roofline model tells you which limit you are hitting.
*   **Key Takeaway:** Before optimizing, determine if you are memory-bound or compute-bound. If memory-bound, focus on reducing memory traffic (tiling, fusion). If compute-bound, focus on instruction-level parallelism and avoiding divergence.

#### 6. Shared Memory Tiling
*   **Detailed Explanation:** **Tiling** is a technique to improve data locality in matrix multiplication. Instead of every thread reading the same global memory values repeatedly, a block of threads cooperatively loads a "tile" (a small sub-matrix) into shared memory. Since shared memory is on-chip and fast, subsequent calculations reuse this data without hitting global memory.
*   **Context & Nuance:** The lecture demonstrated tiling for matrix multiplication. The naive approach reads global memory $N$ times. Tiling reduces this to $N/tile\_size$. The key implementation details include:
    1.  Loading data into shared memory.
    2.  **Synchronization (`__syncthreads()`):** Crucial to ensure all threads have finished writing to shared memory before any thread starts reading from it.
    3.  Handling non-divisible matrix sizes by padding with zeros.
*   **Analogy or Real-World Example:** Imagine a group of 32 people (threads) needing to look up numbers in a huge book (global memory) that is far away. Instead of everyone walking to the book for every calculation, they send one person to fetch a page (tile) and share it among the group (shared memory). This reduces trips to the book.
*   **Key Takeaway:** Tiling transforms a memory-bound problem into a more balanced one by increasing data reuse and reducing global memory traffic. Always use `__syncthreads()` to prevent race conditions when accessing shared memory.

---

### 3. Pathways for Further Exploration

1.  **The Topic/Concept:** Coalesced Memory Access
    *   **Why it Matters:** The lecture mentioned that the next session will cover this. It is the critical next step after tiling. It ensures that when a warp accesses global memory, the 32 threads access 32 *consecutive* memory addresses.
    *   **Search/Study Direction:** Look into "GPU Memory Coalescing" and "Vectorized Memory Access." Study how to arrange data in memory (row-major vs. column-major) to ensure threads read data in a single transaction rather than multiple transactions.

2.  **The Topic/Concept:** Flash Attention Algorithm
    *   **Why it Matters:** The lecture used Flash Attention as a prime example of kernel fusion and tiling. Understanding the original paper is essential for modern LLM optimization.
    *   **Search/Study Direction:** Read the "FlashAttention: Fast and Accurate Attention" paper. Focus on how it blocks the Q, K, V matrices and uses online softmax to avoid materializing the full attention matrix in global memory.

3.  **The Topic/Concept:** CUDA Profiling Tools (Nsight / Nsight Systems)
    *   **Why it Matters:** The lecture mentioned "In-Sight Compute" and the importance of profiling. You cannot optimize what you cannot measure.
    *   **Search/Study Direction:** Learn how to use NVIDIA Nsight Systems and Nsight Compute. Specifically, study how to interpret "Occupancy" metrics, "Warp Stall Reasons," and "Memory Throughput" charts.

4.  **The Topic/Concept:** Tensor Cores
    *   **Why it Matters:** The lecture briefly mentioned Tensor Cores on the RTX 3090 but didn't detail them. These are specialized hardware units for matrix operations (like GEMM) that significantly change the compute-bound landscape.
    *   **Search/Study Direction:** Study "NVIDIA Tensor Core Programming Guide." Understand how to write kernels that leverage `wmma` (Warp Matrix Multiply-Accumulate) instructions or use libraries like cuBLAS to utilize them.

5.  **The Topic/Concept:** Numerical Stability in Floating Point Arithmetic
    *   **Why it Matters:** The lecture touched on how the order of addition affects results due to non-associativity in floating-point math. This is critical for reproducibility and accuracy in deep learning.
    *   **Search/Study Direction:** Investigate "Kahan Summation" and "Pairwise Summation" algorithms. Understand why `float32` vs `float64` matters in training vs. inference.

6.  **The Topic/Concept:** Thread Block Clusters (Hopper Architecture)
    *   **Why it Matters:** The lecture mentioned "thread block groups" are coming. The next generation of GPU architecture (H100) introduces thread block clusters, allowing synchronization between blocks on different SMs.
    *   **Search/Study Direction:** Look into "CUDA Hopper Architecture" and "Thread Block Clusters." Understand how this changes the programming model for large-scale parallelism.

---

### 4. Comprehension & Review Questions

**Recall & Understanding**
1.  What is the primary structural difference between a CPU core and a GPU Streaming Multiprocessor (SM) regarding the Arithmetic Logic Unit (ALU)?
2.  Define "Warp Divergence." What happens to the threads that do not match the executed branch?
3.  What is the "speed of light" in the context of GPU kernel performance?
4.  What is the difference between "local memory" and "shared memory" in terms of scope and speed?

**Application & Analysis**
5.  You are designing a kernel for an RTX 3090. You notice that your register usage is very high, causing low occupancy. How does this negatively impact performance, and what is one strategy to mitigate it?
6.  You have a kernel with a computational intensity of 0.5 FLOPs/Byte. Based on the Roofline Model, is this kernel likely to be memory-bound or compute-bound? Why?
7.  In a matrix multiplication kernel using tiling, why is `__syncthreads()` necessary? What would happen if you omitted it?
8.  You are profiling a PyTorch model and find that GPU utilization is low, but the CPU is busy. Based on the lecture's breakdown of time, which component is likely the bottleneck?
9.  Why is it often better to use a block size of 256 or 512 threads rather than 1024 threads on modern GPUs?

**Critical Thinking & Evaluation**
10. The lecture states that consumer GPUs have poor FP64 performance. Critique the decision to use 64-bit floats for indexing in a kernel that is otherwise optimized for FP32. What are the trade-offs?
11. Kernel fusion significantly improves performance by reducing global memory writes. However, it can sometimes lead to numerical differences. How should an engineer balance the need for performance (fusion) against the need for strict numerical reproducibility?
12. If you were to implement a custom "Softmax" kernel, would you prioritize optimizing for register usage or shared memory usage? Justify your choice based on the data dependencies of the operation.

***

**Answer Key & Explanations**

1.  **Recall:** A CPU core has a single (or few) complex ALUs with heavy fetch/decode logic. An SM has many simple ALUs (e.g., 32 FP32 units) and shares context (like fetch/decode) across a group of threads (a warp).
2.  **Recall:** Warp Divergence occurs when threads in a warp execute different code paths. In older architectures, the non-matching threads are "disabled" (masked) and sit idle while the matching threads execute. In newer architectures, the warp splits into smaller groups, but synchronization becomes more complex.
3.  **Recall:** The "speed of light" is the theoretical maximum performance limit of a kernel, calculated based on the maximum memory bandwidth available (bytes transferred per second). It represents the absolute fastest a kernel could run if it were purely limited by memory bandwidth.
4.  **Recall:** Local memory is per-thread and resides in global memory (slow). Shared memory is per-block, resides on-chip (fast/L1 cache), and is accessible to all threads within that block.
5.  **Application:** High register usage limits the number of warps that can be resident on an SM, leading to low occupancy. Low occupancy means fewer warps can hide memory latency, causing the GPU to wait idly for data. Mitigation includes reducing local variables, using shared memory for intermediate calculations, or using compiler hints to limit register allocation.
6.  **Application:** An intensity of 0.5 FLOPs/Byte is low. This means for every byte of data moved, only 0.5 operations are performed. This places the kernel in the "memory-bound" region of the Roofline Model, where bandwidth is the limiting factor, not compute.
7.  **Application:** `__syncthreads()` is necessary to ensure that all threads have finished writing their data into shared memory before any thread begins reading from it. Without it, a race condition occurs where a thread might read a value that hasn't been written yet, leading to incorrect results.
8.  **Application:** If GPU utilization is low but the CPU is busy, the bottleneck is likely **Data Acquisition** or **Python Overhead**. The lecture notes that if the GPU is not at ~100% utilization, the issue is often that the GPU is waiting for the CPU to feed it data or manage tensors.
9.  **Application:** Block sizes of 256 or 512 divide evenly into the maximum threads per SM (1536 on RTX 3090). This ensures that multiple blocks can fit on an SM, maximizing the opportunity to have many warps ready to execute. A block size of 1024 is too large to fit efficiently alongside other blocks, potentially leaving resources underutilized.
10.  **Critical Thinking:** Using 64-bit floats for indexing on consumer GPUs is a severe performance trap. While it ensures correctness for very large tensors, it triggers the slow FP64/INT64 path (1:64 ratio). The trade-off is that unless the tensor is astronomically large (requiring > $2^{31}$ elements), the performance hit is unjustified. Engineers should use `int32` or `uint32` for indexing whenever possible to stay on the fast path.
11.  **Critical Thinking:** Fusion changes the order of operations (e.g., combining additions). Since floating-point addition is not associative, this can lead to slight numerical differences. An engineer must decide if the tiny performance gain (or memory savings) outweighs the risk of breaking reproducibility. In training, strict reproducibility is often required, so fusion might be disabled or verified for numerical equivalence.
12.  **Critical Thinking:** Softmax requires reading a row of values, finding the max, subtracting, exponentiating, summing, and dividing. This is inherently sequential per-row. It would prioritize **register usage** for the intermediate values (max, sum) to keep the row data local to the thread, avoiding global memory writes for intermediate steps. However, if the row is large, it might use shared memory to hold the row if it exceeds register capacity. The choice depends on the row length, but generally, keeping the row in registers/local memory is key to speed.
