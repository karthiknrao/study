Here is your comprehensive study guide based on the lecture transcript. As your instructor, I have synthesized the raw transcript into a structured masterclass to help you master GPU programming, performance analysis, and Triton kernel design.

---

# Study Guide: GPU Programming, Performance, and Triton Kernels

### 1. Executive Summary & Core Concepts

**Lecture Overview:**
This lecture bridges high-level GPU architecture with low-level kernel programming, focusing on how hardware constraints dictate software performance. We move beyond simple correctness to optimize for performance by understanding the memory hierarchy (registers, shared memory, HBM) and the execution model (threads, warps, thread blocks). The core thesis is that while the programming model (threads/blocks) provides abstraction, high performance requires managing hardware limitations like register pressure, bank conflicts, and memory coalescing.

**Key Concepts Highlight:**
*   **Memory Hierarchy & Bandwidth:** The GPU has a tiered memory system: Registers (fastest, per-thread), L1/Shared Memory (fast, per-SM), L2 Cache (medium, per-chip), and HBM (slowest, large, global). Performance is inversely correlated with memory speed; moving data from HBM is the primary bottleneck.
*   **Warps & Lockstep Execution:** A warp is a group of 32 threads that execute instructions in "lockstep" (simultaneously). If threads in a warp diverge (e.g., `if/else` branches), performance drops because the warp must serialize execution.
*   **Occupancy & Register Pressure:** Occupancy is the ratio of active warps to resident warps on an SM. It is constrained by hardware limits (e.g., max registers per thread = 255). High register usage reduces the number of threads that can fit on an SM, potentially lowering occupancy.
*   **Bank Conflicts:** Shared memory is divided into 32 banks. If multiple threads in a warp access the same bank simultaneously, they must serialize, causing a "bank conflict" and significant performance loss.
*   **Memory Coalescing:** When a warp accesses HBM, memory requests are combined into 128-byte cache lines. "Coalesced" access (consecutive memory addresses) is efficient; non-coalesced access wastes bandwidth.
*   **Triton Programming Model:** Triton is a Python-based DSL where you write code at the **thread block** level, not the individual thread level. You specify how a block loads data, computes, and writes back, and the compiler handles thread synchronization and memory layout.
*   **Kernel Fusion & Compilation:** Naive PyTorch code launches multiple small kernels (slow due to HBM round-trips). `torch.compile` or custom Triton kernels fuse operations into a single kernel, reading from HBM once and writing once, drastically improving performance.
*   **Tiling:** A technique to break large matrices into smaller "tiles" that fit into shared memory. This reduces redundant HBM reads and increases arithmetic intensity.

---

### 2. Deep Dive: Expanded Lecture Notes

#### 1. The GPU Memory Hierarchy
*   **Detailed Explanation:** The GPU is not a single monolithic chip but a collection of Streaming Multiprocessors (SMs). Each SM has local, fast memory (Registers and L1/Shared Memory) and access to global, slow memory (HBM). The hierarchy is crucial because the speed gap between registers and HBM is massive.
*   **Context & Nuance:** The lecture notes that while register counts and SM counts have remained relatively stable (e.g., ~100-200 SMs, 65k registers per SM), HBM capacity has grown significantly. However, bandwidth is the limiting factor. Registers are fastest, followed by L1, L2, and finally HBM.
*   **Analogy:** Think of the GPU as a factory. HBM is the distant warehouse (slow to access, huge capacity). Shared Memory is the local workbench (fast, small). Registers are the tools in the worker's hand (instant access, very limited). You want to keep the tools (registers) and the immediate materials (shared memory) handy, rather than running to the warehouse (HBM) for every single item.
*   **Key Takeaway:** Performance is determined by how well you keep data in fast local memory (registers/shared) rather than constantly fetching from slow global memory (HBM).

#### 2. Warps, Lockstep, and Latency Hiding
*   **Detailed Explanation:** Threads are grouped into **warps** of 32 threads. All threads in a warp must execute the same instruction at the same time (lockstep). This is why branching (`if/else`) is expensive; it forces the warp to execute both branches sequentially. To hide the latency of slow operations (like HBM reads), the SM uses a **warp scheduler** to switch between warps. While one warp waits for memory, another warp performs compute operations.
*   **Context & Nuance:** This "zero-cost context switching" is a hardware feature designed specifically to hide memory latency. If you have high occupancy (many warps), you can hide latency better.
*   **Analogy:** Imagine a restaurant kitchen. A "warp" is a team of 32 chefs who must all chop onions at the exact same time. If one chef needs to go to the pantry (HBM), the whole team stops. To keep the kitchen running, the manager (Warp Scheduler) swaps out the waiting team for another team that is ready to cook (compute).
*   **Key Takeaway:** Avoid branching within a warp, and rely on warp switching (high occupancy) to hide the latency of memory accesses.

#### 3. Occupancy and Register Pressure
*   **Detailed Explanation:** Each thread has a limit of 255 registers. Since an SM has a fixed number of registers (e.g., 65,000), using more registers per thread means fewer threads can reside on the SM. **Occupancy** is the percentage of theoretical maximum warps that are actually active. High register usage lowers occupancy.
*   **Context & Nuance:** Low occupancy isn't always bad. If your threads are doing heavy compute, you might prefer fewer, "fatter" threads (thread coarsening) to reduce scheduling overhead. The lecture provides an example: 128 threads using 160 registers each results in low occupancy (~18%) because the register budget is exhausted.
*   **Analogy:** A theater has 100 seats (registers). If you decide each person takes 5 seats (high register usage), only 20 people can enter. If everyone takes 1 seat, 100 people can enter. Which is better depends on whether the "people" (threads) are doing heavy work or light work.
*   **Key Takeaway:** Register usage directly limits how many threads can run concurrently on an SM, impacting occupancy and latency hiding capabilities.

#### 4. Bank Conflicts and Memory Coalescing
*   **Detailed Explanation:**
    *   **Bank Conflicts:** Shared memory is split into 32 banks. If multiple threads in a warp access the *same* bank in the same cycle, they collide and must serialize. This is common in matrix operations where column access patterns are rigid.
    *   **Memory Coalescing:** When accessing HBM, a warp’s memory requests are merged into 128-byte transactions (cache lines). If threads access consecutive addresses (row-major), it’s "coalesced" and efficient. If they access scattered addresses (column-major on a row-major matrix), you waste bandwidth fetching unused data.
*   **Context & Nuance:** Bank conflicts are a shared memory issue; coalescing is an HBM issue. Both stem from how memory is laid out and accessed. Solutions like "swizzling" rearrange shared memory addresses to avoid conflicts.
*   **Analogy:**
    *   *Bank Conflict:* 32 students trying to use the same single phone line (bank) at once. They have to wait in line.
    *   *Coalescing:* Ordering pizza. If you order 32 consecutive items from one menu page, the kitchen is fast. If you order 32 random items from 32 different pages, the kitchen is slow.
*   **Key Takeaway:** Access patterns matter. Contiguous access (coalescing) maximizes HBM bandwidth, while avoiding shared memory bank collisions prevents serialization.

#### 5. Triton: The Thread-Block Abstraction
*   **Detailed Explanation:** Triton allows you to write kernels in Python-like syntax where you operate on **blocks** of data, not individual elements. You use `tl.load` and `tl.store` to move blocks of data between HBM and registers/shared memory. The compiler handles the underlying thread synchronization and warp management.
*   **Context & Nuance:** In CUDA, you write code for a single thread and manually manage synchronization. In Triton, you define the *block* logic. The compiler generates PTX (intermediate assembly) and then machine code. This abstraction makes it easier to write correct and performant code without deep warp-level knowledge.
*   **Analogy:** In CUDA, you are the individual bricklayer laying one brick at a time. In Triton, you are the foreman telling a crew of 32 bricklayers to lay a whole wall section. You don't tell each bricklayer exactly how to hold the trowel; you just say "lay this section," and the compiler figures out the individual moves.
*   **Key Takeaway:** Think in terms of "What does this block of data do?" rather than "What does this single thread do?" Triton abstracts away the complex thread-level bookkeeping.

#### 6. Kernel Fusion via Compilation
*   **Detailed Explanation:** A naive PyTorch implementation of an operation like GELU (activation function) often results in multiple small kernels (e.g., one for multiplication, one for tanh, one for addition). Each kernel must read from HBM, compute, and write back to HBM. **Kernel Fusion** (via `torch.compile` or custom Triton) combines these into a single kernel that reads once, computes the whole chain, and writes once.
*   **Context & Nuance:** The lecture demonstrated that a compiled/fused kernel is significantly faster than a naive multi-kernel approach because it minimizes HBM traffic. The built-in library kernels are also fast, but custom Triton kernels give you control over the fusion strategy.
*   **Analogy:**
    *   *Naive:* You go to the fridge (HBM) for milk, come back, then go back for bread, come back, then go back for eggs.
    *   *Fused:* You go to the fridge once, grab milk, bread, and eggs, and come back.
*   **Key Takeaway:** Minimizing the number of times data moves between HBM and local memory is the primary driver of performance gains in fused kernels.

#### 7. Tiling for Matrix Multiplication
*   **Detailed Explanation:** Matrix multiplication (MatMul) is memory-bound if done naively (reading entire rows/columns for every element). **Tiling** breaks matrices A and B into smaller blocks (tiles) that fit in shared memory. The kernel iterates over these tiles, loading them into shared memory, performing the dot product, and accumulating results.
*   **Context & Nuance:** This reduces HBM reads from $O(M \cdot N \cdot K)$ to $O(M \cdot N + K \cdot \text{tile\_size})$. It increases **arithmetic intensity** (operations per byte transferred). The lecture shows that tiling allows you to reuse data in shared memory, drastically cutting bandwidth usage.
*   **Analogy:** Instead of reading the entire library (HBM) every time you want to look up one book (element), you move the entire relevant shelf (tile) to your desk (shared memory) and work through it.
*   **Key Takeaway:** Tiling is the fundamental technique for optimizing matrix operations, balancing the trade-off between shared memory capacity and HBM bandwidth.

---

### 3. Pathways for Further Exploration

1.  **Topic: PTX (Parallel Thread Execution) Assembly**
    *   **Why it Matters:** The lecture showed that Triton compiles to PTX. Understanding PTX reveals how the compiler maps high-level Triton code to actual thread instructions and register usage.
    *   **Search/Study Direction:** "Study NVIDIA PTX ISA documentation, specifically how `ld.global` and `st.global` instructions map to memory coalescing and warp behavior."

2.  **Topic: Advanced Tiling Strategies (Swizzling)**
    *   **Why it Matters:** The lecture mentioned "swizzling" to avoid bank conflicts. This is a critical advanced technique for high-performance kernels.
    *   **Search/Study Direction:** "Investigate memory swizzling techniques in CUDA/Triton to mitigate bank conflicts in shared memory during matrix multiplications."

3.  **Topic: Arithmetic Intensity & Roofline Model**
    *   **Why it Matters:** The lecture referenced "arithmetic intensity" as a key metric. The Roofline Model is the standard framework for predicting performance based on compute vs. memory bandwidth.
    *   **Search/Study Direction:** "Learn the Roofline Model for GPUs, focusing on how to calculate FLOPs/Byte and determine if a kernel is compute-bound or memory-bound."

4.  **Topic: Flash Attention Algorithm**
    *   **Why it Matters:** The lecture ended by stating that the concepts learned (tiling, fusion, HBM management) are the ingredients for implementing Flash Attention.
    *   **Search/Study Direction:** "Read the Flash Attention paper (Dao et al.) and study how it uses tiling to keep attention matrices in SRAM/shared memory rather than HBM."

5.  **Topic: Warp Specialization**
    *   **Why it Matters:** The lecture discussed warp scheduling. Warp specialization is an advanced technique where different warps in a block perform different roles (e.g., one warp loads data, another computes).
    *   **Search/Study Direction:** "Explore 'Warp Specialization' in NVIDIA Hopper/Blackwell architectures for overlapping memory loads and tensor core operations."

6.  **Topic: Profiling Tools (Nsight Systems/Compute)**
    *   **Why it Matters:** The lecture emphasized the "benchmark, profile, iterate" loop. Nsight is the industry-standard tool for this.
    *   **Search/Study Direction:** "Learn how to use NVIDIA Nsight Compute to visualize bank conflicts, occupancy, and memory throughput in custom kernels."

---

### 4. Comprehension & Review Questions

**Recall & Understanding**
1.  What is the primary difference in speed and capacity between Registers and HBM?
2.  Define a "Warp" and explain why "control divergence" (branching) is detrimental to performance.
3.  What is "Occupancy" in the context of GPU programming, and what hardware constraint primarily limits it?
4.  Describe the difference between a "Bank Conflict" and "Memory Coalescing."
5.  In the Triton programming model, what is the unit of execution that you define, as opposed to CUDA?

**Application & Analysis**
6.  You are writing a Triton kernel for a vector of 10,000 elements. You set the block size to 1,024. How many thread blocks will be launched? How does the compiler handle the fact that 10,000 is not divisible by 1,024?
7.  A naive MatMul kernel has an arithmetic intensity of $O(1)$ (constant). How does tiling improve this, and what is the new order of magnitude for the reads?
8.  If you have a thread block where threads access memory addresses such that Thread 0 accesses Address 0, Thread 1 accesses Address 4, Thread 2 accesses Address 8... (stride of 4), is this access pattern coalesced? Why or why not?
9.  Why does `torch.compile` or a fused Triton kernel perform better than a naive PyTorch implementation of a complex operation like GELU?
10.  You observe that your kernel has low occupancy. You decide to reduce the number of registers used per thread. What is the likely effect on the number of concurrent threads and overall latency hiding?

**Critical Thinking & Evaluation**
11.  The lecture states that "high occupancy is not always the goal." Critique this statement. Under what circumstances might low occupancy be preferable or acceptable?
12.  Compare the abstraction levels of CUDA, Triton, and PyTorch. Which level is best suited for a developer who needs to implement a highly custom, memory-bound algorithm like Flash Attention, and why?
13.  The lecture mentions that hardware details are "messy" and depend on specific SM counts and register sizes. How does this hardware dependence impact the portability of optimized kernels across different GPU generations (e.g., A100 vs. H100)?

---

***

### Answer Key & Explanations

**1. Registers vs. HBM:**
Registers are the fastest, smallest memory, local to a thread. HBM is the slowest, largest memory, global to the chip. The bandwidth gap between them is significant, making HBM access a major bottleneck.

**2. Warp & Divergence:**
A warp is a group of 32 threads executing in lockstep. Divergence occurs when threads in a warp take different code paths (e.g., `if/else`). This forces the warp to serialize execution (execute the `if` branch, then the `else` branch), effectively doubling the time for that instruction sequence.

**3. Occupancy:**
Occupancy is the ratio of active warps to the maximum resident warps on an SM. It is primarily limited by register usage (max 255 registers per thread) and shared memory usage. If registers are exhausted, fewer threads can fit, lowering occupancy.

**4. Bank Conflict vs. Coalescing:**
*   **Bank Conflict:** Occurs in *Shared Memory* when multiple threads access the same of the 32 banks simultaneously, causing serialization.
*   **Memory Coalescing:** Relates to *HBM Access*. It is the efficiency of combining memory requests into 128-byte transactions. Coalesced (consecutive) access is efficient; scattered access is not.

**5. Triton Unit of Execution:**
In Triton, you define the behavior of a **Thread Block** (or simply a "block"). You specify how a block of data is loaded, processed, and stored, rather than defining the actions of a single thread.

**6. Triton Block Calculation:**
*   **Blocks Launched:** $\lceil 10,000 / 1,024 \rceil = 10$ blocks.
*   **Handling Remainder:** The compiler uses **masking**. The last block will have a mask that is `True` for the valid elements and `False` for the padding, ensuring out-of-bounds memory is not accessed.

**7. Tiling & Arithmetic Intensity:**
Naive MatMul reads $O(M \cdot N \cdot K)$ bytes. Tiling reduces HBM reads to $O(M \cdot N + K \cdot \text{tile\_size})$. This increases arithmetic intensity from $O(1)$ to $O(\text{tile\_size})$, meaning more operations are performed per byte of data moved.

**8. Coalescing Analysis:**
If the stride is 4 bytes (assuming 32-bit floats), Thread 0 accesses bytes 0-3, Thread 1 accesses 4-7, etc. This **is** coalesced because the 32 threads access a contiguous 128-byte block of memory (32 threads * 4 bytes = 128 bytes). This is the ideal scenario.

**9. Kernel Fusion Benefit:**
Naive PyTorch launches multiple kernels (e.g., one for `multiply`, one for `tanh`). Each must read from HBM, compute, and write back. Fusion combines these into one kernel that reads once, computes the whole chain in registers/shared memory, and writes once, drastically reducing HBM traffic.

**10. Register Reduction:**
Reducing registers per thread allows more threads to fit on the SM, increasing occupancy. This improves latency hiding because there are more warps available to switch to when one warp waits for memory.

**11. Critique on Occupancy:**
High occupancy is ideal for hiding latency, but if the kernel is compute-heavy (e.g., heavy math operations), lower occupancy with "fatter" threads (more registers) might be better to reduce scheduling overhead and improve instruction-level parallelism. Low occupancy is acceptable if the compute-to-memory ratio is high.

**12. Abstraction Levels:**
*   **PyTorch:** Highest level, good for general correctness, but limited control over memory layout.
*   **CUDA:** Lowest level, maximum control, but complex and error-prone.
*   **Triton:** Middle ground. Best for custom, memory-bound algorithms like Flash Attention because it allows block-level control (tiling, fusion) without manual warp management.

**13. Hardware Dependence:**
Optimized kernels are tuned for specific hardware (e.g., register counts, SM counts, tensor core capabilities). A kernel optimized for H100 (Blackwell/Hopper) may not perform optimally on A100 (Ampere) due to differences in shared memory size, register file size, and instruction sets. Portability requires re-tuning or using adaptive Triton code.
