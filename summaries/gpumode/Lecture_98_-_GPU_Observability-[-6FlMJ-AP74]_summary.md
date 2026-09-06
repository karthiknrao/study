### 1. Executive Summary & Core Concepts

**Lecture Overview:**
This lecture, presented by Yu-Sheng Zhang (UC Santa Cruz), explores the application of eBPF (extended Berkeley Packet Filter) technology to GPU systems to solve two critical problems: observability and extensibility. While existing GPU profilers (like NVBit and NCU) offer limited, often high-overhead, or closed-source visibility, eBPF provides a safe, dynamic, and programmable interface for both the GPU device (kernel) and the driver layer. The core thesis is that by compiling eBPF programs into PTX (Parallel Thread Execution) for the device and embedding hooks in the driver, we can achieve fine-grained, cross-layer tracing and dynamic policy customization (e.g., memory management and scheduling) without recompiling applications or risking kernel stability.

**Key Concepts Highlight:**
*   **GPU Observability vs. Inflexibility:** The dual problem of GPU stacks being "black boxes" where internal states are invisible, and policies (scheduling, memory management) being hard-coded and rigid, preventing dynamic adaptation.
*   **eBPF for GPUs (bpf-time):** A specialized runtime that compiles eBPF programs into PTX code, allowing them to execute directly on the GPU as part of a CUDA kernel, rather than just on the CPU.
*   **PTX Injection:** The mechanism by which eBPF logic is dynamically inserted into the PTX assembly of a GPU kernel at runtime, similar to dynamic binary instrumentation but safer and more flexible.
*   **Cross-Layer Tracing:** The ability to correlate CPU-side events (via standard eBPF) with GPU-side events (via device-side eBPF) to create a unified view of system performance, overcoming the "silos" of traditional profiling.
*   **Programmable Driver Hooks:** Using eBPF to expose safe, narrow interfaces within the GPU driver (e.g., in the Linux kernel) to modify Unified Virtual Memory (UVM) prefetching policies and TSG (Thread Group Scheduling) parameters dynamically.
*   **TSG (Thread Group Scheduling) & UVM:** The hardware/driver abstractions for GPU scheduling and memory migration. The lecture highlights that these are currently opaque to users but can be controlled via eBPF hooks.
*   **Safety Guarantees:** eBPF’s verifier ensures programs are safe (no infinite loops, no illegal memory access) before execution, making it safer than traditional kernel modules or LD_PRELOAD hacks.

---

### 2. Deep Dive: Expanded Lecture Notes

#### Concept 1: The Problem Landscape: Invisible & Inflexible GPU Stacks
*   **Detailed Explanation:** The GPU software stack consists of three layers: user-space applications (PyTorch, CUDA Runtime), the GPU kernel driver (Linux kernel component handling memory/scheduling), and the GPU device itself (firmware/hardware). Currently, these layers are "invisible" because they are closed-source or provide only high-level metrics (like NCU). They are "inflexible" because policies like Round-Robin scheduling or fixed-time slices are hard-coded.
*   **Context & Nuance:** Traditional tools like `LD_PRELOAD` or `LD_PRELOAD`-based libraries are application-bound and lack cross-process visibility. Hardware profilers (NCU) rely on callbacks that are either high-overhead or coarse-grained. This lack of granularity prevents developers from understanding *why* a specific page is evicted or *why* a kernel is scheduled in a specific order.
*   **Analogy:** Think of the GPU stack as a sealed, fast-moving factory. You can see the final product (output data), but you can’t see the internal assembly lines (scheduling) or the inventory management (memory migration). eBPF acts as a transparent window and a remote control panel for these internal mechanisms.
*   **Key Takeaway:** The GPU ecosystem suffers from a lack of dynamic, safe, and fine-grained control interfaces, forcing developers to rely on static, opaque, or unsafe modification techniques.

#### Concept 2: eBPF Fundamentals & Safety
*   **Detailed Explanation:** eBPF programs are small, verified C-like programs that run inside the Linux kernel. They are attached to specific tracepoints or kernel functions. Crucially, they undergo a **verification step** before execution to ensure they terminate and do not access illegal memory. This makes them far safer than traditional Linux kernel modules, which require a reboot to load and can crash the system if buggy.
*   **Context & Nuance:** In the CPU world, eBPF is used for networking, security, and tracing. The innovation here is extending this paradigm to the GPU. The "verifier" is the key differentiator—it allows dynamic, user-defined logic to run in privileged contexts without compromising system stability.
*   **Analogy:** If a Linux kernel module is like writing a custom operating system patch (high risk, requires reboot), eBPF is like a smart, sandboxed plugin that the OS inspects line-by-line to ensure it doesn’t crash the host before letting it run.
*   **Key Takeaway:** eBPF provides a safe, dynamic, and verified execution environment that bridges the gap between user-space flexibility and kernel/hardware-level power.

#### Concept 3: Device-Side eBPF (bpf-time) & PTX Injection
*   **Detailed Explanation:** To run eBPF on the GPU, the tool **bpf-time** compiles eBPF programs into PTX (the intermediate assembly for NVIDIA GPUs). It uses a runtime injection mechanism:
    1.  The CUDA application is compiled to a `fatbin` (containing PTX).
    2.  `bpf-time` hooks into the CUDA runtime (via `LD_PRELOAD` or ptrace).
    3.  It decompiles the PTX, injects the eBPF logic (as a "trampoline" or inline hook), and reassembles it.
    4.  When the GPU kernel runs, it executes both the original CUDA code and the injected eBPF tracing logic.
*   **Context & Nuance:** This is distinct from NVBit (which is a dynamic binary instrumentation tool that modifies SASS/machine code) or NCU (which is a high-level profiling interface). bpf-time operates at the PTX level, allowing for custom logic insertion that is safer and more flexible than raw binary patching.
*   **Analogy:** Imagine a video game where you can inject a "debug mode" into the game engine at runtime. Instead of changing the game’s core code (risky), you add a small, verified script that logs player actions. bpf-time does this for GPU kernels.
*   **Key Takeaway:** By compiling eBPF to PTX and injecting it into the kernel binary, we can achieve fine-grained, per-thread, per-warp tracing directly on the GPU hardware.

#### Concept 4: Cross-Layer Observability & Correlation
*   **Detailed Explanation:** A major strength of this approach is **cross-layer correlation**. Standard tools struggle to link CPU events (e.g., a specific CUDA API call) with GPU events (e.g., a specific warp stalling on memory). With eBPF, you can:
    *   Use CPU-side eBPF to trace `cudaLaunchKernel` calls.
    *   Use GPU-side eBPF to trace kernel entry/exit and warp states.
    *   Correlate timestamps to calculate "launch latency" or identify if the bottleneck is CPU-side (kernel launch overhead) or GPU-side (SM resource contention).
*   **Context & Nuance:** This solves the "black box" problem. For example, if a kernel is slow, you can now determine if it’s because the CPU took too long to launch it, or if the GPU threads were waiting for SM resources due to other multi-process workloads.
*   **Analogy:** Instead of watching a car’s dashboard (CPU) or the engine (GPU) separately, you now have a telemetry system that links the driver’s input (steering) to the wheel’s rotation, showing exactly how much time is lost in the transmission.
*   **Key Takeaway:** eBPF enables a unified timeline of CPU and GPU activities, allowing for precise root-cause analysis of performance bottlenecks across the entire stack.

#### Concept 5: Driver-Side eBPF & Programmable Policies
*   **Detailed Explanation:** Beyond just observing, eBPF can modify driver behavior. The lecture introduces **programmable hooks** in the GPU driver (Linux kernel) for:
    *   **UVM (Unified Virtual Memory):** Changing page eviction and prefetching policies (e.g., LRU vs. LFU).
    *   **Scheduling:** Adjusting TSG (Thread Group Scheduling) parameters like time slices and interleaving levels, which are normally hidden from user space.
*   **Context & Nuance:** Currently, UVM uses "blackboard" policies that are often suboptimal for AI workloads (e.g., LLM inference). By exposing these via eBPF, users can implement custom policies (like "prefetch more aggressively for MoE models") without recompiling the driver.
*   **Analogy:** Previously, the operating system’s memory manager was a strict manager who followed a fixed rulebook. eBPF allows you to hire a consultant (your eBPF program) who can dynamically change the rulebook in real-time based on current workload demands.
*   **Key Takeaway:** eBPF transforms the GPU driver from a static, opaque component into a programmable service, allowing for safe, dynamic customization of memory and scheduling policies.

#### Concept 6: Performance Optimization for GPU eBPF
*   **Detailed Explanation:** Running eBPF on the GPU is not trivial due to the SIMT (Single Instruction, Multiple Threads) execution model.
    *   **Thread Divergence:** If eBPF code runs per-thread, it can cause massive performance overhead due to branching.
    *   **Solution:** Execute eBPF logic **once per warp** (or per block) instead of once per thread. Since warps execute in lockstep, this reduces divergence.
    *   **Map Placement:** eBPF maps (data structures) must be placed carefully. Placing them in CPU memory causes high latency. The system uses a strategy to place maps in GPU-local memory for fast access, with duplicate versions on the host for CPU-side retrieval.
*   **Context & Nuance:** This highlights the architectural difference between CPU (single-threaded, branch-predicting) and GPU (massively parallel, SIMD) execution. Naively porting CPU eBPF techniques to GPU fails without these optimizations.
*   **Analogy:** On a CPU, you can have a complex decision tree in a loop. On a GPU, if 1000 threads try to take different paths in that tree, the hardware has to serialize them. By forcing the "decision" to happen once per group of threads (warp), you avoid the serialization penalty.
*   **Key Takeaway:** To make eBPF efficient on GPUs, the execution model must be adapted from "per-thread" to "per-warp/block," and data structures (maps) must be optimized for GPU memory hierarchy.

#### Concept 7: The bpf-time Architecture
*   **Detailed Explanation:** **bpf-time** is the specific tool/runtime developed by the speaker. It acts as an `LD_PRELOAD` library that intercepts CUDA runtime calls.
    *   **Workflow:** It takes the eBPF source, compiles it to PTX, injects it into the target application’s PTX, and attaches to the process.
    *   **Flexibility:** It supports standard eBPF maps and helpers, allowing data to flow between the GPU and CPU seamlessly.
*   **Context & Nuance:** This tool bridges the gap between the high-level CUDA application and the low-level hardware, providing a "programmable interface" similar to how eBPF works for CPU scheduling.
*   **Analogy:** bpf-time is the "bridge" between the world of standard eBPF tools (like BCC or libbpf) and the NVIDIA GPU stack. It translates the eBPF dialect into a form the GPU can understand and execute.
*   **Key Takeaway:** bpf-time is the practical implementation that makes GPU eBPF accessible, handling the complex PTX injection and map management for the user.

---

### 3. Pathways for Further Exploration

1.  **The Topic/Concept:** **PTX (Parallel Thread Execution) Assembly & Injection**
    *   **Why it Matters:** Understanding the intermediate representation (PTX) is crucial because bpf-time operates at this level. You need to know how PTX handles function calls and memory to understand how eBPF logic is injected.
    *   **Search/Study Direction:** Study the NVIDIA PTX ISA (Instruction Set Architecture) documentation, specifically focusing on "Function Calls," "Inline Assembly," and "Memory Consistency." Look into how tools like NVBit perform SASS (machine code) injection compared to PTX injection.

2.  **The Topic/Concept:** **GPU Scheduling Abstractions (TSG & Time Slicing)**
    *   **Why it Matters:** The lecture highlights that TSG (Thread Group Scheduling) parameters are hidden. Understanding this layer is key to mastering driver-level eBPF.
    *   **Search/Study Direction:** Research NVIDIA’s "GPU Time Slicing" and "MIG (Multi-Instance GPU)" documentation. Investigate how the Linux kernel’s `nvidia` driver manages context switching and TSG lifecycles.

3.  **The Topic/Concept:** **Unified Virtual Memory (UVM) Page Migration**
    *   **Why it Matters:** The lecture discusses customizing UVM policies. To understand the impact of eBPF hooks here, you must understand the baseline behavior.
    *   **Search/Study Direction:** Deep dive into NVIDIA’s UVM implementation in the Linux kernel. Look for papers on "Transparent Page Migration for GPUs" and how "Prefetching" algorithms (LRU, LFU) affect inference latency.

4.  **The Topic/Concept:** **eBPF Verifier & Safety Constraints**
    *   **Why it Matters:** The safety of eBPF relies on the verifier. Understanding its constraints (e.g., bounded loops, register usage) helps in writing efficient GPU eBPF programs.
    *   **Search/Study Direction:** Read the "eBPF Verifier" documentation in the Linux kernel source. Study how the verifier handles "bounded loops" and "memory safety" to understand why certain patterns are forbidden.

5.  **The Topic/Concept:** **Dynamic Binary Instrumentation (DBI) vs. eBPF**
    *   **Why it Matters:** The lecture contrasts eBPF with tools like NVBit and NCU. Understanding the trade-offs is vital for choosing the right tool.
    *   **Search/Study Direction:** Compare "Static Instrumentation" (compile-time) vs. "Dynamic Instrumentation" (runtime). Look into the overhead differences between SASS-level patching (NVBit) and PTX-level injection (bpf-time).

6.  **The Topic/Concept:** **Cross-Layer Performance Analysis**
    *   **Why it Matters:** The ultimate goal is correlation. Learn how to build "frame graphs" that link CPU syscalls to GPU kernel execution.
    *   **Search/Study Direction:** Study "System-Level Performance Analysis" for heterogeneous computing. Look for case studies on "CPU-GPU Synchronization Overhead" and how to measure "Kernel Launch Latency" using tracing tools.

---

### 4. Comprehension & Review Questions

**Recall & Understanding**
1.  What are the two primary problems identified in the current GPU software stack that motivate the use of eBPF?
2.  What is the core safety mechanism that distinguishes eBPF programs from traditional Linux kernel modules?
3.  In the context of bpf-time, what is "PTX injection" and why is it necessary?
4.  What is the difference between NVBit and NCU (NVIDIA Compute Unified Profiler) as described in the lecture?
5.  What does "cross-layer observability" mean in the context of GPU systems?

**Application & Analysis**
6.  A developer notices that their LLM inference latency is high. They suspect the issue is not in the model itself but in how memory is being managed. How could driver-side eBPF help them diagnose and potentially fix this without recompiling the driver?
7.  You are using bpf-time to trace a CUDA kernel. You find that the eBPF program is causing significant performance degradation due to thread divergence. What specific optimization strategy did the lecture suggest to mitigate this?
8.  Explain how the "map placement strategy" in bpf-time addresses the latency issue between CPU and GPU eBPF traces.
9.  If you wanted to implement a custom "Least Frequently Used" (LFU) eviction policy for GPU memory, which layer of the stack would you target with eBPF, and why?
10.  How does the TSG (Thread Group Scheduling) lifecycle relate to the user-space CUDA runtime, and why is it important to expose TSG parameters via eBPF?

**Critical Thinking & Evaluation**
11.  The lecture argues that eBPF is safer than `LD_PRELOAD` or kernel module patching. Critique this argument: What are the residual risks of running eBPF on the GPU, especially given the "per-warp" execution model?
12.  Compare the "Invisible" problem of the GPU stack with the "Inflexible" problem. Which one poses a greater barrier to AI research, and why?
13.  The lecture mentions that "hardware is evolving faster than software algorithms." Evaluate the long-term sustainability of using eBPF for GPU policy customization. Will this approach become obsolete as GPU architectures (e.g., disaggregated memory, new scheduling fabrics) evolve?

***

### Answer Key & Explanations

**1. The Two Primary Problems:**
*   **Invisible:** The GPU stack is a "black box"; internal states (driver decisions, hardware scheduling) are not visible to users.
*   **Inflexible:** Policies (scheduling, memory management) are hard-coded and cannot be dynamically adjusted without recompiling or unsafe hacks.

**2. Core Safety Mechanism:**
*   The **eBPF Verifier**. Before execution, the verifier checks the program for infinite loops, illegal memory accesses, and other safety hazards. This ensures the program cannot crash the kernel or the system, unlike traditional kernel modules.

**3. PTX Injection:**
*   **What:** It is the process of taking the PTX assembly code of a CUDA kernel and inserting eBPF logic (as a trampoline or inline hook) into it.
*   **Why:** CUDA compiles to PTX, which is then compiled to machine code. Injecting at the PTX level allows for dynamic modification of the kernel’s behavior (tracing/policy) without requiring the user to recompile their C/C++ CUDA source code.

**4. NVBit vs. NCU:**
*   **NVBit:** A dynamic binary instrumentation tool that modifies the SASS (machine code) to insert tracing logic. It offers fine-grained control but is complex and has higher overhead.
*   **NCU:** A high-level profiling interface provided by NVIDIA. It is convenient and requires no code changes, but it is coarse-grained, read-only, and relies on callbacks that may not capture fine-grained, cross-layer details.

**5. Cross-Layer Observability:**
*   The ability to correlate events from different layers of the stack (e.g., CPU-side API calls with GPU-side kernel execution) to get a holistic view of system performance. It breaks down the "silos" where CPU and GPU are monitored separately.

**6. Diagnosing Memory Latency:**
*   Driver-side eBPF can trace **UVM (Unified Virtual Memory) page faults and migration events**. By hooking into the driver, you can see *when* a page fault occurs, *how long* the migration took, and *which* policy was used. You can then dynamically change the prefetching algorithm (e.g., from LRU to LFU) via eBPF hooks to see if it improves latency.

**7. Mitigating Thread Divergence:**
*   The lecture suggests executing the eBPF logic **once per warp** (or per block) rather than once per thread. Since warps execute in lockstep (SIMT), running the tracing logic once for the whole group reduces the branching overhead and divergence costs.

**8. Map Placement Strategy:**
*   Placing eBPF maps entirely in CPU memory causes high latency when accessed by the GPU. The strategy involves:
    *   Placing maps in **GPU-local memory** for fast access by the GPU eBPF program.
    *   Creating **duplicate versions** or using a bidirectional sync mechanism so that the CPU can also access the data (e.g., for reporting) without incurring the high latency of remote memory access.

**9. Custom Eviction Policy:**
*   You would target the **GPU Driver (Linux Kernel)** layer. The driver is responsible for UVM memory management and page eviction decisions. By using eBPF hooks in the driver, you can intercept the eviction decision and replace the default algorithm with a custom one.

**10. TSG Lifecycle & User Space:**
*   The TSG lifecycle is managed by the driver, but the user-space CUDA runtime only sees high-level abstractions. TSG parameters (time slices, interleaving levels) are **not exposed** to user space. Exposing them via eBPF allows users to tune scheduling fairness and isolation for multi-tenant workloads, which is impossible through standard CUDA APIs.

**11. Critique of Safety:**
*   While eBPF is safer than kernel modules, running it on the GPU introduces new risks. The "per-warp" execution model means that a buggy eBPF program could cause **massive performance degradation** (divergence) or **memory corruption** if the verifier fails to catch a subtle GPU-specific memory access error. The "safety" is relative to system stability (crashes), not necessarily performance stability.

**12. Invisible vs. Inflexible:**
*   **Invisible** is arguably a greater barrier to *research* because you cannot debug what you cannot see. However, **Inflexible** is a greater barrier to *optimization* because even if you see the problem, you cannot change the policy without unsafe hacks. eBPF solves both by providing visibility *and* programmability.

**13. Long-term Sustainability:**
*   **Risk:** As GPU architectures evolve (e.g., disaggregated memory, new scheduling fabrics), the specific eBPF hooks may need to be rewritten.
*   **Benefit:** The *paradigm* of eBPF (programmable, safe, dynamic) is likely to persist. The hardware layer will change, but the need for dynamic, safe, user-defined policy hooks will remain. The challenge is keeping the eBPF verifier and the PTX injection tools up-to-date with new NVIDIA architectures.
