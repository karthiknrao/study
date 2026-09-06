Here is your comprehensive study guide based on the AMD Composable Kernel (CK) lecture.

### 1. Executive Summary & Core Concepts

**Lecture Overview:**
This lecture, presented by Hao Cong Wang from AMD, introduces **Composable Kernel (CK)**, AMD’s open-source, C++-templated programming paradigm designed to maximize GPU utilization for AI tensor operators. The core thesis is that traditional GPU programming is hindered by hardware complexity and rapid iteration; CK solves this by abstracting hardware details into reusable "tile-level" components and using coordinate transformations to handle complex data layouts (like convolutions) without explicit memory copying. The presentation demonstrates how CK achieves high performance through a "two-pillar" architecture: **Tile Programming** (for data movement and computation abstraction) and **Coordinate Transformation Primitives** (for algorithmic flexibility), exemplified by a highly optimized implementation of Flash Attention.

**Key Concepts Highlight:**
*   **Composable Kernel (CK):** An open-source, vendor-agnostic (in terms of API design, though AMD-specific in implementation) C++ library that breaks down AI kernels into reusable, composable modules. It aims to hide hardware complexity while allowing deep optimization.
*   **The "Two Pillars" of CK:** The fundamental architectural components of CK: **Tile Programming** (abstracting data layout and movement) and **Coordinate Transformation** (abstracting algorithmic logic).
*   **Tile Programming:** A programming model where data is described as 2D "tiles" (blocks) rather than individual elements. It uses "Tile Windows" (views of memory) and "Distributed Tensors" (data in registers) to manage data flow between HBM, Shared Memory, and Registers.
*   **Coordinate Transformation Primitives:** A set of API calls that virtually remap data coordinates (e.g., padding, merging dimensions, XOR for conflict avoidance) without physically copying data. This allows complex algorithms like convolutions to be expressed as simple matrix multiplications (GEMM).
*   **GEMM (General Matrix Multiply):** The foundational operation in AI. CK expresses most AI workloads (including convolutions and attention) as variations of GEMM, allowing developers to reuse highly optimized matrix multiplication pipelines.
*   **Flash Attention Implementation:** A specific case study showing how CK implements the forward pass of Flash Attention (a fused kernel involving two GEMMs and a Softmax) using CK’s abstractions, resulting in compact, high-performance code.
*   **HIP & ROCm:** The underlying software stack. CK is built on C++ and compiled via HIP CC (part of the ROCm platform), targeting AMD GPUs (MI series) rather than using OpenCL or vendor-specific assembly directly.

### 2. Deep Dive: Expanded Lecture Notes (The "Teaching" Section)

#### 1. The Challenge: Why GPU Utilization is Difficult
*   **Detailed Explanation:** Achieving high performance on modern GPUs (like the MI300 series) is difficult due to three main factors:
    1.  **Complexity:** The memory hierarchy is deep (Global Memory -> L2/L1 Cache -> Shared Memory -> Registers), and there are multiple compute units (Vector ALUs, DPPs, and Matrix/Tensor Cores).
    2.  **Hardware Iteration:** AMD releases new architectures annually (MI100, MI200, MI300, MI325/350). Kernels optimized for one generation may not be optimal for the next.
    3.  **Workload Diversity:** AI workloads range from standard CNNs to complex, fused attention kernels in LLMs. These are high-dimensional and irregular.
*   **Context & Nuance:** This sets the stage for CK. The lecture emphasizes that because hardware changes so rapidly, a "static" kernel approach (like writing raw assembly or using a rigid library) is unsustainable. We need a *programming model* that adapts to hardware changes.
*   **Analogy:** Imagine driving a car where the steering wheel layout changes every year. CK is like a universal adapter that lets you drive using standard controls, even when the car's internal mechanics (hardware) change.
*   **Key Takeaway:** High GPU utilization requires abstracting away hardware specifics to allow for rapid adaptation to new architectures and diverse AI workloads.

#### 2. Composable Kernel (CK) Architecture & Philosophy
*   **Detailed Explanation:** CK is a C++ templated library built on the philosophy of "Composability." It is structured in four levels:
    1.  **Client Apps:** The user’s application.
    2.  **Kernels/Invokers:** The actual kernel code.
    3.  **Templated Kernels:** Reusable, optimized building blocks.
    4.  **Basic Tile Operators:** Low-level hardware primitives.
    *   **Self-Sufficiency:** CK does not rely on backend libraries for its core logic; it is "self-sufficient."
    *   **Open Source:** Unlike some proprietary stacks, CK is open-source, allowing developers to inspect and modify the underlying optimizations.
*   **Context & Nuance:** CK originated in 2018 as a feature of "MI-Open" (AMD’s conv library). It evolved because monolithic kernels were hard to reuse. The key insight is that AI math boils down to **Dot Products, Reductions, Element-wise Ops, and Transformations**.
*   **Analogy:** Instead of building a new house from scratch every time (monolithic kernels), CK provides modular, prefabricated rooms (tiles) that you can snap together.
*   **Key Takeaway:** CK is a modular, C++-templated framework that treats AI kernels as compositions of reusable, optimized "tiles."

#### 3. Pillar 1: Coordinate Transformation Primitives
*   **Detailed Explanation:** This is CK’s method for handling complex data layouts. In a raw memory space, data is 1D. To do math, we need to map threads to data. CK uses "Tensor Views" to virtually transform this mapping.
    *   **Naive Tensor View:** The raw memory layout.
    *   **Transformations:** Functions like `make_padding`, `make_merge`, `make_xor`, and `make_embedded`.
    *   **Crucial Mechanism:** These transformations **do not copy data**. They only change the *address calculation* (stride/offset) logic.
    *   **Example (Convolution to GEMM):** A convolution (NHWC layout) is transformed into a 2D matrix (GEMM layout). The "padding" is handled by setting out-of-bounds reads to zero automatically, rather than physically padding the memory buffer.
*   **Context & Nuance:** This is the "magic" that allows CK to express complex operations (like Im2Col for convolutions) as simple GEMMs. The lecture highlights that the underlying buffer remains unchanged; only the *descriptor* (how we interpret the memory) changes.
*   **Analogy:** Think of a book. The physical pages (memory) don't change, but the "Coordinate Transformation" is like a new index or table of contents that lets you find page 50 by looking up "Chapter 3" instead of counting pages.
*   **Key Takeaway:** Coordinate transformations allow complex AI algorithms to be expressed as simple matrix operations by virtually remapping memory addresses, avoiding expensive data copies.

#### 4. Pillar 2: Tile Programming
*   **Detailed Explanation:** This pillar abstracts *how* data moves and is computed.
    *   **Tile Windows:** Describe how a block (tile) of data resides in Global Memory (HBM) or Shared Memory.
    *   **Distributed Tensors:** Describe data currently in registers (the fastest storage).
    *   **Pipelines:** A "Pipeline" is a sequence of operations (Load -> Compute -> Store) defined at the tile level.
    *   **Policies:** Optimization strategies injected into the pipeline (e.g., how to prefetch, how to shuffle data to avoid bank conflicts).
    *   **Goal:** Non-experts can write functional kernels using high-level tile descriptions, while experts can customize the "Policies" to squeeze out maximum performance.
*   **Context & Nuance:** Tile programming hides the "thread level" complexity. You don't manage individual threads; you manage *tiles* of data. The compiler/CK handles the mapping of threads to tiles.
*   **Analogy:** In tile programming, you don't tell each worker (thread) what to do; you tell the foreman (pipeline) to "move this pallet (tile) from the warehouse (HBM) to the assembly line (Registers)."
*   **Key Takeaway:** Tile programming decouples the *algorithm* (what math to do) from the *implementation* (how to move data efficiently), allowing for portable and high-performance code.

#### 5. Flash Attention Implementation in CK
*   **Detailed Explanation:** The lecture walks through a real-world example: the forward pass of Flash Attention.
    *   **Structure:** Flash Attention is a fused kernel involving two GEMMs (`Q*K` and `P*V`) and a Softmax reduction.
    *   **CK Implementation:**
        1.  Define Tile Windows for Q, K, V.
        2.  Use a `BlockGemmPipeline` for the first GEMM (`Q*K`).
        3.  Perform Softmax (reduction) on the result.
        4.  Use a second `BlockGemmPipeline` for (`P*V`).
        5.  Store the result.
    *   **Performance:** The CK implementation is noted as having "good performance" (competitive) but not necessarily "optimal" (which might require even deeper customization). It is highly compact (~200 lines of code for the core loop).
*   **Context & Nuance:** This demonstrates CK’s power in *Kernel Fusion*. By keeping intermediate results (like the attention scores) in registers/shared memory rather than writing them back to HBM, performance is boosted. The code maps directly to the mathematical steps in the Flash Attention paper.
*   **Analogy:** Instead of writing a letter, mailing it, and waiting for a reply (standard attention), CK allows you to hold the letter in your hand (registers), read it, and immediately write the reply (fused kernel).
*   **Key Takeaway:** CK enables the creation of fused, high-performance kernels like Flash Attention by composing GEMM pipelines and reductions without manual memory management.

#### 6. Kernel Customization & The "Green Block"
*   **Detailed Explanation:** CK is designed for customization.
    *   **Vendor-Optimized Kernels (Red Block):** AMD provides pre-optimized pipelines for common tasks (GEMM, Conv).
    *   **User Customization (Green Block):** Users can create their own pipelines or modify existing ones. For example, adding an epilogue (like ReLU or Quantization) to a GEMM.
    *   **Limitations:** Complex kernels are limited by hardware resources (e.g., 512 VGPRs/registers per thread, 64KB shared memory). If you exceed these, performance drops or compilation fails.
*   **Context & Nuance:** The "Green Block" allows users to take advantage of AMD’s low-level optimizations while adding their own logic. This is crucial for novel AI algorithms that haven't been pre-optimized by AMD yet.
*   **Analogy:** CK provides a high-performance engine (Vendor Kernel). The "Green Block" is the custom chassis you build around it to fit your specific vehicle (application).
*   **Key Takeaway:** CK balances out-of-the-box performance with the flexibility to customize kernels for specific, novel, or highly optimized use cases.

#### 7. Ecosystem, Tooling, and Comparison to Triton
*   **Detailed Explanation:**
    *   **ROCm/HIP:** CK uses HIP C++ (not OpenCL). It is compiled via `hipcc`.
    *   **CK Profiler:** A tool to exhaustively search for the best tile sizes and configurations for a specific problem size. It takes time (10-20 mins for specific ops, hours for all) but ensures optimal performance.
    *   **Triton Comparison:**
        *   **Triton:** Generalized, easy to write, JIT compiled, but often provides "good" not "optimal" performance.
        *   **CK:** Lower-level, C++ templated, AOT (Ahead-of-Time) or JIT, provides "optimal" performance.
        *   **Synergy:** CK and Triton are complementary. CK’s optimization techniques (like the ones used in Flash Attention) are shared with the Triton team to improve Triton’s performance on AMD hardware.
*   **Context & Nuance:** The lecture positions CK as the "expert" tool. If you want the absolute maximum performance on AMD hardware and are willing to write C++ templates, CK is the path. If you want rapid prototyping, Triton is better.
*   **Analogy:** Triton is a powerful sedan (easy to drive, good performance). CK is a race car (requires more skill/setup, but top speed is higher).
*   **Key Takeaway:** CK is the high-performance, customizable layer of the AMD AI stack, distinct from the more generalized Triton, though they share optimization insights.

### 3. Pathways for Further Exploration

1.  **Topic: The Im2Col Algorithm in CK**
    *   **Why it Matters:** The lecture mentioned that convolutions are unified with GEMM via data remapping. Understanding *how* this remapping works is key to mastering CK’s coordinate transformations.
    *   **Search/Study Direction:** Look for the "Im2Col" section in the CK GitHub documentation or the specific code examples for `make_embedded_transform`. Study how the 4D NHWC tensor is virtually mapped to a 2D matrix.

2.  **Topic: Tile Programming Concepts (Windows vs. Distributed Tensors)**
    *   **Why it Matters:** To write custom kernels, you must understand the difference between data in HBM (Tile Windows) and data in registers (Distributed Tensors).
    *   **Search/Study Direction:** Study the CK "Tile Programming" tutorial. Focus on the definitions of `BlockWindow`, `StaticDistributedTensor`, and `BlockGemmPipeline`.

3.  **Topic: Flash Attention Mathematical Derivation**
    *   **Why it Matters:** The CK implementation is a direct mapping of the Flash Attention paper. Understanding the math (why we fuse QK and PV) helps understand *why* the CK code looks the way it does.
    *   **Search/Study Direction:** Read the original "FlashAttention" paper by Tri Dao. Compare the pseudocode in the paper with the CK code lines presented in the lecture.

4.  **Topic: HIP and ROCm Programming Model**
    *   **Why it Matters:** CK is built on HIP. Understanding the underlying memory model (Global vs. Shared vs. Registers) in HIP is essential for debugging CK kernels.
    *   **Search/Study Direction:** Review the AMD ROCm documentation on "HIP Runtime API" and memory hierarchy. Specifically, look at how `hipMemcpy` and shared memory limits (64KB) work.

5.  **Topic: CK Profiler Usage**
    *   **Why it Matters:** The lecture highlighted that CK requires profiling to find optimal tile sizes. Knowing how to use this tool is a practical skill.
    *   **Search/Study Direction:** Look for the "CK Profiler" guide in the CK repository. Understand how to specify problem sizes, data types, and how to interpret the performance output.

6.  **Topic: Comparison: CK vs. CUTLASS (NVIDIA)**
    *   **Why it Matters:** The host mentioned CK is similar in spirit to NVIDIA’s CUTLASS. Comparing the two helps solidify the concepts of "templated kernel libraries."
    *   **Search/Study Direction:** Compare the high-level API of NVIDIA CUTLASS with AMD CK. Note the similarities in "Collective" vs. "Tile" abstractions.

7.  **Topic: Quantization in CK (FP8/INT8)**
    *   **Why it Matters:** The lecture noted that while FP16 is standard, quantization (FP8) requires more effort to optimize in CK because it disrupts the pure GEMM pipeline.
    *   **Search/Study Direction:** Look for CK examples involving "Quantized GEMM" or "FP8" kernels. Study how CK handles scale factors and different data types in the pipeline.

### 4. Comprehension & Review Questions

**Recall & Understanding**
1.  What are the "two fundamental pillars" of the Composable Kernel (CK) architecture?
2.  In CK, what is the difference between a "Tile Window" and a "Distributed Tensor"?
3.  What programming language and compiler are primarily used to build and compile CK kernels?
4.  According to the lecture, what are the three main reasons why high GPU utilization is challenging?
5.  What is the purpose of the "CK Profiler"?

**Application & Analysis**
6.  In the context of CK, how does "Coordinate Transformation" handle padding for a convolution operation without copying data?
7.  How does CK express a Convolution operation using its core abstractions? (Hint: What basic operation is it unified with?)
8.  You are implementing a custom AI layer that requires a GEMM followed immediately by a ReLU activation. How would you structure this using CK’s "Pipeline" and "Epilogue" concepts?
9.  Compare CK and Triton as discussed in the lecture. When would you choose CK over Triton, and vice versa?
10.  In the Flash Attention example, why is it important to keep intermediate results (like the attention scores) in registers/shared memory rather than writing them back to HBM?

**Critical Thinking & Evaluation**
11.  The lecture states that CK is "self-sufficient" and does not need backend libraries. Critique this design choice: What are the potential trade-offs of this approach compared to a system that relies on a dynamic runtime (like PyTorch’s eager execution)?
12.  The speaker mentioned that complex kernels are limited by hardware resources (e.g., 512 VGPRs). How might this limitation affect the design of a very large attention kernel compared to a standard GEMM?
13.  Considering the rapid iteration of AMD hardware (MI100 to MI300), how does CK’s "Tile Programming" abstraction mitigate the risk of code becoming obsolete? Provide a specific example from the lecture.

***

**Answer Key & Explanations**

1.  **Recall:** The two pillars are **Tile Programming** (for data movement/computation abstraction) and **Coordinate Transformation Primitives** (for algorithmic logic/data mapping).
2.  **Recall:** A **Tile Window** describes how a block of data is laid out in memory (HBM or Shared Memory). A **Distributed Tensor** describes data currently residing in registers (the fastest storage for computation).
3.  **Recall:** CK is written in **C++** (using templates) and compiled using **HIP CC** (part of the ROCm stack). It does *not* use OpenCL.
4.  **Recall:** The three reasons are: (1) Complexity of the GPU programming model (memory hierarchy, multiple compute units), (2) Rapid hardware iteration (annual architecture changes), and (3) The diversity and complexity of AI workloads (custom algorithms, fused kernels).
5.  **Recall:** The CK Profiler is a tool used to exhaustively search for the optimal configuration (tile sizes, layouts, etc.) for a specific kernel problem to ensure high performance.
6.  **Application:** CK uses "Coordinate Transformation" to virtually map memory addresses. When a thread reads an address that falls into the "padding" region, the CK runtime automatically returns **zero** for that value, rather than requiring the user to physically write zeros into the memory buffer.
7.  **Application:** CK unifies convolution with **GEMM (General Matrix Multiply)**. It uses coordinate transformations (specifically `make_embedded_transform` or Im2Col) to remap the 4D input tensor (NHWC) into a 2D matrix format that can be processed by highly optimized GEMM kernels.
8.  **Application:** You would define a `BlockGemmPipeline` for the matrix multiplication. Then, you would attach an **Epilogue** (or element-wise operation) to the pipeline’s output. This ensures the ReLU is applied immediately after the GEMM results are computed, keeping the data in registers/shared memory and avoiding an extra memory write/read cycle.
9.  **Application:** Choose **CK** when you need maximum performance on AMD hardware and are willing to write C++ code and manage tile configurations. Choose **Triton** when you need rapid prototyping and general-purpose GPU programming, accepting that performance may be "good" but not "optimal."
10. **Analysis:** Keeping intermediate results in fast on-chip memory (registers/shared memory) reduces **HBM (Global Memory) traffic**. In Flash Attention, writing the attention scores back to HBM and reading them again would significantly increase latency and bandwidth usage. Fusing the operations keeps the data hot in the cache/registers.
11. **Critical Thinking:** *Trade-off:* "Self-sufficiency" means CK is highly optimized and portable across software stacks, but it requires the developer to manage more low-level details (tile sizes, memory layouts). It shifts complexity from the framework (like PyTorch) to the kernel developer. However, it ensures that the kernel is not bottlenecked by a generic runtime, allowing for vendor-specific optimizations that a generic backend might miss.
12. **Critical Thinking:** In a large attention kernel, the "attention scores" matrix (Q*K) can be very large. If it exceeds the available **VGPRs (registers)** or **Shared Memory** (64KB), the kernel will either spill to slower memory (hurting performance) or fail to compile. This forces the developer to use "tiling" strategies (processing smaller chunks of the sequence at a time) to fit within hardware limits, which is exactly what Flash Attention does.
13. **Critical Thinking:** CK’s abstraction allows hardware-specific optimizations (like specific register shuffling or cache hinting) to be encapsulated in "Policies." When hardware changes (e.g., from MI200 to MI300), the *interface* of the Tile Programming remains the same, but the *internal policy* (the vendor-optimized code) can be updated by AMD. The user’s code (the "Green Block") remains valid because it relies on the stable Tile API, not the raw hardware registers.
