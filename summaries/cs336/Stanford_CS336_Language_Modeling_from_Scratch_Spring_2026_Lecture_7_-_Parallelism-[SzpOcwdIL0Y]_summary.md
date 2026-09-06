Welcome to the masterclass on **Parallelism and Distributed Training**. In the previous lecture, we optimized a single GPU by writing efficient kernels. Today, we scale up. We are moving from the "inside the box" view (single GPU memory hierarchy) to the "outside the box" view (how multiple GPUs talk to each other).

The central thesis of this lecture is that while it is easy to throw more hardware at a problem, **orchestrating computation to avoid data transfer bottlenecks** is the true challenge. We will break this down into the hardware topology, the communication primitives (collective operations), and the three primary strategies for parallelizing training: Data Parallelism, Tensor Parallelism, and Pipeline Parallelism.

---

### 1. Executive Summary & Core Concepts

**Lecture Overview:**
This lecture introduces the fundamentals of multi-GPU training, explaining why we scale out (memory capacity and speed) and how we do it. It details the hardware connectivity hierarchy (NVLink vs. InfiniBand vs. Ethernet) and introduces **Collective Operations**—the standard primitives for distributed communication. Finally, it demonstrates three parallelization strategies: **Data Parallelism** (splitting the batch), **Tensor Parallelism** (splitting the matrix dimensions), and **Pipeline Parallelism** (splitting the layers), highlighting the trade-offs in communication overhead for each.

**Key Concepts Highlight:**
*   **Collective Operations:** Standardized communication patterns (e.g., AllReduce, AllGather) that allow multiple devices to exchange data without managing point-to-point routing manually. These are the "verbs" of distributed computing.
*   **NVLink & NVSwitch:** High-bandwidth, low-latency interconnects used within a node (typically 8 GPUs). They are significantly faster than traditional PCIe or Ethernet, making them ideal for Tensor Parallelism.
*   **InfiniBand & Ethernet:** Lower-bandwidth interconnects used between nodes. InfiniBand supports RDMA (bypassing the CPU), while standard Ethernet is slower but cheaper.
*   **Data Parallelism (DDP):** The strategy of replicating the entire model on every GPU and splitting the *data* (batch) across GPUs. Gradients are then synchronized (averaged) via AllReduce.
*   **Tensor Parallelism:** The strategy of splitting the *parameters* (matrix columns/rows) across GPUs within a single layer. This requires constant communication of activations and is best suited for high-bandwidth intra-node links.
*   **Pipeline Parallelism:** The strategy of splitting the *layers* of the model across GPUs. Data flows sequentially from one GPU to the next, requiring careful management of "pipeline bubbles."
*   **RDMA (Remote Direct Memory Access):** A networking technology that allows a GPU to read/write directly to another GPU’s memory without involving the CPU, reducing latency significantly.
*   **Pipeline Bubbles:** Periods of inactivity in pipeline parallelism where a GPU is waiting for data to arrive from the previous stage, reducing overall throughput.

---

### 2. Deep Dive: Expanded Lecture Notes

#### 1. The Hardware Hierarchy & Connectivity
**Detailed Explanation:**
To understand parallelism, we must first understand the physical constraints. In a single GPU, we worried about HBM (High Bandwidth Memory) being slow relative to L1/L2 cache. In a multi-GPU system, the "distance" between compute units becomes the primary bottleneck.
*   **Intra-Node (8 GPUs):** Connected via **NVLink** and routed through an **NVSwitch**. This provides massive bandwidth (e.g., ~1.8 TB/s for NVLink 5).
*   **Inter-Node:** Connected via **InfiniBand** (expensive, high performance) or **Ethernet** (cheaper, slower).
*   **The CPU Bottleneck:** Traditional Ethernet requires data to pass through the CPU’s kernel socket buffer, adding latency. **RDMA** solves this by allowing GPU-to-GPU direct memory access. NVIDIA’s NVLink and InfiniBand support this; standard Ethernet does not (though RoCE is an emerging alternative).

**Context & Nuance:**
The hierarchy of speed is strictly maintained: Shared Memory/L1 > HBM > NVLink (Intra-node) > InfiniBand (Inter-node) > Ethernet. The choice of parallelism strategy depends heavily on this hierarchy. For example, you would never use Tensor Parallelism across nodes (InfiniBand) because the communication overhead would destroy performance. You *would* use it within a node (NVLink).

**Analogy:**
Think of a company.
*   **L1/L2/HBM:** The desk where you work (fast access).
*   **NVLink:** The hallway connecting your office to your colleague’s office (fast, direct).
*   **InfiniBand/Ethernet:** The phone line or email to a branch office in another city (slower, requires protocols).
*   **CPU Bottleneck:** If you don’t have a direct line (RDMA), you have to hand the message to a secretary (CPU) who types it up and mails it. RDMA is like having a direct secure line.

**Key Takeaway:**
Communication speed drops drastically as you move from intra-node (NVLink) to inter-node (InfiniBand/Ethernet), dictating which parallelization strategies are viable.

#### 2. Collective Operations (The Communication Primitives)
**Detailed Explanation:**
Instead of manually coding "GPU 0 sends to GPU 1," we use **Collective Operations**. These are templates for data movement.
*   **Broadcast:** One rank sends data to *all* ranks. (Used for initialization).
*   **Scatter:** One rank splits its data, sending chunk $i$ to rank $i$.
*   **Gather:** All ranks send their chunks to one specific rank (usually Rank 0).
*   **Reduce:** All ranks send data to one rank, which applies an operation (e.g., Sum) and returns the result.
*   **AllGather:** Every rank ends up with the *full* concatenated data from all ranks.
*   **ReduceScatter:** Each rank receives a *reduced* portion of the total data (e.g., Rank 0 gets the sum of the first chunk, Rank 1 gets the sum of the second).
*   **AllReduce:** The most critical operation. It combines "Reduce" and "AllGather." Every rank ends up with the *reduced* result (e.g., the average gradient).
*   **All-to-All:** The most general operation. Each rank sends specific chunks to specific other ranks. Crucial for MoE (Mixture of Experts) routing.

**Context & Nuance:**
In PyTorch, these are abstracted. `torch.distributed` handles the backend (NCCL for GPUs, Gloo for CPUs). **NCCL** (NVIDIA Collective Communications Library) is the engine that figures out the topology (ring vs. tree) and launches the actual CUDA kernels for communication.

**Analogy:**
*   **Reduce:** Everyone in a meeting sends their vote to the manager; the manager tallies it and announces the winner.
*   **AllReduce:** Everyone in the meeting sends their vote to everyone else; everyone tallies it and everyone knows the winner.
*   **AllGather:** Everyone in the meeting shares their notes, and everyone ends up with a copy of *all* the notes.

**Key Takeaway:**
**AllReduce** is the workhorse of Data Parallelism; **AllGather** and **ReduceScatter** are the building blocks for advanced strategies like ZeRO and FSDP.

#### 3. Data Parallelism (DDP)
**Detailed Explanation:**
This is the simplest form of parallelism.
1.  **Replicate:** Every GPU holds a *full copy* of the model parameters.
2.  **Shard Data:** The input batch is split. GPU 0 processes rows 0-31, GPU 1 processes rows 32-63, etc.
3.  **Compute:** Each GPU calculates forward/backward passes on its local data.
4.  **Synchronize:** Because the data was different, the gradients are different. We perform an **AllReduce** (Sum/Average) on the gradients.
5.  **Update:** All GPUs now have identical, averaged gradients and update their parameters identically.

**Context & Nuance:**
DDP is elegant because it requires minimal code changes (just adding the gradient sync step). However, it is memory-inefficient because every GPU stores the full model. If the model doesn't fit on one GPU, DDP fails.

**Analogy:**
Four chefs (GPUs) each have a full recipe book (Model). They each cook a different part of the meal (Data). After tasting their portion, they all call in their feedback (Gradients). They average the feedback and update the recipe book identically.

**Key Takeaway:**
DDP trades memory (storing full models) for simplicity and speed, relying on **AllReduce** to keep parameters synchronized.

#### 4. Tensor Parallelism
**Detailed Explanation:**
Here, we split the *weights* of the model across GPUs, usually within a single node.
*   **Column Parallelism:** We split the matrix $W$ into columns $W_1, W_2, ...$.
*   **Forward Pass:** Each GPU computes $x \cdot W_i$. The result is a partial activation.
*   **Communication:** We must **AllGather** these partial activations to get the full output for the next layer.
*   **Backward Pass:** Gradients are **ReduceScattered**.

**Context & Nuance:**
Tensor Parallelism requires *extremely* high bandwidth because you are communicating large activation tensors at every layer. It is strictly an intra-node strategy (NVLink). If you try this across InfiniBand, the network becomes the bottleneck.

**Analogy:**
Instead of four chefs having the whole recipe, each chef only knows *part* of the recipe (e.g., Chef A knows the sauce, Chef B knows the protein). They must constantly pass ingredients (activations) back and forth to finish the dish.

**Key Takeaway:**
Tensor Parallelism splits the *width* of the computation. It is communication-heavy and requires the fastest interconnects (NVLink).

#### 5. Pipeline Parallelism
**Detailed Explanation:**
Here, we split the model by *layers*.
*   **Stage 1 (GPU 0):** Processes Layers 1-4.
*   **Stage 2 (GPU 1):** Processes Layers 5-8.
*   **Flow:** Data moves like an assembly line. GPU 0 sends activations to GPU 1.
*   **Micro-Batches:** To hide latency, we split the batch into "micro-batches." While GPU 1 processes micro-batch 1, GPU 0 can start processing micro-batch 2. This reduces **Pipeline Bubbles** (idle time).

**Context & Nuance:**
Pipeline Parallelism is more tolerant of slower interconnects (InfiniBand/Ethernet) because it only sends the activations at the *boundaries* of the pipeline stages, not at every layer. However, it suffers from "bubbles" where GPUs wait for data.

**Analogy:**
An assembly line. Station 1 puts the wheels on, Station 2 puts the engine in. Station 1 waits for Station 2 to finish the car before it can start the next one (unless we use micro-batches to keep the line moving).

**Key Takeaway:**
Pipeline Parallelism splits the *depth* of the model. It is suitable for multi-node training but requires careful management of micro-batches to avoid idle time.

---

### 3. Pathways for Further Exploration

1.  **The Topic/Concept:** **ZeRO (Zero Redundancy Optimizer) & FSDP**
    *   **Why it Matters:** The lecture noted that DDP requires storing the full model on every GPU. ZeRO and FSDP (Fully Sharded Data Parallel) solve this by sharding parameters, gradients, and optimizer states across GPUs, allowing models *larger* than a single GPU's memory to be trained.
    *   **Search/Study Direction:** Look into the "ZeRO-1, ZeRO-2, ZeRO-3" papers. Study how they replace the simple `AllReduce` of DDP with `ReduceScatter` and `AllGather` to save memory.

2.  **The Topic/Concept:** **Mixture of Experts (MoE) & All-to-All Communication**
    *   **Why it Matters:** The lecture mentioned `All-to-All` is critical for MoE. Understanding this is key to modern sparse models.
    *   **Search/Study Direction:** Study how "routing" works in MoE architectures. Look for implementations of `torch.distributed.all_to_all` and how load balancing prevents certain "experts" from being overwhelmed.

3.  **The Topic/Concept:** **NCCL Topologies (Ring vs. Tree)**
    *   **Why it Matters:** We mentioned NCCL handles the low-level packets. Understanding *how* it routes data (Ring AllReduce vs. Tree) helps in debugging network bottlenecks.
    *   **Search/Study Direction:** Search for "NCCL Ring Algorithm vs. Tree Algorithm." Understand why Ring is often preferred for AllReduce in modern high-bandwidth clusters.

4.  **The Topic/Concept:** **RoCE (RDMA over Converged Ethernet)**
    *   **Why it Matters:** InfiniBand is expensive. RoCE brings RDMA capabilities to standard Ethernet, changing the economics of large-scale training.
    *   **Search/Study Direction:** Investigate Meta’s papers on training LLMs using RoCE. Compare the latency/throughput of RoCE vs. InfiniBand.

5.  **The Topic/Concept:** **Sequence Parallelism**
    *   **Why it Matters:** The lecture mentioned this as a future topic. It splits the *sequence length* (the time dimension) rather than the batch or layers, which is crucial for very long context windows.
    *   **Search/Study Direction:** Look for "Ring Attention" or "Sequence Parallelism" papers. Understand how it differs from Tensor Parallelism (which splits the feature dimension).

6.  **The Topic/Concept:** **Overlapping Communication and Computation**
    *   **Why it Matters:** The lecture emphasized that async operations are key. In advanced training, you don't just "do the math" then "send the data." You send data *while* doing the math.
    *   **Search/Study Direction:** Study "Asynchronous Gradient Computation" and how frameworks like Horovod or DDP handle overlapping the backward pass communication with the next forward pass.

---

### 4. Comprehension & Review Questions

#### Recall & Understanding
1.  What is the primary difference between **NVLink** and **InfiniBand** in the context of GPU connectivity?
2.  Define the **AllReduce** operation. What are its two constituent steps?
3.  In **Data Parallelism**, what specific data is synchronized across GPUs after the backward pass?
4.  What is **RDMA**, and why is it important for high-performance distributed training?
5.  In **Tensor Parallelism**, how are the model parameters split across GPUs?

#### Application & Analysis
6.  You have a model that is 100GB in size. You have 4 GPUs, each with 24GB of HBM. Can you use standard Data Parallelism (DDP)? Why or why not?
7.  If you are using **Tensor Parallelism**, why is it generally restricted to GPUs within the same node (connected via NVLink)?
8.  In **Pipeline Parallelism**, what is the function of "micro-batches" and how do they mitigate "pipeline bubbles"?
9.  You are training a model across two data centers connected by standard Ethernet. Which parallelism strategy (Data, Tensor, or Pipeline) is most appropriate, and why?
10. In a standard DDP setup, if the batch size is not a multiple of the world size, what is a common technique to handle this?

#### Critical Thinking & Evaluation
11. **Critique:** Data Parallelism replicates the entire model on every GPU. Argue for and against this approach in the context of a trillion-parameter model that does *not* fit on a single GPU.
12. **Synthesis:** Explain the trade-off between **communication overhead** and **memory efficiency** when choosing between Tensor Parallelism and Pipeline Parallelism. Which one would you choose for a 100-node cluster with moderate inter-node bandwidth?
13. **Evaluation:** The lecture states that "it is easy to use a ton of GPUs, but hard to use them effectively." Based on the concepts of **collective operations** and **hierarchical memory**, identify the two biggest factors that determine whether a parallel training run will be efficient or wasted compute.

---

### Answer Key & Explanations

**1. NVLink vs. InfiniBand:**
NVLink is an intra-node interconnect (connecting 8 GPUs on a single node) with very high bandwidth (TB/s). InfiniBand is an inter-node interconnect (connecting different servers) with lower bandwidth (typically 100s of GB/s).

**2. AllReduce:**
AllReduce is a collective operation that reduces data (e.g., sums gradients) across all ranks and then broadcasts the result back to *all* ranks. It is conceptually equivalent to `ReduceScatter` + `AllGather`.

**3. DDP Synchronization:**
The **gradients** are synchronized. Each GPU computes gradients on its local data shard, and these gradients are averaged (via AllReduce) so all GPUs update their parameters identically.

**4. RDMA:**
Remote Direct Memory Access allows a GPU to read/write directly to another GPU’s memory without involving the CPU. This bypasses the CPU’s kernel buffer, significantly reducing latency and CPU overhead.

**5. Tensor Parallelism Split:**
Parameters are split by **columns** (or rows). For a matrix $W$, GPU 0 might hold $W_{0..N}$ and GPU 1 holds $W_{N..2N}$. Each GPU computes a partial matrix multiplication.

**6. Model Fit:**
No. In DDP, *each* GPU must hold the full model. A 100GB model cannot fit on a 24GB GPU. You would need a sharding strategy like ZeRO or FSDP, or Tensor/Pipeline parallelism to distribute the weights.

**7. Tensor Parallelism Restriction:**
Tensor Parallelism requires communication at *every* layer. This generates massive traffic. Only the high-bandwidth, low-latency links of NVLink (intra-node) can handle this without becoming a severe bottleneck. InfiniBand is too slow for this frequency.

**8. Micro-batches:**
Micro-batches split the batch into smaller chunks. This allows the pipeline to be "filled" so that while one GPU is processing a chunk, the next GPU is already working on the previous chunk, reducing the idle time (bubbles) in the pipeline.

**9. Multi-Data-Center Strategy:**
**Pipeline Parallelism** is most appropriate. It requires less communication (only at stage boundaries) and can tolerate the higher latency of Ethernet. Tensor Parallelism would be too slow, and DDP might be limited if the model doesn't fit on one node.

**10. Batch Size Handling:**
You can **pad** the data with zeros or dummy samples to make the batch size a multiple of the world size, ensuring equal distribution.

**11. DDP Critique:**
*   *Against:* It wastes memory. If the model is too big for one GPU, DDP is impossible. It replicates memory unnecessarily.
*   *For:* It is simple to implement and highly efficient for small/medium models where the model fits in memory. It maximizes throughput by keeping all GPUs busy with data.

**12. Communication vs. Memory Trade-off:**
*   **Tensor Parallelism:** High communication (at every layer), Low memory per GPU (weights are sharded). Best for high-bandwidth (NVLink).
*   **Pipeline Parallelism:** Low communication (only at boundaries), High memory per GPU (each GPU holds full layers). Best for lower-bandwidth (InfiniBand) or multi-node setups.

**13. Efficiency Factors:**
1.  **Bandwidth Hierarchy:** Are you using the right parallelism for the hardware? (e.g., don't use Tensor Parallelism over Ethernet).
2.  **Overlap:** Are you overlapping communication and computation? If you wait for data to finish sending before computing, you waste time. The key is asynchronous operation.
