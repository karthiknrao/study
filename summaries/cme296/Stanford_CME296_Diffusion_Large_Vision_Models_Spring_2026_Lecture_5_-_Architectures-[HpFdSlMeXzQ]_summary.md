Here is your comprehensive study guide for **Lecture 5 of CME 296: Architectures Behind Image Generation Models**.

### 1. Executive Summary & Core Concepts

**Lecture Overview:**
This lecture marks the transition from deriving training losses (diffusion, flow matching, score matching) to examining the specific neural network architectures used to implement these generative processes. We move beyond the "black box" of the generation model to understand why specific components—such as U-Net structures, Convolutional layers, and Transformers—are chosen. The core thesis is that while U-Nets provide strong local detail preservation, they struggle with global structural consistency; consequently, the field is shifting toward **Diffusion Transformers (DiT)** and **Multi-Modal Diffusion Transformers (MMDiT)** to better handle global context and conditional generation. Finally, we explore how **Position Embeddings** (specifically RoPE) are optimized to provide spatial awareness to these models.

**Key Concepts Highlight:**
*   **Inductive Bias:** The assumption built into a model architecture that dictates how it processes data. For example, Convolutional Neural Networks (CNNs) have a spatial locality bias (pixels only see neighbors), whereas Transformers have no inherent locality bias (allowing global attention).
*   **U-Net Architecture:** A symmetric encoder-decoder structure using down-sampling (convolutions/pooling) and up-sampling (transpose convolutions). It uses skip connections to preserve local details while learning global structure, making it ideal for early diffusion models like DDPM.
*   **Self-Attention Mechanism:** A core component of Transformers where every element (token/patch) interacts with every other element to compute relevance. It allows the model to capture long-range dependencies (e.g., a mirror reflection matching the object).
*   **Patchification:** The process of dividing a latent image tensor into smaller, non-overlapping patches (e.g., $p \times p \times c$). These patches are treated as "tokens" in a Transformer, allowing the model to process images using sequence-based architectures.
*   **Adaptive Layer Normalization (AdaLN):** A technique for injecting conditioning signals (time step $t$ and text $C$) into a Transformer. It uses the condition to generate scale ($\gamma$), shift ($\beta$), and gate ($\alpha$) parameters that modulate the feature vectors, allowing the model to dynamically adjust its focus based on the noise level.
*   **MMDiT (Multi-Modal Diffusion Transformer):** The modern architecture (e.g., Stable Diffusion 3) that treats image patches and text tokens as a joint sequence. It uses **joint attention** (where text and image interact directly) rather than cross-attention, allowing for more nuanced generation (e.g., distinguishing "white walls" from "brown bear").
*   **RoPE (Rotary Position Embeddings):** A method for injecting position information directly into the Attention mechanism (Queries and Keys) rather than adding it to the input embeddings. It uses rotation matrices to encode relative positions, offering better generalization and spatial awareness in 2D image grids.

---

### 2. Deep Dive: Expanded Lecture Notes (The "Teaching" Section)

#### Concept 1: Inductive Bias and the U-Net
*   **Detailed Explanation:** When building a generation model, we want it to understand global structure (the whole image) and local details (textures). The **U-Net** was the dominant architecture for this because it mimics human visual scanning. It consists of an **Encoder** (down-sampling via convolutions and pooling) which reduces resolution to capture global context, and a **Decoder** (up-sampling via transpose convolutions) which restores resolution. Crucially, it uses **Skip Connections**: it copies feature maps from the encoder to the decoder at each corresponding level. This ensures that high-frequency local details (edges, textures) computed early in the down-sampling phase are not lost during the up-sampling phase.
*   **Context & Nuance:** The U-Net relies on **Convolutional operations**. A convolution uses a filter of size $F \times F \times C$ to scan the image. The "receptive field" (the area of the input a single output pixel can see) grows as more convolutional layers are stacked. To get a global view, you need many layers or aggressive down-sampling.
*   **Analogy:** Think of the U-Net like a painter who first sketches the big shapes (down-sampling/global view) and then adds fine brushstrokes (up-sampling/local details). The skip connections are like the painter keeping a reference photo (local details) on their desk while painting the fine details, ensuring they don't lose the texture of the fur or the edge of the book.
*   **Key Takeaway:** The U-Net is effective for local details but struggles when local details need to influence distant parts of the image (e.g., reflections), because convolutions are inherently localized.

#### Concept 2: The Shift to Transformers (DiT)
*   **Detailed Explanation:** To solve the "global structure" problem, we turn to **Transformers**. In a Vision Transformer (ViT), an image is **patchified** (cut into $p \times p$ blocks). These patches are linearly projected into vectors (tokens). The **Self-Attention** mechanism then allows every patch to interact with every other patch. This removes the locality bias. If a teddy bear is in the top left and a mirror is in the bottom right, the attention mechanism can directly link the texture of the bear to the reflection in the mirror.
*   **Context & Nuance:** The **Diffusion Transformer (DiT)** (2022) applies this to generation. Instead of predicting noise, it predicts the **velocity** (or mean/covariance) of the latent distribution. The model takes a noisy latent $x_t$, time $t$, and condition $C$ as input, and outputs the velocity vector to move the state closer to the clean image.
*   **Analogy:** If the U-Net is a painter working with a local brush, the Transformer is a committee of artists. Every artist (patch) looks at every other artist to decide how to paint their section. This ensures consistency across the whole canvas.
*   **Key Takeaway:** Transformers excel at global consistency because they allow direct, long-range interactions between any two parts of the image, overcoming the locality limitation of CNNs.

#### Concept 3: Conditional Injection via Adaptive Layer Norm (AdaLN)
*   **Detailed Explanation:** How do we tell the Transformer *when* it is in the diffusion process and *what* to generate? We use **AdaLN**. We take the time step $t$ (represented as sinusoidal embeddings) and the condition $C$ (text embedding), add them together, and pass them through an MLP. This produces three vectors: $\alpha$ (gate), $\gamma$ (scale), and $\beta$ (shift).
    *   **Scaling ($\gamma$):** Modifies the intensity of specific features.
    *   **Shifting ($\beta$):** Adjusts the baseline of the features.
    *   **Gating ($\alpha$):** Controls how much of the modulated feature flows into the next layer.
    *   **AdaLN-Zero:** At the start of training, $\alpha, \gamma, \beta$ are initialized to 0. This means the modulation is a no-op initially, allowing the model to learn basic structures before learning to apply conditions.
*   **Context & Nuance:** This is "global" modulation. It applies the same $\alpha, \gamma, \beta$ to *all* patches. This is efficient but has a limitation: if the prompt is "Brown bear on white walls," the model applies the same "brown" and "white" modulation to *every* pixel, which is incorrect (the bear should be brown, the walls white).
*   **Analogy:** Imagine a dimmer switch (gate) and a color filter (scale/shift) applied to the whole room. It’s useful for setting the "mood" (noise level), but it can't change the color of just one specific object in the room.
*   **Key Takeaway:** AdaLN is the standard way to inject time and text conditions into a DiT, but it treats all image patches identically, which becomes a bottleneck for complex spatial prompts.

#### Concept 4: MMDiT and Joint Attention
*   **Detailed Explanation:** To fix the "global modulation" limitation, modern models like **Stable Diffusion 3** use **MMDiT**. Instead of using AdaLN for text, they use **Joint Attention**.
    *   **Cross-Attention:** Image patches (Queries) look at Text tokens (Keys/Values). The text is treated as a static reference.
    *   **Joint Attention:** Image patches and Text tokens are concatenated into a single sequence. They attend to *each other*. The text tokens can also attend to the image patches.
*   **Context & Nuance:** This allows the model to learn that "white" in the text should only influence the "wall" patches, not the "bear" patches, because the attention weights can be spatially specific.
    *   **Single-Stream vs. Double-Stream:** In Single-Stream, text and image tokens use the same projection layers. In Double-Stream, they have separate projection layers (like different tools for a painter vs. a poet) before being combined in attention.
*   **Analogy:** Cross-Attention is like a painter reading a script. Joint Attention is like a painter and a poet in the same room, negotiating. The poet can say, "Actually, the wall should be brighter," and the painter adjusts, while the bear remains brown.
*   **Key Takeaway:** MMDiT with Joint Attention allows for finer-grained control over *where* specific text conditions apply in the image, solving the spatial ambiguity of global AdaLN.

#### Concept 5: Position Embeddings and RoPE
*   **Detailed Explanation:** Transformers are "permutation invariant"—they don't know the order of tokens. We must inject position information.
    *   **Absolute Position Embeddings:** Added to the input. The original Transformer used hardcoded sinusoidal functions. ViT used learned embeddings. DiT used hardcoded 2D embeddings.
    *   **RoPE (Rotary Position Embeddings):** Instead of adding position to the input, RoPE **rotates** the Query ($Q$) and Key ($K$) vectors inside the Attention mechanism. The rotation amount depends on the position index. The dot product of rotated vectors naturally encodes the *relative* distance between positions.
*   **Context & Nuance:** In 2D images, we need to handle X and Y coordinates.
    *   **Axial RoPE:** Treats X and Y axes separately (e.g., even indices for X, odd for Y).
    *   **Mixed RoPE:** Mixes X and Y rotations in the same vector to capture interactions between axes.
    *   **Centered Coordinates:** For high-resolution images, coordinates are centered at 0 (middle of the image) rather than starting at 0 (top-left). This helps the model understand "center" vs. "edge" regardless of resolution.
*   **Analogy:** Absolute embeddings are like putting a name tag on a person. RoPE is like knowing the *distance* between people based on how they are rotated relative to each other. RoPE is more flexible because it works even if the sequence length changes at inference time.
*   **Key Takeaway:** RoPE is the current state-of-the-art for position encoding because it injects position directly into the attention computation, allowing the model to understand relative spatial relationships (like "left of" or "above") more robustly.

---

### 3. Pathways for Further Exploration

1.  **The Topic/Concept:** **Classifier-Free Guidance (CFG) in MMDiT**
    *   **Why it Matters:** We mentioned conditioning, but how does the model actually "emphasize" the prompt? CFG involves predicting the image twice (conditional and unconditional) and interpolating.
    *   **Search/Study Direction:** Look into how CFG scales with the "guidance factor" ($w$) and why it can cause artifacts (oversaturation) if pushed too high.

2.  **The Topic/Concept:** **Flow Matching vs. DDPM**
    *   **Why it Matters:** The lecture mentioned predicting "velocity" (Flow Matching) vs. "noise" (DDPM). Understanding the mathematical difference helps in understanding why the loss functions differ.
    *   **Search/Study Direction:** Study the "vector field" interpretation of Flow Matching. Why is L2 regression on velocity considered "tractable" compared to score matching?

3.  **The Topic/Concept:** **Latent Space Geometry (VAE vs. DiVAE)**
    *   **Why it Matters:** We use a VAE to compress images into latents. The structure of this latent space dictates how well the Transformer can learn.
    *   **Search/Study Direction:** Investigate "Vector Quantized VAEs" (VQ-VAE) vs. continuous VAEs. How does quantizing the latent space affect the generative quality?

4.  **The Topic/Concept:** **FLOPs and Scaling Laws in Transformers**
    *   **Why it Matters:** The lecture noted that scaling parameters isn't enough; you must scale patch size too.
    *   **Search/Study Direction:** Look for papers on "Scaling Laws for Vision Transformers." Understand the trade-off between sequence length (number of patches) and computational cost (FLOPs).

5.  **The Topic/Concept:** **2D RoPE and Fourier Analysis**
    *   **Why it Matters:** The lecture mentioned "Axial" vs. "Mixed" RoPE and artifacts.
    *   **Search/Study Direction:** Study the "Fourier transform" interpretation of RoPE. Why does "Mixed RoPE" reduce artifacts compared to "Axial RoPE" when reconstructing spatial signals?

---

### 4. Comprehension & Review Questions

**Recall & Understanding**
1.  Define "Inductive Bias" in the context of the U-Net architecture.
2.  What is the primary difference between the "Encoder" and "Decoder" phases of a U-Net?
3.  In a Vision Transformer (ViT), what is the "Patchify" operation?
4.  What are the three parameters ($\alpha, \gamma, \beta$) generated by Adaptive Layer Norm (AdaLN), and what is their function?
5.  How does "AdaLN-Zero" differ from standard AdaLN during the initialization phase?
6.  What is the main structural difference between a U-Net and a Diffusion Transformer (DiT)?

**Application & Analysis**
7.  **Scenario:** You are generating an image of a "Red Apple on a White Plate." Using a standard DiT with AdaLN, explain why the model might struggle to distinguish the red color of the apple from the white color of the plate.
8.  **Scenario:** If you were to use a U-Net to generate a picture of a person looking into a mirror, what specific limitation of convolutional layers would likely cause the reflection to look inconsistent with the person?
9.  **Analysis:** In the context of RoPE, why is it beneficial to rotate Queries and Keys rather than simply adding a position vector to the input embeddings?
10. **Application:** You are designing a system to generate high-resolution images (e.g., 4K). Why is "Centered Coordinate" RoPE preferred over standard "Top-Left Origin" coordinates?
11. **Analysis:** Compare "Cross-Attention" and "Joint Attention" for injecting text conditions. Which one allows the text tokens to influence the *structure* of the image generation more dynamically, and why?

**Critical Thinking & Evaluation**
12. **Critique:** The lecture states that "there is no consensus" on how to inject conditions. Based on the limitations of AdaLN, argue why the industry is shifting toward MMDiT (Joint Attention) despite the increased computational cost.
13. **Evaluation:** Consider the trade-off between **Patch Size** and **Model Parameters** in DiTs. If you halve the patch size (increasing the number of tokens) but keep the model size constant, what happens to the computational complexity (FLOPs), and how does this affect the model's ability to capture fine details?
14. **Synthesis:** Synthesize the concepts of **Latent Space**, **Position Embedding**, and **Attention**. Why is it critical that position embeddings (like RoPE) are applied *within* the attention mechanism of a model operating in a *latent* space, rather than just being added to the input?

---

### Answer Key & Explanations

**1. Define "Inductive Bias" in the context of the U-Net architecture.**
*   **Answer:** Inductive bias is the assumption built into the model. For U-Nets, the bias is **spatial locality**—the assumption that neighboring pixels are more related than distant pixels. This is why convolutions (which look at local neighborhoods) are the core component.

**2. What is the primary difference between the "Encoder" and "Decoder" phases of a U-Net?**
*   **Answer:** The Encoder **down-samples** (reduces spatial dimensions) to capture global structure using pooling and convolutions. The Decoder **up-samples** (increases spatial dimensions) to restore resolution, using transpose convolutions and skip connections to recover local details.

**3. In a Vision Transformer (ViT), what is the "Patchify" operation?**
*   **Answer:** It is the process of dividing the input image (or latent tensor) into non-overlapping square patches (e.g., $16 \times 16$ pixels). These patches are then linearly projected into vectors to be treated as tokens.

**4. What are the three parameters ($\alpha, \gamma, \beta$) generated by AdaLN, and what is their function?**
*   **Answer:**
    *   $\alpha$ (Gate): Controls the flow/modulation strength.
    *   $\gamma$ (Scale): Scales the feature vectors.
    *   $\beta$ (Shift): Shifts the feature vectors (bias).
    *   These are derived from the time step and text condition to modulate the patch embeddings.

**5. How does "AdaLN-Zero" differ from standard AdaLN during the initialization phase?**
*   **Answer:** In AdaLN-Zero, $\alpha, \gamma, \beta$ are initialized to **0**. This means that at the start of training, the modulation is a "no-op" (does nothing), allowing the network to learn basic features before learning how to apply the conditions.

**6. What is the main structural difference between a U-Net and a Diffusion Transformer (DiT)?**
*   **Answer:** A U-Net relies on **convolutions** (local interactions) and skip connections. A DiT relies on **Self-Attention** (global interactions between all patches) and uses **AdaLN** to inject conditions, rather than skip connections for details.

**7. Scenario: Red Apple on White Plate (AdaLN limitation).**
*   **Answer:** AdaLN applies the same modulation ($\alpha, \gamma, \beta$) to *all* patches globally. It cannot selectively apply "red" modulation to the apple patches and "white" modulation to the plate patches simultaneously. It treats the image as a single context, making it hard to enforce spatially distinct colors based on text.

**8. Scenario: Mirror Reflection (U-Net limitation).**
*   **Answer:** Convolutional filters are local. To connect the person (top left) to the mirror (bottom right), information must pass through many layers. This "bottleneck" can lose specific texture details or consistency, leading to a reflection that doesn't perfectly match the original object.

**9. Analysis: RoPE vs. Additive Position Embeddings.**
*   **Answer:** RoPE injects position into the **Attention mechanism** (Q and K). This allows the model to compute **relative** positions (distance between tokens) directly during the dot product. Additive embeddings are static and added to the input, which can interfere with the semantic content of the token and doesn't inherently encode relative distance as cleanly.

**10. Application: High-Res Images and Centered Coordinates.**
*   **Answer:** In high-res images, the "center" of the image is semantically important (e.g., the subject). If coordinates start at 0,0 (top-left), the model struggles to distinguish "center" from "edge" as resolution scales. Centering coordinates at 0 (middle) makes the coordinate system resolution-independent and semantically meaningful.

**11. Analysis: Cross-Attention vs. Joint Attention.**
*   **Answer:** **Joint Attention** allows text tokens to interact with image tokens *and* text tokens with other text tokens. This is more dynamic because the text representation can adapt based on the image context (e.g., "white" means something different next to "wall" vs. "bear"). Cross-Attention treats text as a static reference.

**12. Critique: Why shift to MMDiT?**
*   **Answer:** While AdaLN is efficient, it fails at spatial nuance (as seen in the Apple/Plate example). MMDiT (Joint Attention) allows the model to learn *where* specific text tokens apply. Although computationally more expensive (longer sequences), the quality gain in spatial accuracy justifies the cost for high-end generation.

**13. Evaluation: Patch Size vs. Model Parameters.**
*   **Answer:** Halving the patch size quadruples the number of tokens (sequence length). Since Attention complexity is $O(N^2)$, halving the patch size **quadruples** the computational cost (FLOPs). If the model size stays constant, the model has more tokens to process but the same capacity, potentially leading to better detail capture but requiring much more compute.

**14. Synthesis: Latent Space, Position, and Attention.**
*   **Answer:** In Latent Space, dimensions are abstract. Without proper position encoding, the Transformer loses spatial structure. RoPE within Attention ensures that the model understands the *relative* spatial arrangement of latent patches. This is critical because the "latent" grid still has a 2D structure (height/width) that dictates how pixels map back to the real image. If position is ignored, the generated image may be a "soup" of features without coherent spatial layout.
