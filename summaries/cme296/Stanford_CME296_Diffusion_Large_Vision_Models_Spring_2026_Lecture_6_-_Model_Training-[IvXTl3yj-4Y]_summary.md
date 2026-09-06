Here is a comprehensive study guide based on Lecture 6 of CME296, structured to help you master the practical aspects of training text-to-image generation models.

### 1. Executive Summary & Core Concepts

**Lecture Overview:**
This lecture bridges the gap between theoretical architectures (UNet, DiT) and the practical lifecycle of deploying a text-to-image model. It outlines the three primary stages of the training lifecycle: **Pre-training** (learning to generate images), **Post-training** (learning to generate *good* images via alignment and fine-tuning), and **Tuning/Distillation** (optimizing for specific use cases or production efficiency). The lecture details critical training techniques such as the **Logit Normal distribution** for timestep sampling, **REPA** for accelerated representation learning, **Reward Feedback Learning** for preference alignment, and **Progressive Distillation** to reduce inference latency.

**Key Concepts Highlight:**
*   **The Training Lifecycle:** The structured progression from Pre-training (basic image generation) to Post-training (quality/alignment) to Tuning (personalization/efficiency).
*   **Logit Normal Timestep Sampling:** A sampling strategy where time steps $t$ are drawn from a Logit Normal distribution rather than a Uniform distribution, emphasizing "hard" middle steps to improve training stability.
*   **Resolution-Dependent Noise Shifting:** A mathematical adjustment where the noise level $t$ is rescaled based on image resolution to account for the fact that low-resolution images perceive the same noise level as "noisier" due to fewer spatial correlations.
*   **REPA (Representation Alignment):** A training trick that couples the diffusion loss with a loss term aligning the DiT’s internal representations with a pre-trained encoder (like a Vision Transformer), significantly speeding up convergence.
*   **Reward Feedback Learning (RFL):** A post-training method where a trained reward model scores generated images, and the generation model is optimized to maximize this reward, aligning outputs with human preferences.
*   **Flow GRPO & Diffusion DPO:** Preference optimization techniques adapted from LLMs. Flow GRPO uses group relative rewards, while Diffusion DPO directly optimizes the policy to prefer winning images over losing ones.
*   **Progressive Distillation:** A technique to reduce the number of sampling steps (e.g., from 1000 to 1) by iteratively halving the steps, forcing the student model to match the teacher’s output over a progressively smaller interval.

---

### 2. Deep Dive: Expanded Lecture Notes

#### Concept 1: The Training Lifecycle
*   **Detailed Explanation:** The lecture defines a clear pipeline for developing a production-ready model.
    1.  **Pre-training:** The model learns the distribution of images. It is compute-intensive and focuses on "how to generate images" (any images).
    2.  **Post-training:** The model learns "how to generate *good* images." This involves **Continued Training** (adding new knowledge/domains) and **Supervised Fine-Tuning** (improing aesthetics/lighting).
    3.  **Tuning/Distillation:** Personalizing the model (e.g., specific characters) or making it efficient for production (fewer steps).
*   **Context & Nuance:** Pre-training is the foundation; without it, the model is useless. Post-training refines the *behavior* and *quality* of the model. The distinction between "knowledge" (what the model knows) and "behavior" (how it generates) is crucial for post-training strategies.
*   **Analogy:** Think of a chef. Pre-training is learning basic knife skills and cooking techniques. Post-training is learning specific recipes and plating aesthetics. Tuning is becoming a specialist (e.g., only making sushi) or learning to cook faster for a busy restaurant.
*   **Key Takeaway:** A model is not "done" after pre-training; it requires iterative refinement through post-training and specific tuning to meet production standards.

#### Concept 2: Timestep Sampling & The Logit Normal Distribution
*   **Detailed Explanation:** In Flow Matching/Diffusion, we sample a timestep $t$ to determine how much noise is added.
    *   **The Problem:** If we sample $t$ uniformly, we treat "easy" tasks (low noise, $t \approx 1$) and "hard" tasks (high noise, $t \approx 0$) equally. However, the "hard" part of the generation is the middle range ($t \approx 0.5$), where the model must decide on structure.
    *   **The Solution:** Use the **Logit Normal distribution**. We define $t$ such that $\text{logit}(t) = \log(\frac{t}{1-t})$. We sample a variable $x$ from a Normal distribution, and $t = \sigma(x)$ (sigmoid). This concentrates samples in the middle, where the task is hardest.
*   **Context & Nuance:** The lecture explains that $t=0$ (pure noise) is "easy" to predict because the model just predicts the mean of the target distribution. $t=1$ (clean) is "easy" because it’s almost done. The middle is "hard" because the model must resolve ambiguity (e.g., placing eyes on a teddy bear).
*   **Analogy:** Imagine learning to drive. The hardest part isn't starting (easy) or parking (easy); it's merging onto the highway (hard). You should practice merging more than you practice starting the car.
*   **Key Takeaway:** To train robustly, we must oversample the "hard" timesteps (mid-range noise) using a Logit Normal distribution, not a Uniform one.

#### Concept 3: Resolution-Dependent Noise Shifting
*   **Detailed Explanation:** Perceived noise is not absolute; it depends on resolution. A low-resolution image looks "noisier" than a high-resolution one at the same noise level $t$ because spatial correlations (neighbors) are lost more quickly in low-res images.
    *   **The Math:** We want the uncertainty in estimating the clean pixel value $C$ to be consistent across resolutions.
    *   **Derivation:** By averaging pixel values, the standard deviation of the estimate scales with $\frac{1}{\sqrt{N}}$ (where $N$ is the number of pixels/resolution).
    *   **The Shift:** To match perceived noise, we adjust the timestep $t$ for higher resolutions. If you have a low-res timestep $t_n$, the corresponding high-res timestep $t_m$ must be shifted to account for the reduced uncertainty in high-res images.
*   **Context & Nuance:** This is critical for multi-resolution training. If you don't shift $t$, the model treats a low-res noisy image and a high-res noisy image as equally "hard," which is perceptually incorrect.
*   **Analogy:** Blurring a photo. A small blur on a high-res photo looks subtle. The same blur on a low-res photo looks devastating. You need to adjust the "blur amount" based on the image size to keep the visual impact constant.
*   **Key Takeaway:** Noise levels must be rescaled based on resolution ($1/\sqrt{N}$) so that the model perceives consistent difficulty across different image sizes.

#### Concept 4: REPA (Representation Alignment)
*   **Detailed Explanation:** REPA accelerates training by leveraging the knowledge of a pre-trained encoder (e.g., a Vision Transformer).
    *   **Mechanism:** The DiT model predicts a vector field. REPA adds a loss term that forces the internal representations (embeddings) of the DiT to be similar (via projection) to the representations of a pre-trained encoder.
    *   **Why Early Layers?** The lecture notes that aligning *early* layers yields better results. Early layers capture semantic structure, while later layers capture local details. Pre-trained encoders are excellent at semantic representation.
*   **Context & Nuance:** This acts as a "shortcut." Instead of the DiT learning from scratch how to represent a "teddy bear," it is nudged to look like the representation a powerful pre-trained model has already learned.
*   **Analogy:** Learning a language. Instead of learning grammar from scratch, you use a dictionary (pre-trained encoder) to understand the meaning of words (semantics) faster.
*   **Key Takeaway:** REPA significantly speeds up training (up to 18x) by aligning DiT internal states with pre-trained encoder features, particularly in early layers.

#### Concept 5: Post-Training & Preference Alignment
*   **Detailed Explanation:** Post-training focuses on "good" images. Two main approaches:
    1.  **Reward Feedback Learning (RFL):** Train a Reward Model (using Bradley-Terry loss on pairwise human ratings) to score images. Then, optimize the generation model to maximize this reward. The reward model is differentiable, allowing backpropagation.
    2.  **Flow GRPO & Diffusion DPO:**
        *   **Flow GRPO:** Generates a *group* of diverse images for a prompt. Computes relative rewards within the group. Updates the policy based on "advantage" (how much better an image is than the group average). Includes a KL divergence term to prevent "reward hacking" (optimizing the metric but losing quality).
        *   **Diffusion DPO:** Directly optimizes the flow matching loss to predict velocities for *winning* images and *worse* velocities for *losing* images.
*   **Context & Nuance:** These methods bridge the gap between "in-distribution" prompts (long, detailed, like "a warm intimate indoor scene...") and user inputs ("teddy bear reading"). **Prompt Enhancement** is often needed to translate user intent into the complex prompts the model was tuned on.
*   **Analogy:** The "Food Inspector." In RFL, an inspector tastes your dish and says "this is better than that." In DPO/GRPO, you compare multiple dishes and refine your recipe to consistently produce the "best" one.
*   **Key Takeaway:** Post-training aligns the model with human preferences using reward models and preference optimization, ensuring outputs are not just valid, but *good*.

#### Concept 6: Personalization (DreamBooth & LoRA)
*   **Detailed Explanation:**
    *   **DreamBooth:** Personalizes the model to a specific subject (e.g., *your* specific teddy bear). You train on a few images of the subject using a special token (e.g., `<class>`).
    *   **The Problem:** Overfitting. The model forgets how to generate other things.
    *   **The Solution:** **Prior Preservation Loss.** You add a loss term that ensures the model continues to generate high-quality images for *other* prompts (reference text-image pairs).
    *   **LoRA (Low-Rank Adaptation):** Instead of updating all billions of parameters, you inject small, low-rank matrices into the attention layers. This trains only a fraction of the weights, making personalization cheap and reusable.
*   **Context & Nuance:** DreamBooth is powerful for high-fidelity specific subjects but expensive. LoRA makes it efficient.
*   **Analogy:** DreamBooth is like a tailor making a suit for *one* specific person. LoRA is like learning a new stitch that applies to many suits, saving fabric and time.
*   **Key Takeaway:** Use DreamBooth + LoRA + Prior Preservation Loss to personalize models efficiently without catastrophic forgetting.

#### Concept 7: Distillation for Efficiency
*   **Detailed Explanation:** Production models need to be fast. Distillation reduces steps.
    *   **Progressive Distillation:** Start with a teacher model (e.g., 1000 steps). Train a student to match the teacher but only every 2 steps (500 steps). Repeat halving until you reach 1 step. This avoids "collapse" where a student tries to jump from pure noise to clean image in one step (too hard).
    *   **InstaFlow:** Uses **Reflow** to straighten the flow paths (making the ODE path more linear) and then applies distillation. It uses **LPIPS** (Learned Perceptual Image Patch Similarity) instead of MSE for loss, as MSE leads to blurry images (regression to the mean).
    *   **Consistency Models:** Map any point on the trajectory to the final clean image. If you start at noise, you predict the end. If you start at noise + a bit, you predict the *same* end. This ensures consistency and allows for few-step generation.
*   **Context & Nuance:** MSE loss is bad for images because it averages out details. LPIPS uses a pre-trained CNN to compare feature maps, preserving texture.
*   **Analogy:** Progressive Distillation is like teaching a piano student. You don't ask them to play a whole sonata in one go. You teach them the first half, then the second half, then combine them. InstaFlow is like straightening a winding road so a car can drive faster.
*   **Key Takeaway:** To make models fast, use Progressive Distillation and InstaFlow. Always prefer perceptual losses (LPIPS) over pixel-wise losses (MSE) for high-quality generation.

---

### 3. Pathways for Further Exploration

1.  **Topic:** **Bradley-Terry Model in Preference Optimization**
    *   **Why it Matters:** The lecture mentions using this for Reward Models. Understanding the math behind pairwise comparisons is essential for building custom alignment pipelines.
    *   **Search Direction:** Look into "Bradley-Terry model derivation for preference learning" and "RLHF (Reinforcement Learning from Human Feedback) loss functions."

2.  **Topic:** **Low-Rank Adaptation (LoRA) Mathematics**
    *   **Why it Matters:** LoRA is the standard for efficient fine-tuning. Understanding *why* low-rank matrices work (e.g., intrinsic dimensionality of tasks) is key to advanced optimization.
    *   **Search Direction:** Study the "LoRA: Low-Rank Adaptation of Large Language Models" paper, focusing on the $W = W_0 + BA$ formulation.

3.  **Topic:** **Rectified Flow and InstaFlow**
    *   **Why it Matters:** This is the cutting edge of fast generation. Understanding "reflow" and "straightening" paths is crucial for next-gen sampling methods.
    *   **Search Direction:** Read the "InstaFlow: Distributed-Flow Modeling for Fast Image Generation" paper. Focus on the "reflow" procedure and how it reduces the number of ODE integration steps.

4.  **Topic:** **LPIPS (Learned Perceptual Image Patch Similarity)**
    *   **Why it Matters:** The lecture emphasized that MSE fails for images. LPIPS is the standard metric for perceptual quality.
    *   **Search Direction:** Investigate the "LPIPS: A New Measure for Perceptual Similarity" paper. Understand how it uses VGG features to mimic human perception.

5.  **Topic:** **Consistency Models (CM) vs. Progressive Distillation**
    *   **Why it Matters:** These are the two main competitors for few-step generation. Understanding the trade-offs (stability vs. quality) is vital for production choices.
    *   **Search Direction:** Compare the "Consistency Models" paper (Song et al.) with "Progressive Distillation" (Salimzadeh et al.). Look for benchmarks on "1-step vs 4-step generation quality."

6.  **Topic:** **Reward Hacking & KL Divergence Regularization**
    *   **Why it Matters:** In Flow GRPO, the KL term prevents the model from "gaming" the reward model. Understanding this stability issue is critical for robust alignment.
    *   **Search Direction:** Search for "KL regularization in RLHF" and "reward hacking examples in diffusion models."

---

### 4. Comprehension & Review Questions

**Recall & Understanding (40%)**
1.  What is the primary difference between the "Pre-training" and "Post-training" phases of the model lifecycle?
2.  Why is a Uniform distribution inappropriate for sampling timesteps during training?
3.  Define the "Logit Normal" distribution in the context of timestep sampling.
4.  What is the specific mathematical relationship between image resolution ($N$) and the uncertainty of pixel value estimation?
5.  What is the "Prior Preservation Loss" in the context of DreamBooth, and why is it necessary?
6.  What is the difference between "Continued Training" and "Supervised Fine-Tuning" in post-training?
7.  Name the two main losses used in InstaFlow to evaluate the student model.
8.  What is "Reward Hacking," and what term is added to the loss to mitigate it?

**Application & Analysis (40%)**
9.  **Scenario:** You are training a model to generate high-resolution 4K images. You notice that the model struggles with fine details (like eyes) but performs well on overall structure. Based on the lecture, how should you adjust the **timestep sampling** distribution, and why?
10. **Scenario:** You have a pre-trained Diffusion Transformer and a pre-trained Vision Transformer (ViT). You want to speed up the training of the DiT. How would you apply **REPA**? Which layers of the DiT should you align, and what is the expected benefit?
11. **Analysis:** Compare **Progressive Distillation** and **Consistency Models**. How does Progressive Distillation handle the "difficulty" of the generation task compared to Consistency Models?
12. **Application:** You are using **Flow GRPO**. You generate a group of 4 images for the prompt "a cat." Two are good, two are bad. How is the "advantage" calculated, and how does this influence the policy update?
13. **Analysis:** Why does the lecture suggest using **LPIPS** instead of MSE for distillation? What happens to the image quality if you use MSE?
14. **Scenario:** A user inputs "teddy bear." The model outputs a blurry, low-detail teddy bear. You suspect the prompt is too simple for the model's current training distribution. What technique from the lecture would you apply to the user's input to improve the result?

**Critical Thinking & Evaluation (20%)**
15. **Critique:** The lecture states that "hard" timesteps are in the middle ($t \approx 0.5$). Critique the assumption that $t=0$ (pure noise) is an "easy" task. Is it truly easy for a model to predict the mean of the data distribution, or does it require significant capacity? How does this affect the choice of sampling distribution?
16. **Evaluation:** In **Diffusion DPO**, the model is incentivized to predict *worse* velocities for "losing" images. Evaluate the risk of this approach: Could the model inadvertently destroy its ability to generate *any* valid images if the "losing" images are still valid but just less preferred? How does the KL divergence term help or fail to help here?

***

### Answer Key & Explanations

**1. Pre-training vs. Post-training:**
*   **Pre-training:** Teaches the model *how* to generate images (learning the data distribution).
*   **Post-training:** Teaches the model *how to generate good* images (alignment, aesthetics, following specific instructions).

**2. Uniform Timestep Sampling:**
*   Uniform sampling treats all timesteps equally. However, the "hard" part of generation is the middle range (structural ambiguity). The ends are "easy" (predicting mean or final tweaks). Uniform sampling under-optimizes the hard parts.

**3. Logit Normal Distribution:**
*   A distribution where $t$ is derived from a Normal variable via the sigmoid function ($t = \sigma(x)$). It concentrates samples in the middle ($0.5$), ensuring the model practices on "hard" tasks more frequently.

**4. Resolution & Uncertainty:**
*   The standard deviation of the pixel estimate scales as $\frac{1}{\sqrt{N}}$. As resolution ($N$) increases, uncertainty decreases. Therefore, higher resolutions can tolerate higher noise levels ($t$) for the same *perceived* noise.

**5. Prior Preservation Loss:**
*   A loss term added during personalization (DreamBooth) that forces the model to continue generating high-quality images for *other* prompts. It prevents "catastrophic forgetting" where the model forgets general skills to focus on the specific subject.

**6. Continued Training vs. SFT:**
*   **Continued Training:** Adds new *knowledge* (e.g., a new domain/object).
*   **Supervised Fine-Tuning:** Improves *behavior* (e.g., aesthetics, lighting, prompt adherence) without necessarily adding new semantic knowledge.

**7. InstaFlow Losses:**
*   **MSE (Mean Squared Error):** Used for initial "warm-up" or regression.
*   **LPIPS (Learned Perceptual Image Patch Similarity):** Used for final quality to avoid blurriness.

**8. Reward Hacking & Mitigation:**
*   **Reward Hacking:** The model optimizes the reward metric but produces low-quality or degenerate images.
*   **Mitigation:** A **KL Divergence** term is added to the loss, penalizing the new policy for deviating too far from the old (pre-trained) policy.

**9. Scenario: High-Res Training & Timesteps:**
*   You should shift the timestep distribution. High-res images have lower uncertainty for the same $t$. To match *perceived* noise, you must increase the effective noise level for high-res images relative to low-res ones. You should also ensure the Logit Normal distribution is tuned to focus on the specific "hard" regions relevant to high-res details.

**10. Scenario: REPA Application:**
*   Project the DiT's internal representations (embeddings) onto the space of the pre-trained ViT's representations.
*   **Layers:** Align **early layers** of the DiT.
*   **Benefit:** Early layers capture semantic structure. Aligning them with a strong pre-trained encoder speeds up convergence (18x faster) by providing a "semantic shortcut."

**11. Analysis: Progressive Distillation vs. Consistency Models:**
*   **Progressive Distillation:** Iteratively halves the number of steps. It assumes the student can learn to match the teacher over shorter intervals step-by-step.
*   **Consistency Models:** Enforces that *any* point on the trajectory maps to the *same* final image. It is a direct constraint on the function's consistency rather than a step-by-step halving approach.

**12. Application: Flow GRPO:**
*   **Advantage Calculation:** The reward of a specific image is compared to the *average* reward of the group. High reward = positive advantage.
*   **Update:** The policy is updated to increase the probability of generating images with positive advantage and decrease it for negative advantage.

**13. Analysis: LPIPS vs. MSE:**
*   **MSE:** Minimizes pixel-wise error, leading to "regression to the mean" (blurry averages).
*   **LPIPS:** Compares high-level features (via a frozen CNN), preserving texture and sharpness. It is perceptually aligned with human vision.

**14. Scenario: Simple Prompt:**
*   Apply **Prompt Enhancement**. This technique takes the user's short input ("teddy bear") and expands it into a long, detailed, "in-distribution" prompt (e.g., "a warm intimate indoor scene featuring a plush teddy bear...") that the model was specifically tuned to handle.

**15. Critique: Is $t=0$ "Easy"?**
*   **Critique:** While $t=0$ is "easy" in the sense that the model only needs to predict the *mean* of the data distribution (a coarse global structure), it is not trivial. It requires significant capacity to capture the global distribution of the data. However, relative to the middle steps (where local details and ambiguous structures must be resolved), it is computationally "cheaper" in terms of decision complexity. The Logit Normal distribution acknowledges this by sampling less at $t=0$ and more at $t=0.5$.

**16. Evaluation: DPO Risk & KL Term:**
*   **Risk:** If "losing" images are still valid but just less preferred, forcing the model to predict *worse* velocities for them could degrade the model's general capability.
*   **KL Term:** The KL divergence term penalizes large deviations from the original policy. It acts as a "leash," ensuring the model doesn't stray too far into "bad" velocity predictions, thus preserving general quality while still preferring "good" images. However, if the KL term is too weak, the model may still degrade; if too strong, it may fail to learn preferences.
