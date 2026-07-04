# Activation Steering in LLMs: Foundations, Recent Research, and Abliteration

Activation steering is one of the most active and arguably the most consequential sub-field of mechanistic interpretability right now. It has produced a steady stream of papers since 2023, a small industry of open-source tools, and — in the form of *abliteration* — a real, ongoing crisis for open-weight model safety. This report walks through the basics, the most important recent papers, the applications, and the open problems.

## 1. What Activation Steering Actually Is

The basic idea is deceptively simple. Every transformer LLM has a **residual stream** — a running sum of vectors that gets modified at every layer. If you record the residual stream activation at some layer for a "happy" prompt and subtract the activation for a "sad" prompt, you get a *steering vector*. Add that vector (times some coefficient) to the residual stream during generation, and the model starts producing happier text — without any retraining, weight update, or prompt engineering [1][2].

Mathematically:

```
h' = h + α · v
```

where `h` is the residual stream hidden state at layer `l`, `v` is the steering vector, and `α` is the intervention strength. This works because high-level concepts ("sentiment", "refusal", "honesty", "topic") tend to be encoded as **linear directions** in the high-dimensional activation space of large transformers — the *Linear Representation Hypothesis* [3][4]. The transformer is a stack of linear and near-linear operations, so a small nudge along the right direction propagates through the rest of the forward pass and biases the output distribution.

The intervention is typically applied at one or a few "best" layers (often mid-network, where conceptual abstraction is strongest), sometimes at all token positions after the user prompt, sometimes only at the last token. The choice of layer, coefficient, token position, and vector shape are all free hyperparameters.

Two important up-front distinctions:
- **Activation steering** is sometimes used interchangeably with **activation engineering**, **representation engineering (RepE)**, and **inference-time intervention (ITI)**. They share the same core mechanism but differ in vocabulary and tradition. RepE is the most formal umbrella term and explicitly covers both reading (finding directions) and control (applying them) [3].
- **Abliteration** is a *subtype* of activation steering — specifically, directional ablation of the refusal direction. People frequently confuse "activation steering = abliteration", but steering is the family; abliteration is one specific attack in that family.

## 2. The Foundational Papers (2023)

Four 2023 papers basically established the field:

**Activation Addition (ActAdd)** — Turner et al., August 2023 [1]. Introduced the contrast-pair method: pick two contrasting prompts (e.g. "Love" vs "Hate"), take the difference of their activations at a chosen layer, and add it back during inference. Demonstrated SOTA on negative-to-positive sentiment shift and detoxification with LLaMA-3, OPT, GPT-2, and GPT-J. Lightweight — no optimization needed, works with as few as two prompts.

**Contrastive Activation Addition (CAA)** — Panickssery et al. (Rimsky et al.), December 2023 [2]. Refined ActAdd by *averaging* activation differences across many positive/negative pairs, producing a cleaner steering vector. Tested on Llama 2 Chat for hallucination vs. factuality steering. This is the workhorse method most subsequent papers build on.

**Representation Engineering (RepE)** — Zou et al., October 2023 [3]. Formalized the whole paradigm with Linear Artificial Tomography (LAT): stimulate the model with concept-specific prompts, collect neural activity, fit a linear model (PCA, logistic regression) to identify a "reading vector" for the concept. Distinguished between Reading Vector Intervention (add/subtract a fixed direction) and Contrast Vector Intervention (use paired-prompt differences). Argued RepE is a top-down transparency approach analogous to neuroimaging.

**Inference-Time Intervention (ITI)** — Li, Patel, Viégas, Pfister, Wattenberg, NeurIPS 2023 spotlight [4]. Targeted *truthfulness* specifically. Found that certain attention heads have a clear distribution difference between true and false statements, trained linear probes per head to identify "truth directions", and at inference time shifted those heads' activations along the truth direction. On Alpaca (instruction-tuned LLaMA), ITI lifted TruthfulQA accuracy from 32.5% to 65.1%, with the trade-off that higher intervention strength hurt helpfulness.

Together, these papers made two claims that have shaped everything since: (a) high-level concepts live as linear directions in activation space, and (b) you can move the model's behavior along those directions at inference time without touching weights.

## 3. The 2024–2025 Wave: More Sophisticated Steering

Once the basic recipe was established, the next two years of work pushed in several directions.

### Sparse autoencoder (SAE)-based steering
The biggest single trend. Sparse autoencoders decompose dense, polysemantic activations into a much larger, mostly-inactive dictionary of more interpretable features. Steering in SAE space lets you isolate a single causally relevant feature instead of a global "mean difference" vector.

- **O'Brien et al. (2024)** — early work on SAE-targeted steering for refusal features [5].
- **FGAA (Feature Guided Activation Additions)** — Soo et al., January 2025 [6]. Operates inside SAE latent space, applies density filtering and Top-K selection to pick the most semantically meaningful SAE latents, outperforms CAA, SAE-TS, and SAE-decoder steering on Gemma-2 2B and 9B.
- **SAS (Sparse Activation Steering)** — Bayat et al., February 2025 [7]. Uses contrastive prompt pairs inside the SAE feature space; scaling the SAE improves monosemanticity of the resulting vectors.
- **LF-Steering** — Yang et al., January 2025 [8]. Identifies SAE features responsible for *semantic inconsistency* in paraphrase-robust QA; reports up to 8.9% accuracy gains on NLU tasks.
- **CorrSteer** — Cho, Wu, Koshiyama, August 2025 (ICML 2026) [9]. Picks features by *correlating sample correctness with SAE activations at generation time*, no contrastive dataset needed; shows +27.2% on HarmBench with only 108 samples.
- **GSAE (Graph-Regularized SAE)** — Yeon et al., December 2025 [10]. Adds graph regularization over a neuron co-activation graph to learn safety-steering directions that respect feature dependencies; on Llama-3-8B it improves selective refusal on JailbreakBench by 20.1 points and on HarmBench by 16.8 points.
- A complementary finding: when you select SAE features by *output* influence (what the feature does to the output distribution) rather than *input* activation, steering is dramatically more effective [11].

### Beyond single vectors
- **Conceptors** — Postmus & Abreu, NeurIPS 2024 [12]. Represent activation sets as ellipsoidal regions rather than single vectors; Boolean operations on conceptors compose more reliably than adding vectors.
- **Activation Transport (AcT)** — Rodriguez et al. at Apple, ICLR 2025 [13]. Frames steering as an optimal-transport problem, transforming the entire distribution of harmful activations to match harmless ones rather than subtracting one direction. Reports layer-selective intervention at 40–60% network depth beats full-network abliteration.
- **Affine Concept Editing (ACE)** — Marshall, Scherlis, Belrose, November 2024 [14]. Shows that in fact refusal is best modeled as an affine function, not a linear one; ACE combines subspace projection with activation addition and works on ten models including Llama 3 70B.
- **REAL** — Zhan et al., 2024 [15]. A VQ-AE per module to localize which transformer modules are behaviorally relevant, then steer only those.
- **Multi-Directional Refusal Suppression (SOM)** — Piras et al., November 2025 [16]. Uses Self-Organizing Maps to learn multiple refusal directions instead of one; substantially higher attack success than single-direction abliteration with only marginal capability loss.
- **AlphaSteer** — Sheng et al., ICLR 2026 [17]. Learns steering vectors with a *null-space constraint* so that on benign prompts the steering is mathematically zero — preserving capabilities while still enhancing safety on harmful prompts.
- **Sparse Steering** (Bayat et al.), Optimal Transport Steering, and many others continue this thread.

### Specialized techniques
- **CAST (Conditional Activation Steering)** — IBM's [18] extension that only applies the steering vector when a classifier detects the relevant context.
- **Adaptive Activation Steering** — auto-tunes the intervention strength per prompt.
- **Entropic Activation Steering** — uses entropy signals to control LLM agents.
- **Activation Scaling** — a simpler cousin that just scales existing activations rather than adding a new direction.
- **REPBEND** — moves activation steering's idea into fine-tuning, defining a loss based on vector differences to permanently bake a safety direction into weights [19].

### Surveys and taxonomies
Two 2025 surveys are worth bookmarking:
- **Wehner et al.**, "Taxonomy, Opportunities, and Challenges of RepE for LLMs" (Feb 2025) [20]. Proposes a unified pipeline of Representation Identification → Operationalization → Control. Lists major open problems: hyperparameter sensitivity, OOD generalization, capability degradation, multi-concept interference, and the fundamental question of whether activation-space interpretability is even theoretically sound.
- **Bartoszcze et al.**, "RepE for LLMs: Survey and Research Challenges" (Feb 2025) [21]. Formalizes the goals and methods of RepE.

## 4. Applications Beyond Safety

Steering has been applied well beyond refusal and truthfulness:

- **Sentiment / topic / style transfer** — the original ActAdd use cases, now mature.
- **Code generation** — type-prediction correction and code style steering.
- **Chain-of-thought compression (ASC)** — Azizi et al., 2026 [22]. Single steering vector reduces CoT verbosity by up to 67.4% on MATH500, GSM8K, and LiveCodeBench with no accuracy loss, delivering ~2.7× end-to-end speedup. Practical implications for inference cost.
- **Music generation** — Panda et al., June 2025 [23]. Activation steering of MusicGen for timbre, genre, and style transfer; SAE-based concept discovery in audio is the first of its kind.
- **Theorem proving** — steering Lean-based provers toward better tactic selection.
- **Vision-Language Models** — SteerVLM and similar lightweight steering modules.
- **Preference / personalization alignment** — slider-style control over chatbot style (e.g., "budget" vs. "luxury").

## 5. Abliteration — The Big Application (and Controversy)

Abliteration is by far the most-discussed application of activation steering, and it's worth its own deep dive.

### Where the term comes from
The mechanistic finding that makes abliteration possible is from **Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee & Nanda**, "Refusal in Language Models Is Mediated by a Single Direction" (June 2024, cited 800+ times) [24]. They showed that across 13 open-source chat models up to 72B parameters, there exists a single direction in the residual stream such that erasing it from activations prevents the model from refusing harmful prompts, and adding it back elicits refusal even on harmless prompts. They also mechanistically explained why GCG-style adversarial suffixes work — they hijack attention heads that would otherwise propagate the refusal direction.

The technique of *projecting that direction out of the model's weights* to permanently remove refusal was popularized in mid-2024 by the anonymous "FailSpy" account under the name **abliteration** (a play on "ablate" + "literation"). It was rapidly picked up, packaged, and turned into a movement by **Maxime Labonne** in his widely-read June 2024 Hugging Face blog post [25], which explained how to take a Llama-3-8B-Instruct, identify the refusal direction via mean activation difference on harmful/harmless prompts, project that direction out of attention and MLP weights, and ship the resulting uncensored model. His merged-uncensored variant, NeuralDaredevil-8B, briefly held the #1 spot on the Open LLM Leaderboard in the 8B category.

### The standard recipe
1. Collect harmful and harmless instructions (commonly `mlabonne/harmful_behaviors` and `mlabonne/harmless_alpaca`).
2. Run both through the model, record residual stream activations at every layer and token position.
3. Compute the mean activation difference at each (layer, position); normalize to get candidate refusal directions.
4. Select the direction that maximizes refusal drop while keeping the model coherent.
5. Project the direction out of every write matrix that contributes to the residual stream — attention output projections (`W_O`) and MLP down-projections (`W_down`) are the most common targets. Formally, for weight matrix `W`: `W' = W - r̂ r̂ᵀ W`.
6. Save the modified model. No training data, no gradients, no GPU cluster — minutes on a consumer GPU.

### Refinements of the technique
- **Norm-preserving ablation** — only project out the directional component, preserving the row norms of the weight matrix. grimjim showed this both improves capability preservation and reveals that norm effects were confounding earlier results [26].
- **Projected abliteration** — also remove the mechanistically irrelevant components of the refusal direction.
- **Norm-preserving biprojected abliteration** — extends to removing the corresponding component in every layer measured from any reference layer, so each layer's intervention doesn't disturb other layers' harmless directions [26].

### Tools (as of late 2025 / early 2026)
The Cross-Architecture Evaluation by Young (December 2025) [27] benchmarked the four main open-source tools on 16 instruction-tuned models:

| Tool | Method | Compatibility (16 models) | Strength | Weakness |
|---|---|---|---|---|
| **Heretic** (p-e-w) [28] | TPE Bayesian optimization + parametrized directional ablation | 16/16 | Universal; lowest KL divergence via auto-search | 30–110 min/model; GSM8K ∆ up to −18.8 pp |
| **DECCP / llm-abliteration** (jim-plus / FailSpy lineage) | Single-pass orthogonalization, supports norm-preserving + projected variants, 4-bit sharded | 11/16 | Fast (~2–20 min); best capability preservation (avg GSM8K ∆ −0.13 pp) | Limited model coverage |
| **ErisForge** (Tsadoq) [29] | Wraps decoder layers with `AblationDecoderLayer` for ablation and `AdditionDecoderLayer` for direction injection | 9/16 | Lowest avg GSM8K ∆ (−0.28 pp); supports injection as well as ablation | Slower than single-pass |
| **FailSpy / abliterator** | TransformerLens-based activation hooks | 5/16 | Interactive exploration | Limited to TransformerLens-supported models |

**Key empirical findings from the benchmark**:
- Single-pass methods preserve capabilities better than Bayesian-optimized ones on average, but Heretic's distribution shift is lower when its optimization converges.
- Mathematical reasoning (GSM8K) is the most fragile capability, with swings from +1.51 pp to −18.81 pp depending on tool and architecture.
- The Heretic implementation supports most dense models, many MoEs, some multimodal models, and even Qwen3.5-style hybrid architectures; pure SSMs and certain research architectures aren't supported.

By April 2026, an empirical analysis by Alice [30] reported that HuggingFace hosts over **4,800 models tagged "abliterated"** with cumulative downloads exceeding **3.5 million**, and Heretic alone has surpassed **17,800 GitHub stars and 1,781 forks** — making abliteration one of the fastest-spreading open-source ML movements in history.

### Empirical effect
Abliteration reliably drops refusal rates from 90–100% to 0–15% on standard harmful-prompt benchmarks. It is, at this point, a *solved attack* on single-direction safety training in open-weight models.

## 6. Defenses Against Abliteration — The Arms Race

Once abliteration was demonstrated, the safety community began publishing defenses. The current state:

**Extended-refusal fine-tuning** — Abu Shairah et al., May 2025 [31]. The simplest and most effective defense so far. Train the model to refuse using multi-component refusals: (i) neutral topic overview, (ii) explicit refusal, (iii) ethical rationale. This *spreads the safety signal across multiple token positions and semantic dimensions* so that no single direction captures it. Result: post-abliteration refusal drops only ~10 percentage points (vs. 70–80 points for standard fine-tuning). Trade-off: some coherence loss if the model is heavily abliterated.

**Tamper-resistant safeguards (TAR)** — Tamirisa et al., ICLR 2025 [32]. Adversarially trains the safeguard (refusal or unlearning) against simulated weight-modification attacks, using meta-learning. Claims resistance to thousands of fine-tuning steps. Conceptually broader than activation-space defenses — it protects against any weight tampering, not just abliteration.

**APRA (Abliteration Prevention via Refusal Aliases)** [33]. Specifically targets the *extractability* of the refusal direction by applying rank-k updates to residual-stream writer matrices, replacing refusal-inducing activations with random aliases and patching downstream reader matrices to compensate. On Llama-3.1-8B and Gemma-2-9B, substantially improves post-abliteration refusal with minimal utility loss.

**AlphaSteer** [17]. Learns refusal steering vectors constrained to the null space of benign-prompt activations, so steering is mathematically zero on benign inputs while still inducing refusal on malicious ones. Reduces the safety/utility trade-off.

**Always Refuse / RAS (Refusal Activation Steering)** [34]. Training-free, inference-time addition of a refusal direction built from contrastive activations; improves robustness against single- and multi-turn jailbreaks at the cost of benchmark degradation on benign tasks.

**AntiDote** [35]. Bi-level adversarial training for tamper-resistant LLMs, September 2025. Another meta-training defense.

**SRA (Spectral Refusal Ablation)** — more recent work that disentangles refusal from capabilities by operating on spectral components of weight matrices.

**GRP-Obliteration** (Russinovich et al., 2026) — strips safety using only a single unlabeled prompt, a worrying escalation on the attack side [30].

**Adversarial training methods** — ReFAT (Yu et al., 2024), DeepRefusal (Xie et al., Sep 2025) — explicitly simulate or probabilistically ablate refusal directions during fine-tuning so the model rebuilds safety in a way that's robust to activation-space attacks, achieving up to 95% reduction in jailbreak success rates.

The honest summary: defenses are catching up, but the attack remains fundamentally cheap and effective. No published defense today defeats abliteration *and* preserves full capability. Defenders have largely concluded that **distributing the safety signal across many dimensions** (temporal, semantic, causal) is the only durable strategy — which is why extended-refusal and multi-dimensional approaches have dominated.

## 7. Critical Open Problems

The field is far from settled. Three threads deserve attention:

### Is "single direction" even the right model?
A flurry of 2025–2026 work challenges the single-direction account of refusal:
- **Yeo et al., "Beyond I'm Sorry, I Can't" (Sep 2025)** [36]. SAEs reveal that refusal has a small core of shared latents plus a long tail of style/domain-specific latents. Ablating only the dominant direction collapses this rich structure.
- **Joad et al., "There Is More to Refusal than a Single Direction" (Feb 2026)** [37]. Across eleven refusal categories (safety, incomplete requests, anthropomorphization, over-refusal), the corresponding directions are geometrically distinct (cosine similarities 0.4–0.6, some near-orthogonal). Yet linear steering along any of them produces nearly identical *behavioral* refusal-vs-over-refusal trade-offs. Different directions primarily change *how* the model refuses (style, framing) rather than *whether*.
- **Wollschläger et al. (2025)** — refusal can be mediated by multiple independent directions forming "concept cones" with representational independence.
- **Zhang & Sun (2025)** — decomposed refusal into *harm-detection* and *refusal-execution* sub-directions.
- **Marshall et al. (ACE)** [14] — refusal is best modeled as an affine function, not purely linear.

The consensus emerging: the single-direction story is *operationally useful* (you really can ablate a single vector and break refusal) but *mechanistically incomplete*. The geometric complexity doesn't change what abliteration does, but it does bound how durable single-vector defenses can be.

### Reasoning models are different
A surprising and important finding from **Yang et al., "Beyond a Single Direction: CoT Disrupts Simple Steering of Refusal" (May 2026)** [38]: in large reasoning models like DeepSeek-R1-Distill-LLaMA-8B, refusal is *jointly* encoded in residual stream activations *and* in the chain-of-thought trace. Steering activations alone reverses refusal in only 39% of cases; removing CoT entirely pushes that to 70%; regenerating CoT *under* steering pushes it to 94%. Implication: activation steering is less effective on reasoning models, and CoT itself is an attack surface that activation-level defenses don't reach.

### Open conceptual questions
Wehner's 2025 survey [20] crystallizes the deep worries:
- RepE is *hyperparameter-sensitive* — change the layer or coefficient and results swing wildly.
- RepE *doesn't generalize OOD* — operators extracted on one prompt distribution underperform on others.
- RepE can *hurt general capability*, sometimes invisibly.
- Multi-concept steering suffers *destructive interference*.
- The most philosophically uncomfortable issue: **activation-space interpretability may be partly doomed** — features you find in activations might be statistical artifacts the model doesn't actually use in computation.

## 8. Takeaways

- **Activation steering is the most practical offshoot of mechanistic interpretability.** It works, it's cheap, and it lets you do inference-time control without retraining. The field's foundational claim — high-level concepts are linear directions in activation space — is real enough to be useful even if mechanistically incomplete.
- **The 2024–2025 frontier is SAE-based, multi-dimensional, and theoretically self-critical.** Sparse autoencoders give finer-grained steering; conceptors, optimal transport, and affine methods generalize the math; surveys are openly cataloguing the weaknesses.
- **Abliteration is the headline application** — and a real safety problem. It is fast, cheap, effective, and broadly deployed (4,800+ HuggingFace models, 3.5M+ downloads as of early 2026). Single-direction safety training is dead as a standalone defense.
- **The defense frontier is "spread the signal across many dimensions."** Extended-refusal fine-tuning, tamper-resistant training, refusal-direction obfuscation (APRA), null-space-constrained steering (AlphaSteer), and SAE-aware multi-vector attacks (SOM) are the active research directions.
- **Reasoning models and multi-dimensional refusal accounts are the new frontier.** The single-direction story is operationally useful but no longer sufficient as a model of how LLMs actually represent safety behavior. Expect the next wave of work to focus on reasoning models, on multi-vector representations, and on combining activation-level and CoT-level interventions.

If you only read three papers: Arditi et al. (2024) for the refusal-direction finding [24], Arditi et al. (2024) and Labonne (2024) for abliteration itself [24][25], and Wehner et al. (2025) for the taxonomy and open problems [20]. Everything else in the field is reacting to or refining these three.

---

## References

[1] Turner et al., "Steering Language Models With Activation Engineering," arXiv:2308.10248, Aug 2023. https://arxiv.org/abs/2308.10248

[2] Panickssery, Gabrieli, Schulz, Tong, Hubinger, Turner, "Steering Llama 2 via Contrastive Activation Addition," ACL 2024 / arXiv:2312.06681, Dec 2023. https://arxiv.org/abs/2312.06681

[3] Zou et al., "Representation Engineering: A Top-Down Approach to AI Transparency," arXiv:2310.01405, Oct 2023. https://arxiv.org/abs/2310.01405

[4] Li, Patel, Viégas, Pfister, Wattenberg, "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model," NeurIPS 2023, arXiv:2306.03341. https://arxiv.org/abs/2306.03341

[5] O'Brien et al., "Steering Language Model Refusal with Sparse Autoencoders," ICML 2025. https://icml.cc/virtual/2025/50928

[6] Soo et al., "Interpretable Steering of Large Language Models with Feature Guided Activation Additions," ICLR 2025 / arXiv:2501.09929. https://arxiv.org/abs/2501.09929

[7] Bayat et al., "Steering Large Language Model Activations in Sparse Spaces," arXiv:2503.00177, Feb 2025. https://arxiv.org/abs/2503.00177

[8] Yang et al., "LF-Steering: Latent Feature Activation Steering for Enhancing Semantic Consistency," arXiv:2501.11036, Jan 2025. https://arxiv.org/pdf/2501.11036

[9] Cho, Wu, Koshiyama, "CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features," ICML 2026 / arXiv:2508.12535. https://arxiv.org/abs/2508.12535

[10] Yeon, Cinus, Wu, Luceri, "GSAE: Graph-Regularized Sparse Autoencoders for LLM Safety Steering," arXiv:2512.06655, Dec 2025. https://arxiv.org/abs/2512.06655

[11] "SAEs Are Good for Steering — If You Select the Right Features," arXiv:2505.20063. https://arxiv.org/html/2505.20063v1

[12] Postmus & Abreu, "Steering Large Language Models using Conceptors," NeurIPS 2024 / arXiv:2410.16314. https://arxiv.org/abs/2410.16314

[13] Rodriguez et al. (Apple), "Controlling Language and Diffusion Models by Transporting Activations," ICLR 2025 / arXiv:2410.23054. https://arxiv.org/abs/2410.23054

[14] Marshall, Scherlis, Belrose, "Refusal in LLMs is an Affine Function," arXiv:2411.09003, Nov 2024. https://arxiv.org/abs/2411.09003

[15] Zhan et al., "REAL: Reading Out Transformer Activations for Precise Localization in Language Model Steering." https://openreview.net/forum?id=P38RYdkFLI

[16] Piras et al., "SOM Directions are Better than One: Multi-Directional Refusal Suppression in Language Models," AAAI 2026 / arXiv:2511.08379. https://arxiv.org/abs/2511.08379

[17] Sheng et al., "AlphaSteer: Learning Refusal Steering with Principled Null-Space Constraint," ICLR 2026 / arXiv:2506.07022. https://arxiv.org/abs/2506.07022

[18] IBM, "Conditional Activation Steering (CAST)," https://github.com/IBM/activation-steering/blob/main/docs/faq.md

[19] "Representation Bending for Large Language Model Safety," ACL 2025. https://aclanthology.org/2025.acl-long.1173.pdf

[20] Wehner, Abdelnabi, Tan, Krueger, Fritz, "Taxonomy, Opportunities, and Challenges of Representation Engineering for LLMs," arXiv:2502.19649, Feb 2025. https://arxiv.org/abs/2502.19649

[21] Bartoszcze et al., "Representation Engineering for Large-Language Models: Survey and Research Challenges," arXiv:2502.17601, Feb 2025. https://arxiv.org/abs/2502.17601

[22] Azizi, Potraghloo et al., "Activation Steering for Chain-of-Thought Compression," ACL Findings 2026 / arXiv:2507.04742. https://openreview.net/forum?id=LLxSS9i2JD

[23] Panda et al., "Fine-Grained Control Over Music Generation with Activation Steering," arXiv:2505.18186. https://arxiv.org/html/2505.18186v3

[24] Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda, "Refusal in Language Models Is Mediated by a Single Direction," arXiv:2406.11717, Jun 2024. https://arxiv.org/abs/2406.11717

[25] Labonne, "Uncensor any LLM with abliteration," HuggingFace Blog, Jun 2024. https://huggingface.co/blog/mlabonne/abliteration

[26] grimjim, "Norm-Preserving Biprojected Abliteration," HuggingFace Blog, 2025. https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration

[27] Young, "Comparative Analysis of LLM Abliteration Methods: A Cross-Architecture Evaluation," arXiv:2512.13655, Dec 2025. https://arxiv.org/abs/2512.13655

[28] p-e-w, "Heretic: Fully automatic censorship removal for language models," GitHub. https://github.com/p-e-w/heretic

[29] Tsadoq, "ErisForge: Dead Simple LLM Abliteration," GitHub. https://github.com/Tsadoq/ErisForge

[30] Alice, "Empirical Analysis of Activation-Space Abliteration on LLM Safety," April 2026. https://go.alice.io/hubfs/alice-abliteration-report-april2026.pdf

[31] Abu Shairah, Hammoud, Ghanem, Turkiyyah, "An Embarrassingly Simple Defense Against LLM Abliteration Attacks," arXiv:2505.19056, May 2025. https://arxiv.org/abs/2505.19056

[32] Tamirisa et al., "Tamper-Resistant Safeguards for Open-Weight LLMs," ICLR 2025 / arXiv:2408.00761. https://arxiv.org/abs/2408.00761

[33] "LLM Abliteration Prevention Via Refusal Aliases (APRA)," OpenReview 2026. https://openreview.net/pdf/1c3961ec046b51cb7de713394e073baa45ac95d7.pdf

[34] "Always Refuse: Steering LLMs Against Jailbreaks with Refusal Activation Steering (RAS)," AAAI 2026. https://ojs.aaai.org/index.php/AAAI/article/view/42191/46152

[35] Sanyal, Ray, Mandal, "AntiDote: Bi-level Adversarial Training for Tamper-Resistant LLMs," arXiv:2509.08000, Sep 2025. https://arxiv.org/abs/2509.08000

[36] "Beyond I'm Sorry, I Can't: Dissecting Large Language Model Refusal," arXiv:2509.09708, Sep 2025. https://tldr.takara.ai/p/2509.09708

[37] Joad, Hawasly, Boughorbel, Durrani, Sencar, "There Is More to Refusal in Large Language Models than a Single Direction," arXiv:2602.02132, Feb 2026. https://arxiv.org/abs/2602.02132

[38] Yang, Meier, Zhao, Ruas, Gipp, "Beyond a Single Direction: Chain-of-Thought Disrupts Simple Steering of Refusal," arXiv:2605.26772, May 2026. https://arxiv.org/abs/2605.26772