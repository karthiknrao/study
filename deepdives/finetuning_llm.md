# Deep Dive: LLM Fine-Tuning Methods (2025-2026)

A comprehensive survey of fine-tuning techniques, frameworks, papers, and repos.

---

## Table of Contents

1. [Taxonomy of Fine-Tuning Approaches](#1-taxonomy-of-fine-tuning-approaches)
2. [Full Fine-Tuning](#2-full-fine-tuning)
3. [Parameter-Efficient Fine-Tuning (PEFT)](#3-parameter-efficient-fine-tuning-peft)
   - [LoRA](#31-lora-low-rank-adaptation)
   - [QLoRA](#32-qlora-quantized-lora)
   - [DoRA](#33-dora-weight-decomposed-low-rank-adaptation)
   - [GaLore](#34-galore-gradient-low-rank-projection)
   - [Spectrum](#35-spectrum)
   - [Other PEFT Variants](#36-other-peft-variants)
4. [Preference Alignment Methods](#4-preference-alignment-methods)
   - [RLHF / PPO](#41-rlhf--ppo)
   - [DPO](#42-dpo-direct-preference-optimization)
   - [ORPO](#43-orpo-odds-ratio-preference-optimization)
   - [KTO](#44-kto-kahneman-tversky-optimization)
   - [GRPO](#45-grpo-group-relative-policy-optimization)
5. [Knowledge Distillation](#5-knowledge-distillation)
6. [Continued Pre-Training (CPT)](#6-continued-pre-training-cpt)
7. [Frameworks & Libraries](#7-frameworks--libraries)
   - [Unsloth](#71-unsloth)
   - [Axolotl](#72-axolotl)
   - [LLaMA-Factory](#73-llama-factory)
   - [TRL (Hugging Face)](#74-trl-hugging-face)
   - [DeepSpeed](#75-deepspeed)
   - [TorchTune (Meta)](#76-torchtune-meta)
   - [PEFT (Hugging Face)](#77-peft-hugging-face)
8. [Key Papers](#8-key-papers)
9. [Notable GitHub Repos](#9-notable-github-repos)
10. [Decision Guide: Which Method to Use](#10-decision-guide-which-method-to-use)

---

## 1. Taxonomy of Fine-Tuning Approaches

LLM fine-tuning can be organized along two axes:

| Axis | Options |
|---|---|
| **Training objective** | SFT, RLHF/PPO, DPO, ORPO, KTO, GRPO |
| **Parameter-update strategy** | Full fine-tuning, LoRA, QLoRA, DoRA, GaLore, Spectrum, Adapters |

These are orthogonal: you can do SFT with LoRA, or DPO with full fine-tuning, etc.

---

## 2. Full Fine-Tuning

Updates **all** model weights on your task data.

- **Pros**: Maximum performance, best for large domain shifts
- **Cons**: Requires enormous GPU memory (model + optimizer states + gradients); creates a separate model copy per task; risk of catastrophic forgetting
- **When to use**: Large domain shift from pretraining data (e.g., continued pretraining on a new language or domain), or when you need peak performance and have the compute budget
- **Key insight** (NeurIPS 2025 - "LoRA vs Full Fine-tuning: An Illusion of Equivalence"): Full fine-tuning finds different solution neighborhoods than LoRA — the learned representations are not equivalent even when task performance is similar

---

## 3. Parameter-Efficient Fine-Tuning (PEFT)

PEFT methods keep base weights frozen and only update a small fraction of parameters. This reduces memory 10-20x while retaining 90-95% of full fine-tuning quality.

### 3.1 LoRA (Low-Rank Adaptation)

**Paper**: Hu et al., 2022 — [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

The foundational PEFT method. Injects trainable low-rank matrices into transformer layers:

```
W' = W + A·Bᵀ   where A ∈ R^(d×r), B ∈ R^(r×d), r ≪ d
```

- **Trainable params**: ~0.1-1% of total
- **Rank (r)**: typically 8, 16, 32, 64, 128. Higher rank = more capacity but more memory.
- **Inference**: zero-cost — adapters are merged into base weights after training
- **Targets**: usually applied to Q, K, V, O projection matrices in attention layers

**Recent finding** (arXiv:2602.04998, 2025): *"Learning Rate Matters: Vanilla LoRA May Suffice"* — properly tuned learning rate with vanilla LoRA matches or beats fancier variants (PiSSA, DoRA, MiLoRA) in many cases.

### 3.2 QLoRA (Quantized LoRA)

**Paper**: Dettmers et al., 2023 — [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)

Combines 4-bit quantization of the base model with LoRA adapters trained in 16-bit:

- **Base model**: quantized to 4-bit NormalFloat (NF4) — a data type optimized for normally-distributed neural network weights
- **Adapters**: trained in BF16
- **Memory**: 65B model trainable on a single 48GB GPU
- **Paged optimizers**: prevents OOM during memory spikes

QLoRA democratized LLM fine-tuning by making it possible on consumer GPUs.

### 3.3 DoRA (Weight-Decomposed Low-Rank Adaptation)

**Paper**: Liu et al., 2024 — [arXiv:2402.09353](https://arxiv.org/abs/2402.09353) (ICML 2024 Oral, NVIDIA)

Decomposes pre-trained weights into **magnitude** and **direction** components, then applies LoRA only to the directional component:

```
W' = m · (W/||W|| + ΔV)    where m = magnitude (scalar), ΔV = LoRA update to direction
```

- **Consistently outperforms LoRA** across tasks and models
- **Intuition**: magnitude controls "how much" a weight matters; direction controls "what it does". Decomposing these allows the model to learn more like full fine-tuning
- **Repo**: [github.com/NVlabs/DoRA](https://github.com/NVlabs/DoRA)

### 3.4 GaLore (Gradient Low-Rank Projection)

**Paper**: Zhao et al., 2024 — [arXiv:2403.03507](https://arxiv.org/abs/2403.03507) (ICML 2024)

Instead of low-rank weights, projects **gradients** into a low-rank subspace:

- Allows full-parameter learning but with memory-efficient optimizer states
- 8-bit GaLore reduces optimizer memory by up to **82.5%**, total training memory by **63.3%**
- First demonstration of pre-training a 7B model on a single consumer 24GB GPU
- Works for both pre-training and fine-tuning

**Variant**: Natural GaLore ([arXiv:2410.16029](https://arxiv.org/abs/2410.16029)) — accelerates GaLore further with natural gradient methods.

### 3.5 Spectrum

**Paper**: [arXiv:2408.10156](https://arxiv.org/abs/2408.10156)

Uses **Signal-to-Noise Ratio (SNR)** analysis across layers to identify the most informative layers, then selectively fine-tunes only those:

- Competitive with full fine-tuning, beats QLoRA on benchmarks
- Arcee AI reports **42% training speedup** integrating Spectrum into their pipeline
- Particularly effective in distributed training setups
- Supported in Hugging Face TRL and Phil Schmid's 2025 fine-tuning guide

### 3.6 Other PEFT Variants

| Method | Key Idea | Paper |
|---|---|---|
| **PiSSA** | Initializes LoRA with SVD of original weights for better starting point | Meng et al., 2024 |
| **MiLoRA** | Mid-rank initialization for LoRA | Wang et al., 2024 |
| **Init[AB]** | Optimal initialization strategy for LoRA matrices | Li et al., 2025 |
| **SparseLoRA** | Contextual sparsity to accelerate fine-tuning | arXiv:2506.16500 |
| **LoRA+** | Different learning rates for A and B matrices | 2024 |
| **RSLoRA** | Rank-stabilized LoRA with improved scaling | 2024 |
| **ScaLoRA** | Optimally scaled LoRA for high-rank fine-tuning | arXiv:2510.23818 |
| **LaMDA** | Spectrally decomposed low-dimensional adaptation | arXiv:2406.12832 |
| **MoRA** | High-rank updating via non-square matrices | Jiang et al., 2024 |
| **IA³** | Infused adapters by inhibiting/ amplifying activations | Liu et al., 2022 |
| **BitFit** | Only train bias terms (extreme minimalism) | Ben Zaken et al., 2022 |
| **Prefix Tuning** | Trainable "virtual tokens" prepended to input | Li & Liang, 2021 |
| **Adapters** | Small bottleneck modules inserted between layers | Houlsby et al., 2019 |

---

## 4. Preference Alignment Methods

These methods align LLM outputs with human (or AI) preferences. They typically run **after** SFT.

### 4.1 RLHF / PPO

**The original alignment approach** (used for ChatGPT, Claude, etc.):

1. **SFT** on high-quality demonstrations
2. Train a **reward model** from human preference data (chosen vs. rejected pairs)
3. Optimize the LLM policy using **PPO** (Proximal Policy Optimization) to maximize reward

- **Pros**: Handles complex, nuanced goals; well-studied
- **Cons**: Very expensive (requires separate reward model + RL training loop); unstable; complex engineering
- **Still dominant** for frontier model training at major labs

### 4.2 DPO (Direct Preference Optimization)

**Paper**: Rafailov et al., 2023 — [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)

Eliminates the reward model and RL loop entirely. Treats the LLM itself as an implicit reward model:

```
Loss = -log σ(β · (log π_θ(y_w|x) / π_ref(y_w|x) - log π_θ(y_l|x) / π_ref(y_l|x)))
```

- **Much simpler** than RLHF — single training run on preference pairs
- **Requires**: (prompt, chosen, rejected) triplets
- **Often exceeds PPO-based RLHF** in controllability
- **Limitation**: needs a reference model in memory; can be sensitive to data quality

**Debate** (arXiv:2404.10719 — "Is DPO Superior to PPO?"): PPO can outperform DPO when properly tuned, but DPO is far easier to get right.

### 4.3 ORPO (Odds Ratio Preference Optimization)

**Paper**: Hong et al., 2024 — [arXiv:2403.07691](https://arxiv.org/abs/2403.07691)

Combines SFT and alignment into a **single step**:

- Uses a log odds ratio term appended to the NLL loss
- No reference model needed
- Weak penalty on rejected responses + strong adaptation signal on chosen responses
- **Single-step** — simpler pipeline than DPO
- Can outperform RLHF and DPO on some benchmarks

### 4.4 KTO (Kahneman-Tversky Optimization)

**Paper**: Ethayarajh et al., 2024

Uses **binary** (thumbs up/down) feedback instead of preference pairs:

- Based on prospect theory from behavioral economics
- Much easier data collection — no need for pairwise comparisons
- Robust to noisy labels
- Best when you have point-wise feedback rather than paired preferences

### 4.5 GRPO (Group Relative Policy Optimization)

Used prominently in **DeepSeek-R1** training:

- Eliminates the need for a separate reward model by using group-relative rewards
- Samples a group of outputs per prompt, scores them, and uses relative rankings as the reward signal
- Key enabler of the reasoning model training pipeline

---

## 5. Knowledge Distillation

Transfers knowledge from a large **teacher** model to a smaller **student** model:

- **Logits-based**: Student learns to match teacher's output probability distribution (soft labels)
- **Feature-based**: Student learns to match intermediate representations
- **Data distillation**: Use teacher to generate high-quality training data for student
- **KD-LoRA**: Hybrid — applies LoRA during distillation for further efficiency

**Use cases**: Deploy smaller, faster models that retain most of the teacher's capability. Common in production where inference cost matters.

**Key tools**: NVIDIA TensorRT Model Optimizer, NVIDIA NeMo, Microsoft Azure AI Foundry distillation pipeline.

---

## 6. Continued Pre-Training (CPT)

Further pre-trains a base model on a large corpus of domain-specific unlabeled text **before** fine-tuning:

- Used when the target domain is very different from the pretraining data (e.g., medical, legal, new language)
- **Risk**: Catastrophic forgetting of original capabilities
- **Mitigation**: Replay mixing (include some original data), LoRA-based CPT
- Can be combined with Spectrum for layer selection

**Example**: KnItLM — uses continual pretraining with LoRA to inject new knowledge, then merges LoRA weights with instruction-tuned model without destroying instruction-following capability.

---

## 7. Frameworks & Libraries

### 7.1 Unsloth

- **GitHub**: [github.com/unslothai/unsloth](https://github.com/unslothai/unsloth) (~54K stars)
- **Best for**: Single-GPU fine-tuning, rapid prototyping, limited hardware
- **Speed**: 2-5x faster than standard Hugging Face training; ~24% faster than TorchTune with compile
- **Memory**: ~80% less VRAM usage via custom Triton kernels
- **Features**: No-code web UI, supports QLoRA/DPO/ORPO/KTO/PPO, model export
- **Limitation**: Open-source version only supports single-GPU training
- **Website**: [unsloth.ai](https://unsloth.ai/)

### 7.2 Axolotl

- **GitHub**: [github.com/axolotl-ai-cloud/axolotl](https://github.com/axolotl-ai-cloud/axolotl) (~11.4K stars)
- **Best for**: Config-driven production training, multi-GPU setups, ML researchers
- **Features**: YAML-based configuration, FSDP2 and DeepSpeed integration, LoRA/QLoRA optimizations, multimodal model support (beta)
- **Strength**: Most granular control, extensive configuration options, reproducible experiments
- **Trade-off**: Slightly slower than Unsloth/TorchTune due to abstraction layers over HF Transformers

### 7.3 LLaMA-Factory

- **GitHub**: [github.com/hiyouga/LlamaFactory](https://github.com/hiyouga/LlamaFactory) (~68K stars)
- **Best for**: GUI-first approach, broad model support, low learning curve
- **Features**: Web UI (LLaMA Board), supports 100+ LLMs and VLMs, full-tuning/LoRA/QLoRA (2-8 bit via AQLM/AWQ/GPTQ/LLM.int8/HQQ/EETQ), GaLore, BAdam
- **Strength**: Easiest to get started with; can integrate Unsloth's implementation (`use_unsloth: true`)
- **Latest**: v0.9.4 (Dec 2025)

### 7.4 TRL (Hugging Face)

- **GitHub**: Part of [github.com/huggingface/trl](https://github.com/huggingface/trl) (~17.6K stars)
- **Best for**: RLHF/GRPO training, deep integration with HF ecosystem
- **Features**: DPO, IPO, KTO, ORPO, PPO, GRPO, reward modeling, SFT trainer, Spectrum support
- **Latest**: v0.15.0 (Mar 2026)
- **Trade-off**: Medium-high learning curve; requires understanding of HF Transformers ecosystem

### 7.5 DeepSpeed

- **GitHub**: [github.com/deepspeedai/DeepSpeed](https://github.com/deepspeedai/DeepSpeed) (Microsoft)
- **Best for**: Industrial-scale training across hundreds of GPUs, pre-training from scratch
- **Features**: ZeRO optimization (stages 1-3), offloading, pipeline/tensor/model parallelism
- **Trade-off**: High complexity; overkill for most fine-tuning scenarios; essential for pre-training at scale

### 7.6 TorchTune (Meta)

- **GitHub**: [github.com/pytorch/torchtune](https://github.com/pytorch/torchtune)
- **Best for**: PyTorch-native fine-tuning with compile optimizations
- **Features**: Built on native PyTorch (not HF Transformers), supports LoRA/QLoRA, distributed training
- **Trade-off**: Less model coverage than HF-based frameworks

### 7.7 PEFT (Hugging Face)

- **GitHub**: [github.com/huggingface/peft](https://github.com/huggingface/peft)
- **The foundational library** for parameter-efficient methods
- **Supports**: LoRA, QLoRA, Adapters, Prefix Tuning, IA³, Prompt Tuning, DoRA, and more
- Used under the hood by Unsloth, Axolotl, TRL, and LLaMA-Factory

### Framework Comparison Summary

| Framework | Stars | Best For | Learning Curve | Multi-GPU | Speed |
|---|---|---|---|---|---|
| **LLaMA-Factory** | 68K | GUI, broad support | Low | Yes | Good |
| **Unsloth** | 54K | Speed, single-GPU, beginners | Low-Med | No (OSS) | Best |
| **TRL** | 18K | RLHF/GRPO, HF ecosystem | Med-High | Yes | Good |
| **Axolotl** | 11K | Config-driven, production, research | Medium | Yes (FSDP2/DS) | Good |
| **DeepSpeed** | — | Pre-training, massive scale | High | Yes (best) | N/A |
| **TorchTune** | — | PyTorch-native, compile | Medium | Yes | Very Good |

---

## 8. Key Papers

### Foundational
| Paper | Year | Contribution |
|---|---|---|
| [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685) | 2022 | The foundational PEFT method |
| [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) | 2023 | 4-bit quantization + LoRA |
| [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) | 2023 | Eliminates RL from alignment |
| [Delta Tuning: A Comprehensive Study of PEFT](https://arxiv.org/abs/2203.06904) | 2022 | Early PEFT survey |

### Advanced PEFT Methods
| Paper | Year | Contribution |
|---|---|---|
| [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353) | 2024 | Magnitude + direction decomposition (ICML 2024 Oral) |
| [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507) | 2024 | Low-rank gradients for full-param training (ICML 2024) |
| [Spectrum](https://arxiv.org/abs/2408.10156) | 2024 | SNR-based layer selection |
| [PiSSA](https://arxiv.org/abs/2404.02948) | 2024 | SVD-initialized LoRA |
| [SparseLoRA](https://arxiv.org/abs/2506.16500) | 2025 | Contextual sparsity |

### Alignment Methods
| Paper | Year | Contribution |
|---|---|---|
| [ORPO: Monolithic Preference Optimization](https://arxiv.org/abs/2403.07691) | 2024 | Single-step SFT + alignment |
| [KTO: Model Alignment as Prospect Theory Optimization](https://arxiv.org/abs/2402.01306) | 2024 | Binary feedback alignment |
| [Is DPO Superior to PPO for LLM Alignment?](https://arxiv.org/abs/2404.10719) | 2024 | Comprehensive DPO vs PPO study |
| [A Comprehensive Survey of LLM Alignment Techniques](https://arxiv.org/abs/2407.16216) | 2024 | RLHF, RLAIF, PPO, DPO survey |

### Surveys & Analyses
| Paper | Year | Contribution |
|---|---|---|
| [Learning Rate Matters: Vanilla LoRA May Suffice](https://arxiv.org/abs/2602.04998) | 2025 | LR tuning > method complexity |
| [LoRA vs Full Fine-tuning: An Illusion of Equivalence](https://openreview.net/forum?id=xp7B8rkh7L) | 2025 | LoRA and full FT find different solutions (NeurIPS 2025) |
| [A Comprehensive Survey of PEFT for LLMs](https://arxiv.org/abs/2403.14608) | 2024 | Full PEFT taxonomy |
| [Parameter-efficient fine-tuning: methodologies survey](https://link.springer.com/article/10.1007/s10462-025-11236-4) | 2025 | Springer survey |
| [Fine-tuning and Utilization Methods of Domain-specific LLMs](https://arxiv.org/abs/2401.02981) | 2024 | Domain-specific FT survey |
| [EfficientLLM: Evaluation on Architecture, Pretraining, Fine-Tuning](https://arxiv.org/abs/2505.13840) | 2025 | Benchmarks LoRA/DoRA/PISSA/etc across 7 model architectures |

---

## 9. Notable GitHub Repos

### Frameworks & Tools
| Repo | Description |
|---|---|
| [hiyouga/LlamaFactory](https://github.com/hiyouga/LlamaFactory) | Unified fine-tuning of 100+ LLMs with Web UI |
| [unslothai/unsloth](https://github.com/unslothai/unsloth) | 2-5x faster fine-tuning with 80% less memory |
| [axolotl-ai-cloud/axolotl](https://github.com/axolotl-ai-cloud/axolotl) | Config-driven fine-tuning framework |
| [huggingface/peft](https://github.com/huggingface/peft) | HF library for PEFT methods |
| [huggingface/trl](https://github.com/huggingface/trl) | HF library for RLHF, DPO, GRPO, etc. |
| [deepspeedai/DeepSpeed](https://github.com/deepspeedai/DeepSpeed) | Microsoft distributed training |
| [pytorch/torchtune](https://github.com/pytorch/torchtune) | PyTorch-native LLM fine-tuning |
| [NVlabs/DoRA](https://github.com/NVlabs/DoRA) | Official DoRA implementation |

### Awesome Lists & Curated Resources
| Repo | Description |
|---|---|
| [Curated-Awesome-Lists/awesome-llms-fine-tuning](https://github.com/Curated-Awesome-Lists/awesome-llms-fine-tuning) | Comprehensive resource collection |
| [pdaicode/awesome-LLMs-finetuning](https://github.com/pdaicode/awesome-LLMs-finetuning) | Papers, tools, datasets |
| [hannibal046/Awesome-LLM](https://github.com/hannibal046/Awesome-LLM) | General LLM resource list |
| [horseee/Awesome-Efficient-LLM](https://github.com/horseee/Awesome-Efficient-LLM) | Efficient LLM methods |
| [Mr-Loevan/DPO-Survey](https://github.com/Mr-Loevan/DPO-Survey) | DPO papers and variants |
| [rkinas/reasoning_models_how_to](https://github.com/rkinas/reasoning_models_how_to) | Reasoning model training methods (PPO, DPO, KTO, ORPO) |

### Practical Notebooks & Guides
| Repo | Description |
|---|---|
| [philschmid/deep-learning-pytorch-huggingface](https://github.com/philschmid/deep-learning-pytorch-huggingface) | Phil Schmid's fine-tune-llms-in-2025 notebook |
| [ashishpatel26/LLM-Finetuning](https://github.com/ashishpatel26/LLM-Finetuning) | 20+ fine-tuning notebooks |
| [Tushar-Sethi/LLM-FineTuning](https://github.com/Tushar-Sethi/LLM-FineTuning) | Various FT technique notebooks |
| [rohan-paul/LLM-FineTuning-Large-Language-Models](https://github.com/rohan-paul/LLM-FineTuning-Large-Language-Models) | Practical FT examples |

---

## 10. Decision Guide: Which Method to Use

### By Scenario

```
Do you need to inject new factual knowledge?
  YES → Continued Pre-Training (CPT) + SFT
  NO ↓

Do you need the model to follow specific output formats/styles?
  YES → SFT with LoRA or QLoRA
  NO ↓

Do you need to align with human preferences/values?
  YES → Have pairwise preference data?
          YES → DPO (easiest) or RLHF/PPO (most control)
          NO → Have binary feedback?
                  YES → KTO
                  NO → Can you generate preference data?
                          YES → ORPO or GRPO
                          NO → Start with SFT
  NO ↓

Do you need a smaller/faster model for deployment?
  YES → Knowledge distillation + quantization
  NO → You might not need fine-tuning — try prompting or RAG first
```

### By Hardware

| Hardware | Recommended Approach |
|---|---|
| Consumer GPU (8-16GB) | QLoRA (r=8-16) via Unsloth |
| Single GPU (24-48GB) | QLoRA (r=16-64) or LoRA via Unsloth/Axolotl |
| Multi-GPU (2-8x A100) | LoRA or full FT via Axolotl + DeepSpeed/FSDP |
| Cluster (8+ GPUs) | Full FT + DeepSpeed ZeRO-3 |

### By Data Availability

| Data | Recommended Approach |
|---|---|
| <100 examples | Prompt engineering / few-shot (fine-tuning won't help much) |
| 100-1K examples | LoRA/QLoRA SFT |
| 1K-10K examples | LoRA/QLoRA SFT + preference alignment (DPO/KTO) |
| 10K-100K examples | Full FT or LoRA with higher rank |
| 100K+ examples | CPT + SFT + alignment pipeline |

---

## Key Takeaways

1. **LoRA/QLoRA is sufficient for ~95% of production fine-tuning needs**. Proper learning rate tuning matters more than choosing a fancy variant.
2. **DPO has largely replaced RLHF** for practical alignment due to simplicity, though frontier labs still use RLHF/PPO for maximum control.
3. **Unsloth is the fastest for single-GPU**; **Axolotl or LLaMA-Factory** for multi-GPU; **DeepSpeed** for pre-training at scale.
4. **The fine-tuning vs RAG decision**: If your task requires new *behavior* (format, style, reasoning pattern), fine-tune. If it needs new *knowledge*, try RAG first — it's cheaper and more maintainable.
5. **Full fine-tuning still matters** for large domain shifts, but the gap with PEFT is narrowing rapidly with methods like DoRA and GaLore.
6. **GRPO is emerging as important** for reasoning model training (DeepSeek-R1 path) — no reward model needed.

---

*Last updated: 2026-04-10*
