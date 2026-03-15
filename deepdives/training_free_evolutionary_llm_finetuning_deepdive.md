# Training-Free & Evolutionary Approaches to GRPO-Type Fine-Tuning: A Deep Dive

**Research Date:** 2026-03-15
**Focus:** Non-gradient optimization methods for LLM post-training and RL fine-tuning

---

## Executive Summary

The landscape of LLM fine-tuning is undergoing a paradigm shift. While gradient-based methods (PPO, GRPO, DPO) have dominated post-training, a new wave of **training-free** and **evolutionary approaches** is emerging as a compelling alternative. These methods offer:

1. **Massive cost reduction** (100x cheaper in some cases)
2. **Better scalability** (6x faster convergence at scale)
3. **Improved stability** (no gradient explosions, reward hacking resistance)
4. **Privacy guarantees** (no gradient leakage)
5. **Hardware flexibility** (works on quantized models, no backprop needed)

---

## Table of Contents

1. [Training-Free GRPO](#1-training-free-grpo)
2. [Evolution Strategies at Scale](#2-evolution-strategies-at-scale)
3. [ESSA: Evolutionary Strategies for Scalable Alignment](#3-essa-evolutionary-strategies-for-scalable-alignment)
4. [EA4LLM: Evolutionary Algorithms for LLM Optimization](#4-ea4llm-evolutionary-algorithms-for-llm-optimization)
5. [BBoxER: Black-Box Evolutionary Retrofitting](#5-bboxer-black-box-evolutionary-retrofitting)
6. [Zeroth-Order Optimization Methods](#6-zeroth-order-optimization-methods)
7. [Test-Time Learning Approaches](#7-test-time-learning-approaches)
8. [Comparative Analysis](#8-comparative-analysis)
9. [Key Insights & Future Directions](#9-key-insights--future-directions)
10. [References](#10-references)

---

## 1. Training-Free GRPO

**Paper:** [Training-Free Group Relative Policy Optimization](https://arxiv.org/abs/2510.08191)
**Authors:** Yuzheng Cai et al. (Tencent Cloud AI)
**Date:** October 2025

### Core Innovation

Training-Free GRPO fundamentally reimagines policy optimization by shifting from **parameter space** to **context space**. Instead of updating model weights, it maintains a frozen LLM and learns an external "experiential knowledge" library.

### Key Mechanism

```
Traditional GRPO:  θ_new = θ_old + α∇J(θ)  [Gradient updates]
Training-Free GRPO: context_new = context_old + distilled_experience  [In-context learning]
```

**The Semantic Group Advantage:**
- Replaces numerical group advantages with **semantic** advantages
- Uses LLM introspection to extract high-quality experiences from successful rollouts
- Distills these as token priors integrated during API calls

### Algorithm Overview

1. **Group Sampling:** Generate multiple responses per prompt (like GRPO)
2. **Semantic Advantage:** Use LLM to introspect on what made successful responses work
3. **Experience Distillation:** Extract patterns into structured experiential knowledge
4. **Token Prior Integration:** Seamlessly inject during inference (no weight changes)

### Performance Highlights

| Metric | Result |
|--------|--------|
| Training samples needed | **~100 samples** (vs thousands for fine-tuning) |
| Cost reduction | **$8 vs $800** (100x cheaper than full fine-tuning) |
| Applied to DeepSeek-V3.1-Terminus | Outperforms fully fine-tuned 32B LLM |
| Out-of-domain generalization | Significantly improved |

### When to Use

- **Limited training data** (dozens of samples)
- **Frozen API-based models** (no access to weights)
- **Rapid iteration** needed
- **Cost-sensitive** deployments
- **Multi-tenant** scenarios (one model, many adaptations)

---

## 2. Evolution Strategies at Scale

**Paper:** [Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning](https://arxiv.org/abs/2509.24372)
**Authors:** Xin Qiu, Yulu Gan, Conor F. Hayes et al. (Cognizant AI Lab)
**Date:** September 2025 (revised February 2026)

### Breakthrough Finding

> *"This paper overturns the widespread belief that Evolution Strategies does not scale to modern model sizes."*

First successful application of ES to **full-parameter fine-tuning** of billion-parameter LLMs without dimensionality reduction.

### Why ES Works for LLMs

The paper reveals several surprising properties:

1. **High-dimensional search is tractable** - ES can search over billions of parameters
2. **Natural robustness** - Better tolerance to long-horizon and delayed rewards
3. **Reward hacking resistance** - Reduced susceptibility compared to RL
4. **Training stability** - No gradient explosions or vanishing gradients

### ES vs RL Comparison

| Property | Evolution Strategies | Reinforcement Learning |
|----------|---------------------|------------------------|
| Gradient access | Not required | Required |
| Memory footprint | Lower (no optimizer states) | Higher (Adam states, gradients) |
| Parallelization | Embarrassingly parallel | Requires synchronization |
| Reward horizon | Better for delayed rewards | Struggles with long horizons |
| Reward hacking | More resistant | Susceptible |
| Variance | Higher per-sample | Lower per-sample |
| Sample efficiency | Lower | Higher |

### Implementation Insights

```python
# Simplified ES for LLM fine-tuning
def es_finetune(model, reward_fn, n_generations):
    θ = model.parameters()  # Current parameters

    for gen in range(n_generations):
        # 1. Sample perturbations
        ε_i = [sample_noise(θ) for i in range(population_size)]

        # 2. Evaluate fitness (parallel forward passes)
        F_i = [reward_fn(model, θ + σ*ε) for ε in ε_i]

        # 3. Update parameters (weighted by fitness)
        θ = θ + α * Σ(F_i - mean(F)) * ε_i / (std(F) * population_size)

    return θ
```

### Performance Results

| Model | Task | ES vs GRPO |
|-------|------|------------|
| Qwen2.5-0.5B to 8B | GSM8K | **ES consistently outperforms** |
| LLaMA-0.5B to 8B | Reasoning | **ES consistently outperforms** |
| Cross-seed variance | All | **6.4% vs 43.3%** (ES much more stable) |

### Key Advantages Over RL

1. **No reward shaping needed** - Works with sparse, delayed rewards
2. **No critic model** - Eliminates value function approximation errors
3. **Better exploration** - Population-based search covers more of parameter space
4. **Hardware-friendly** - Pure forward passes, highly parallelizable

---

## 3. ESSA: Evolutionary Strategies for Scalable Alignment

**Paper:** [ESSA: Evolutionary Strategies for Scalable Alignment](https://arxiv.org/abs/2507.04453)
**Authors:** Daria Korotyshova, Boris Shaposhnikov et al.
**Date:** July 2025 (revised December 2025)

### Core Innovation

ESSA combines ES with **dimensionality reduction** via LoRA + SVD compression, making evolutionary search practical for very large models.

### Architecture

```
Full Model → LoRA Adapters → SVD Decomposition → Optimize Singular Values Only
                                    ↓
                            Much smaller search space
```

**Key insight:** Instead of optimizing all LoRA parameters, only optimize the **singular values** from SVD decomposition of adapter matrices.

### Technical Details

1. **Low-Rank Adaptation:** Apply LoRA to reduce parameter count
2. **SVD Compression:** Decompose adapter matrices: `W = U Σ V^T`
3. **Singular Value Optimization:** Only evolve Σ (diagonal matrix)
4. **Quantization Support:** Works in INT4 and INT8 inference mode

### Performance Benchmarks

| Model | Benchmark | Improvement over GRPO |
|-------|-----------|----------------------|
| Qwen2.5-Math-7B | GSM8K | **+12.6%** accuracy |
| Qwen2.5-Math-7B | PRM800K | **+14.8%** accuracy |
| LLaMA3.1-8B | IFEval | **+22.5%** accuracy |

### Scaling Behavior (Critical Finding)

| GPU Count | Task | Time to Near-Optimal |
|-----------|------|---------------------|
| 16 GPUs | Qwen2.5-32B PRM800K | **2x faster** than GRPO |
| 128 GPUs | Qwen2.5-32B PRM800K | **6x faster** than GRPO |

> *"ESSA shows stronger scaling than gradient-based methods at scale."*

### Why ESSA Scales Better

1. **No gradient synchronization** - Each worker evaluates independently
2. **No optimizer state communication** - Just fitness values
3. **Memory efficient** - Only need model + noise, not gradients
4. **Quantization compatible** - Can run on smaller hardware

---

## 4. EA4LLM: Evolutionary Algorithms for LLM Optimization

**Paper:** [EA4LLM: A Gradient-Free Approach to Large Language Model Optimization via Evolutionary Algorithms](https://arxiv.org/abs/2510.10603)
**Authors:** WenTao Liu, Siyu Song, Hao Hao, Aimin Zhou
**Date:** October 2025

### Ambitious Claim

> *"First time empirically verifying full-parameter optimization from the pretraining stage across model sizes ranging from 0.5B to 32B."*

This goes beyond fine-tuning - EA4LLM claims to train LLMs from scratch using only evolutionary algorithms.

### Key Contributions

1. **Pre-training capability:** Can optimize from random initialization
2. **Scale validation:** Tested on 0.5B to 32B parameters
3. **Non-differentiable architectures:** Opens door to new model designs

### Implications

| Traditional Constraint | EA4LLM Freedom |
|----------------------|----------------|
| All operations must be differentiable | Any operation allowed |
| Requires high-memory GPUs | More hardware flexibility |
| Complex backpropagation | Simple forward passes |
| Specific architecture requirements | Architecture-agnostic |

### Challenges Addressed

- **High variance** of ES estimators (mitigated through population strategies)
- **Sample efficiency** (improved via intelligent mutation strategies)
- **Computational cost** (reduced through parallelization)

---

## 5. BBoxER: Black-Box Evolutionary Retrofitting

**Paper:** [Tuning without Peeking: Provable Generalization Bounds and Robust LLM Post-Training](https://arxiv.org/abs/2507.01752)
**Authors:** Ismail Labiad, Mathurin Videau et al.
**Date:** July 2025 (revised January 2026)

### Unique Positioning

BBoxER is designed as an **add-on** on top of gradient-based optimization, not a replacement. It provides:

1. **Provable generalization bounds** (non-vacuous)
2. **Privacy guarantees** (no gradient leakage)
3. **Robustness to data poisoning**
4. **Resistance to extraction attacks**

### Information Bottleneck Principle

```
Training Data → [BBoxER Compression] → Compressed Representation → Model Update
                        ↑
                  Information Bottleneck
                  (Prevents overfitting, ensures generalization)
```

### Theoretical Guarantees

| Guarantee | Mechanism |
|-----------|-----------|
| Generalization | Implicit compression creates information bottleneck |
| Privacy | No gradients exposed → no gradient-based attacks |
| Robustness | Population-based → outliers have limited impact |
| Membership inference resistance | Provable bounds on information leakage |

### Use Cases

- **Privacy-sensitive deployments** (healthcare, finance)
- **Adversarial environments** (potential data poisoning)
- **Regulated industries** (need provable guarantees)
- **Fine-tuning as service** (isolate client data)

---

## 6. Zeroth-Order Optimization Methods

### Overview

Zeroth-order (ZO) optimization estimates gradients using only function evaluations, enabling fine-tuning without backpropagation.

### Key Papers

1. **[Learning a Zeroth-Order Optimizer for Fine-Tuning LLMs](https://arxiv.org/abs/2510.00419)** (October 2025)
   - Introduces **ZO Fine-tuner** with learnable direction sampling
   - Adapts to model-specific structures

2. **[Zero-Order Optimization for LLM Fine-Tuning via Learnable Perturbation Distribution](https://arxiv.org/abs/2602.13659)** (February 2026)
   - **LOREN**: Curvature-aware ZO optimization
   - Reformulates gradient preconditioning as perturbation distribution estimation

3. **[Performance Study of Modern Zeroth-Order Optimization Methods for LLM Fine-Tuning](https://link.springer.com/article/10.3103/S1060992X25601770)**
   - Comparative analysis of **12 ZO methods**
   - Evaluated by memory, quality, time, convergence

4. **[Zeroth-Order Fine-Tuning of LLMs in Random Subspaces](https://openaccess.thecvf.com/content/ICCV2025/papers/Yu_Zeroth-Order_Fine-Tuning_of_LLMs_in_Random_Subspaces_ICCV_2025_paper.pdf)**
   - Reduces gradient estimation variance
   - Minimizes memory overhead

5. **[Quantized Zeroth-Order Fine-Tuning (QuZO)](https://aclanthology.org/2025.emnlp-main.271.pdf)**
   - Works with quantized models (INT4/INT8)
   - 18x VRAM reduction vs regular fine-tuning

### ZO vs ES: Key Differences

| Property | Zeroth-Order | Evolution Strategies |
|----------|--------------|---------------------|
| Gradient estimation | Explicit (finite differences) | Implicit (fitness weighting) |
| Perturbation strategy | Often random | Often learned/adaptive |
| Typical use | Memory-efficient fine-tuning | Full optimization |
| Scale | LoRA/adapter level | Full parameters possible |

### Memory Comparison

| Method | Memory Overhead |
|--------|-----------------|
| Standard fine-tuning | 100% (gradients + optimizer states) |
| ZO optimization | ~20-30% (model + perturbations) |
| QuZO (quantized) | ~5-6% (18x reduction) |

---

## 7. Test-Time Learning Approaches

### Overview

Test-time learning adapts models during inference without explicit fine-tuning.

### Key Papers

1. **[Test-Time Learning for Large Language Models (TLM)](https://arxiv.org/abs/2505.20633)**
   - Formulates TTL as **input perplexity minimization**
   - Self-supervised enhancement during deployment

2. **[TAO: Test-time Adaptive Optimization](https://www.databricks.com/blog/tao-using-test-time-compute-train-efficient-llms-without-labeled-data)** (Databricks)
   - Uses test-time compute + RL
   - No labeled data required

3. **[EvoTest: Evolutionary Test-Time Learning](https://openreview.net/forum?id=JFnnajbkvP)**
   - Evolves entire agentic system after each episode
   - No fine-tuning or gradients required

4. **[Grounded Test-Time Adaptation (GTTA)](https://arthurchen189.github.io/projects/gtta/)**
   - Adapts agents during deployment
   - Stays anchored to factual knowledge

### Evolution vs Test-Time Learning

| Aspect | Evolutionary Methods | Test-Time Learning |
|--------|---------------------|-------------------|
| When applied | Training phase | Deployment phase |
| Data access | Training set | Live inputs |
| Adaptation speed | Generations | Per-query |
| Typical goal | Model improvement | Domain adaptation |

---

## 8. Comparative Analysis

### Performance Summary

| Method | Type | Memory | Speed | Stability | Best For |
|--------|------|--------|-------|-----------|----------|
| **Training-Free GRPO** | In-context | Minimal | Fast | High | API models, limited data |
| **ES at Scale** | Full ES | Medium | Medium | Very High | Full fine-tuning, delayed rewards |
| **ESSA** | ES + LoRA | Low | Fast | Very High | Large models, limited hardware |
| **EA4LLM** | Full EA | High | Slow | High | Pre-training, non-diff architectures |
| **BBoxER** | Black-box ES | Medium | Medium | Very High | Privacy-sensitive, regulated |
| **ZO Methods** | Gradient-free | Low | Medium | Medium | Memory-constrained, quantized |
| **Test-Time Learning** | Online | Minimal | Variable | Medium | Deployment adaptation |

### Cost Comparison

| Method | Relative Cost | Sample Efficiency |
|--------|---------------|-------------------|
| PPO/GRPO | 100% (baseline) | High |
| Training-Free GRPO | **1%** | Very High (100 samples) |
| ES at Scale | 50-80% | Medium |
| ESSA | 30-50% | Medium-High |
| BBoxER | 40-60% | Medium |

### Stability Comparison (Cross-Seed Variance)

| Method | Variance (RSE) |
|--------|----------------|
| GRPO | 43.3% |
| ES at Scale | **6.4%** |
| PPO | ~35-50% |

---

## 9. Key Insights & Future Directions

### Paradigm Shift Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM Fine-Tuning Evolution                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Era 1: Gradient-Based (2022-2024)                              │
│  ├── PPO: Actor-critic, complex but effective                   │
│  ├── DPO: Offline, simpler but limited                          │
│  └── GRPO: No critic, group-relative advantages                 │
│                                                                  │
│  Era 2: Gradient-Free (2025-2026)                               │
│  ├── Training-Free GRPO: In-context optimization                │
│  ├── ES at Scale: Full parameter ES fine-tuning                 │
│  ├── ESSA: LoRA + SVD compressed ES                             │
│  └── BBoxER: Privacy-preserving black-box                       │
│                                                                  │
│  Era 3: Hybrid & Test-Time (Emerging)                           │
│  ├── Test-time compute + evolution                              │
│  ├── Gradient + black-box combinations                          │
│  └── Continuous deployment adaptation                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Insights

1. **ES scales surprisingly well** - The belief that ES doesn't scale is wrong
2. **Training-free is viable** - In-context learning can replace weight updates
3. **Stability matters** - ES variance (6.4%) vs GRPO (43.3%) is a huge advantage
4. **Hardware flexibility** - No backprop means more deployment options
5. **Privacy by design** - Black-box methods have natural privacy guarantees

### Open Questions

1. **Hybrid methods:** Can we combine gradient-based and gradient-free optimally?
2. **Sample efficiency:** Can ES match RL's sample efficiency?
3. **Architecture search:** Will non-differentiable architectures become mainstream?
4. **Theoretical understanding:** Why does ES work so well for LLMs?

### Future Directions

1. **ES + RL Hybrids:** Use ES for exploration, RL for exploitation
2. **Adaptive ES:** Learn mutation strategies per-parameter
3. **Multi-objective ES:** Optimize multiple metrics simultaneously
4. **Distributed ES:** Leverage massive parallelism more effectively
5. **ES for Architecture:** Search over non-differentiable model components

---

## 10. References

### Primary Papers (Training-Free & Evolutionary)

1. Cai, Y. et al. (2025). [Training-Free Group Relative Policy Optimization](https://arxiv.org/abs/2510.08191). arXiv:2510.08191.

2. Qiu, X. et al. (2025). [Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning](https://arxiv.org/abs/2509.24372). arXiv:2509.24372.

3. Korotyshova, D. et al. (2025). [ESSA: Evolutionary Strategies for Scalable Alignment](https://arxiv.org/abs/2507.04453). arXiv:2507.04453.

4. Liu, W. et al. (2025). [EA4LLM: A Gradient-Free Approach to Large Language Model Optimization via Evolutionary Algorithms](https://arxiv.org/abs/2510.10603). arXiv:2510.10603.

5. Labiad, I. et al. (2025). [Tuning without Peeking: Provable Generalization Bounds and Robust LLM Post-Training](https://arxiv.org/abs/2507.01752). arXiv:2507.01752.

### Zeroth-Order Optimization

6. [Learning a Zeroth-Order Optimizer for Fine-Tuning LLMs](https://arxiv.org/abs/2510.00419). arXiv:2510.00419.

7. [Zero-Order Optimization for LLM Fine-Tuning via Learnable Perturbation Distribution](https://arxiv.org/abs/2602.13659). arXiv:2602.13659.

8. [Performance Study of Modern Zeroth-Order Optimization Methods for LLM Fine-Tuning](https://link.springer.com/article/10.3103/S1060992X25601770).

9. [Zeroth-Order Fine-Tuning of LLMs in Random Subspaces](https://openaccess.thecvf.com/content/ICCV2025/papers/Yu_Zeroth-Order_Fine-Tuning_of_LLMs_in_Random_Subspaces_ICCV_2025_paper.pdf). ICCV 2025.

### Test-Time Learning

10. [Test-Time Learning for Large Language Models](https://arxiv.org/abs/2505.20633). arXiv:2505.20633.

11. [TAO: Test-time Adaptive Optimization](https://www.databricks.com/blog/tao-using-test-time-compute-train-efficient-llms-without-labeled-data). Databricks Blog.

12. [EvoTest: Evolutionary Test-Time Learning](https://openreview.net/forum?id=JFnnajbkvP). OpenReview.

### Background on GRPO

13. Shao, Z. et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. (Original GRPO paper)

14. [GRPO: Group Relative Policy Optimization](https://cameronrwolfe.substack.com/p/grpo). Cameron R. Wolfe, Substack.

15. [GRPO++: Tricks for Making RL Actually Work](https://cameronrwolfe.substack.com/p/grpo-tricks). Cameron R. Wolfe, Substack.

---

## Appendix: Quick Reference Cards

### Training-Free GRPO Quick Start

```python
# Conceptual implementation
class TrainingFreeGRPO:
    def __init__(self, llm_api, experience_bank):
        self.llm = llm_api  # Frozen LLM
        self.experiences = experience_bank

    def optimize(self, prompt, n_samples=8):
        # 1. Generate group of responses
        responses = [self.llm.generate(prompt) for _ in range(n_samples)]

        # 2. Evaluate semantic advantage
        best_response = self.semantic_advantage(responses)

        # 3. Extract and distill experience
        experience = self.llm.introspect(prompt, best_response)
        self.experiences.add(experience)

        # 4. Use as token prior in next calls
        return self.llm.generate(prompt, context=self.experiences)
```

### ES Fine-Tuning Quick Start

```python
# Simplified ES for LLM fine-tuning
def es_finetune(model, reward_fn, config):
    θ = model.parameters()
    σ = config.noise_std
    α = config.learning_rate
    N = config.population_size

    for generation in range(config.n_generations):
        # Sample noise
        noises = [torch.randn_like(θ) for _ in range(N)]

        # Evaluate fitness (parallel)
        fitness = []
        for ε in noises:
            perturbed = θ + σ * ε
            fitness.append(reward_fn(model.with_params(perturbed)))

        # Normalize fitness
        fitness = (fitness - mean(fitness)) / std(fitness)

        # Update parameters
        θ = θ + α / (N * σ) * sum(f * ε for f, ε in zip(fitness, noises))

    return θ
```

### When to Choose Which Method

```
┌─────────────────────────────────────────────────────────────────┐
│                    Decision Tree                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Do you have access to model weights?                           │
│  ├── NO → Training-Free GRPO                                    │
│  └── YES ↓                                                       │
│                                                                  │
│  Do you need privacy guarantees?                                │
│  ├── YES → BBoxER                                               │
│  └── NO ↓                                                        │
│                                                                  │
│  Is memory constrained?                                         │
│  ├── YES → ESSA or ZO Methods                                   │
│  └── NO ↓                                                        │
│                                                                  │
│  Do you need full fine-tuning?                                  │
│  ├── YES → ES at Scale                                          │
│  └── NO (adapters OK) → ESSA                                    │
│                                                                  │
│  Is model quantized?                                            │
│  ├── YES → QuZO or ESSA                                         │
│  └── NO → Any method                                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

*End of Deep Dive*
