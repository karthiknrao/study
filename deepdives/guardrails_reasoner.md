# Reasoning-based LLM Safeguards — Reference

A curated list of recent papers related to **GuardReasoner: Towards Reasoning-based LLM Safeguards** ([arXiv:2501.18492](https://arxiv.org/abs/2501.18492)), which trains guard models to produce explicit reasoning chains before safety verdicts.

## Source paper

- **GuardReasoner** — Yue Liu, Hongcheng Gao, Shengfang Zhai, Jun Xia, Tianyi Wu, Zhiwei Xue, Yulin Chen, Kenji Kawaguchi, Jiaheng Zhang, Bryan Hooi. [arXiv:2501.18492](https://arxiv.org/abs/2501.18492) (Jan 2025; v2 Oct 2025). Reasoning SFT + Hard-Sample DPO on 127K samples / 460K reasoning steps. ICLR Workshop 2025. [Code](https://github.com/yueliu1999/GuardReasoner).

## Direct descendants / same authors

- **GuardReasoner-VL** — Yue Liu et al. [arXiv:2505.11049](https://arxiv.org/abs/2505.11049) (May 2025). Extends the framework to vision-language models; 123K samples / 631K reasoning steps spanning text, image, and text-image; trained with online RL. NeurIPS 2025. [Code](https://github.com/yueliu1999/GuardReasoner-VL).

## Other reasoning-based guard models

- **GSPR: Aligning LLM Safeguards as Generalizable Safety Policy Reasoners** — Haoran Li, Yulin Chen et al. [arXiv:2509.24418](https://arxiv.org/abs/2509.24418) (Sept 2025). Group Relative Policy Optimization over varied fine-grained safety taxonomies from multiple benchmarks; lower inference-token cost.
- **IntentionReasoner: Facilitating Adaptive LLM Safeguards through Intent Reasoning and Selective Query Refinement** — Yuanzhe Shen, Zisu Huang et al. [arXiv:2508.20151](https://arxiv.org/abs/2508.20151) (Aug 2025). ~163K annotated queries; intent reasoning + safety classification + query rewriting to reduce over-refusal.
- **RSafe: Incentivizing proactive reasoning to build robust and adaptive LLM safeguards** — [arXiv:2506.07736](https://arxiv.org/abs/2506.07736) (June 2025). RL-based alignment for proactive safety prediction.
- **ThinkGuard** — Wen et al., 2025. CoT rationales from expert models for stronger generalization (cited in later work).
- **R2-Guard** — Kang & Li, 2024/2025. Probabilistic graphical-model reasoning over structured safety knowledge; earlier work that GuardReasoner builds on.
- **MrGuard: A Multilingual Reasoning Guardrail for Universal LLM Safety** — Yang et al. ([cogcomp paper](https://cogcomp.seas.upenn.edu/papers/YDLRL25.pdf)). Synthetic multilingual supervision + curriculum-guided GRPO.

## Multilingual / efficient extensions

- **ConsistentGuard — Unlocking LLM Safeguards for Low-Resource Languages via Reasoning and Alignment with Minimal Training Data** — Zhuowei Chen, Bowei Zhang et al. [arXiv:2510.10677](https://arxiv.org/abs/2510.10677) (Oct 2025). Reasoning-based multilingual safeguard trained on only ~1K samples; evaluated on 3 datasets across 6 languages.
- **PSRT: Accelerating LRM-based Guard Models via Prefilled Safe Reasoning Traces** — [arXiv:2509.21768](https://arxiv.org/abs/2509.21768) (Sept 2025). Pre-filled reasoning traces to cut inference cost for reasoning-guard models.
- **Robust and Efficient Guardrails with Latent Reasoning** — [arXiv:2605.29068](https://arxiv.org/abs/2605.29068) (May 2026). Latent (non-CoT) reasoning for cheaper, more robust guardrails.

## Empirical studies, surveys, analyses

- **Safety Through Reasoning: An Empirical Study of Reasoning Guardrail Models** — [arXiv:2505.20087](https://arxiv.org/abs/2505.20087) (May 2025). Findings of EMNLP 2025.
- **SoK: Evaluating Jailbreak Guardrails for Large Language Models** — [arXiv:2506.10597](https://arxiv.org/abs/2506.10597) (June 2025). Systematizes the guardrail landscape; cites GuardReasoner and SelfDefend.
- **Rethinking LLM Safety Reasoning through the Lens of …** — [arXiv:2509.22250](https://arxiv.org/abs/2509.22250) (Sept 2025).
- **Safety-Aware Reasoning Can Defend Large Language Models** — [EMNLP 2025 main](https://aclanthology.org/2025.emnlp-main.1493.pdf).

## Local files

- `GuardReasoner-Towards-Reasoning-based-LLM-Safeguards.pdf` — the source paper.
