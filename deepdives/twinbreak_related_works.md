# Papers & Repos Similar to TwinBreak (arXiv 2506.07596)

> **Source paper:** *TwinBreak: Jailbreaking LLM Security Alignments based on Twin Prompts*
> Krauß, Dashtbani & Dmitrienko — USENIX Security 2025
> Core idea: treat the LLM safety mechanism as an embedded backdoor and prune the "refusal" parameters in open-weight LLMs by comparing activation patterns between highly similar harmful + harmless ("twin") prompts.

---

## 1. The Paper & Its Code

| Item | Link | Notes |
|------|------|-------|
| arXiv | https://arxiv.org/abs/2506.07596 | |
| USENIX version (PDF) | https://www.usenix.org/system/files/usenixsecurity25-krauss.pdf | |
| HTML (v1, full appendix) | https://arxiv.org/html/2506.07596v1 | |
| **Official code** | https://github.com/tkr-research/twinbreak | USENIX artifact (Python). Models: LLaMA-2, LLaMA-3.1, Gemma-2/3, Qwen-2.5, Mistral, DeepSeek (1B–72B). |
| TwinPrompt dataset | bundled in the repo under `dataset/json/twinprompt.json` | 100 harmful/harmless twin pairs based on HarmBench |
| HF abliterated reimplementation | https://huggingface.co/CSMaya/er_ablations_qwen_2.5-3B_twinbreak | Community "abliteration" build on top of TwinBreak |
| Promptfoo vulnerability card | https://www.promptfoo.dev/lm-security-db/vuln/twin-prompt-jailbreak-2041ba42 | Security-DB entry |
| Lit review | https://www.themoonlight.io/en/review/twinbreak-jailbreaking-llm-security-alignments-based-on-twin-prompts | |

---

## 2. Closest Siblings (White-Box Pruning / Safety-Parameter Removal)

These are the papers that use the same fundamental approach (find & surgically remove the safety-related parameters).

### Foundational predecessor
- **Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications**
  Wei, Huang, Huang, Xie, Qi, Xia, Mittal, Wang, Henderson — arXiv:2402.05162
  https://arxiv.org/abs/2402.05162
  Showed safety-critical regions are only ~3% of weights / ~2.5% of ranks; removing them drove ASR from 0 → >90% with utility intact. This is the technical root that TwinBreak improves on.

### Direct competitors (and follow-ups)
- **A Framework for Evading LLM Safety Alignment (DBDI)**
  arXiv:2511.06852 — AAAI 2026
  https://arxiv.org/html/2511.06852v4
  Bi-directional intervention (Harm Detection + Refusal Execution vectors). **Explicitly benchmarks against TwinBreak** and beats it (95.96% vs 94.62% ASR on AdvBench, 0.750 vs 0.702 StrongREJECT). Leverages TwinPrompt directly.
- **Safety Layers in Aligned LLMs: The Key to LLM Security** (Li et al., ICLR 2025)
  https://openreview.net/forum?id=kUH1yPMAn7
  Shows safety is concentrated in a contiguous band of middle layers; proposes Safely Partial-Parameter Fine-Tuning (SPPFT).
- **Safety Alignment Should Be Made More Than Just a Few Tokens Deep** (Qi, Panda, Lyu et al.) — **ICLR 2025 Outstanding Paper**
  https://openreview.net/forum?id=6Mxhg9PtDE
  The "shallow safety alignment" thesis: alignment is local to first few generated tokens. Killed fine-tuning-based and adversarial-suffix attacks by extending safety beyond the prefix.
- **Towards Understanding Safety Alignment: A Mechanistic Perspective from Safety Neurons** (Chen et al., NeurIPS 2025)
  https://neurips.cc/virtual/2025/poster/119475
  Identifies ~5% safety neurons and shows causal control of refusal behaviour via dynamic activation patching.
- **Safety Knowledge Neurons / SafeTuning** (Zhao et al., Sep 2025)
  Reported by https://www.emergentmind.com/topics/construction-based-method-for-llm-jailbreaks — adjusts identified "safety knowledge neurons" to suppress jailbreak responses.

### Other safety-located interventions (same flavor)
- **Antidote: Post-fine-tuning Safety Alignment for LLMs against Harmful Fine-Tuning Attack** (Huang et al., ICML 2025)
  https://icml.cc/virtual/2025/poster/46150 — one-shot pruning of harmful weights after a fine-tuning attack.
- **Safe Pruning LoRA — SPLoRA** (Ao, Dong, Hu, Ramchurn — TACL 2025, Nov 2025)
  https://direct.mit.edu/tacl/article/doi/10.1162/TACL.a.44/133861/Safe-Pruning-LoRA-Robust-Distance-Guided-Pruning
  https://github.com/AoShuang92/SPLoRA
  Selective LoRA-pruning (E-DIEM metric) to preserve safety while fine-tuning.
- **Pruning for Protection: Increasing Jailbreak Resistance in Aligned LLMs Without Fine-Tuning** (arXiv:2401.10862)
  https://arxiv.org/abs/2401.10862 — defensive view: WANDA pruning at 10–20% sparsity *increases* jailbreak resistance.
- **NeST — Neuron Selective Tuning for LLM Safety**
  Cited extensively in the LLM-safety literature.
- **Weight Orthogonalization (WO)**
  https://www.promptfoo.dev/lm-security-db/tag/whitebox — orthogonalize model weights to the refusal direction (W' = W − r rᵀ W). Bypasses safety without utility loss.
- **Amnesia** — activation-space white-box attack at inference time.
- **SAHA — Safety Attention Head Attack** — targeted perturbations to safety-critical attention heads via Ablation-Impact Ranking (AIR).
- **Refusal in Language Models Is Mediated by a Single Direction** (Arditi et al., 2024) — the activation-difference insight that TwinBreak generalises with the twin-prompt pairing.
- **GateBreaker: Gate-Guided Attacks on Mixture-of-Expert LLMs** (USENIX Security 2026 prepub)
  https://www.usenix.org/system/files/conference/usenixsecurity26/sec26_prepub_wu.pdf — MoE-targeted extension that cites TwinBreak as baseline.
- **Activation Surgery: Jailbreaking White-box LLMs without Touching the Prompt** (arXiv:2603.14278)
  Uses a benign "organ donor" twin prompt and does *layer-wise activation substitution* — explicitly called out as parallel work to TwinBreak.
- **A Causal Perspective for Enhancing Jailbreak Attack and Defense** (NDSS 2026)
  https://www.ndss-symposium.org/wp-content/uploads/2026-f797-paper.pdf — uses TwinBreak as attack baseline.
- **Vanishing Discriminability in LLM Hidden States Fuels Jailbreak** (NDSS 2026)
  https://arxiv.org/html/2503.11185v2 — compares with TwinBreak's data transformation strategy.
- **A Granular Study of Safety Pretraining under Model Abliteration** (arXiv:2510.02768)
  https://arxiv.org/abs/2510.02768 — controlled study on SmolLM2-1.7B checkpoints; abliteration vs safety pretraining.
- **DualEdit: Mitigating Safety Fallback in Editing-based Backdoor Injection on LLMs** (arXiv:2506.13285, June 2025)
  https://github.com/zhaozetong/DualEdit — backdoor twin: edits specific weights to embed triggers; conceptually adjacent.

### Multilingual / sparse-editing defenses (sibling angle)
- **Multilingual Safety Alignment Via Sparse Weight Editing** (arXiv:2602.22554, Feb 2026)
  https://arxiv.org/abs/2602.22554
- **Interpretable Safety Alignment via SAE-Constructed Low-Rank Subspace Adaptation** (arXiv:2512.23260, Dec 2025)
  https://arxiv.org/abs/2512.23260
- **NeWTral: Automatic Safety Alignment Restoration through MoE Weight-Space Translation** (arXiv:2605.04992, 2026)
  https://arxiv.org/html/2605.04992v1

---

## 3. Activation-/Direction-Based Attacks (Same Intent, No Pruning)

- **GCG (Greedy Coordinate Gradient)** — Zou, Wang, Jia, Mittal, Mittal. The canonical gradient-based adversarial-suffix attack. https://github.com/llm-attacks/llm-attacks (often first white-box baseline cited alongside TwinBreak.)
- **I-GCG — Improved techniques for optimization-based jailbreaking on LLMs** (ICLR 2025)
- **Don't Say No: Jailbreaking LLM by Suppressing Refusal (DSN)** — ACL 2025 white-box
- **AutoDAN** — hierarchical genetic algorithm jailbreak
- **LLM-Adaptive Attacks** (tml-epfl) — random-search on logprobs. https://github.com/tml-epfl/llm-adaptive-attacks
- **Amnesia / Weight Orthogonalization / SAHA** — see Section 2; pure-activation variants.

---

## 4. Black-Box / Prompt-Crafting Jailbreaks (Contrast & Adjacent)

- **PAIR — Prompt Automatic Iterative Refinement** (Chao et al., SaTML 2025)
  https://github.com/patrickrchao/JailbreakingLLMs
- **TAP — Tree of Attacks with Pruning** (NeurIPS 2024)
  https://neurips.cc/virtual/2024/poster/95078
- **DRA — Making Them Ask and Answer** (USENIX Security 2024)
  https://github.com/LLM-DRA/DRA
- **Persuasive Adversarial Prompts / AutoJailbreak / LLMStinger** — RL-fine-tuned attacker (arXiv).
- **GASP: Efficient Black-Box Generation of Adversarial Suffixes** (CVPR '25)
  https://github.com/llm-gasp/gasp
- **Logic Jailbreak / LogiBreak** (arXiv:2505.13527, May 2025) — universal black-box via logical expression framing.
- **Crescendo** (USENIX Security 2025)
  https://www.usenix.org/system/files/conference/usenixsecurity25/sec25cycle1-prepub-805-russinovich.pdf

---

## 5. Surveys / Reading Lists (Recommended Intake)

- **Jailbreaking LLMs: A Survey of Attacks, Defenses and Evaluation** (2026)
  https://www.techrxiv.org/doi/10.36227/techrxiv.176773228.86819800
- **A Survey of Jailbreaking Attacks on Large Language and Multimodal Models** (ACM 2025)
  https://dl.acm.org/doi/10.1145/3789982.3790001
- **Jailbreak Attacks and Defenses Against LLMs: A Survey** (arXiv:2407.04295)
  https://arxiv.org/abs/2407.04295
- **A Survey of Modern LLM Jailbreaks** (OpenReview 2024)
  https://openreview.net/pdf/f6f1229d69bcb50271c6f5a4b01e327d2cc039f2.pdf
- **TeleAI-Safety: A Comprehensive LLM Jailbreaking Benchmark** (arXiv:2512.05485, 2025)
  https://arxiv.org/pdf/2512.05485
- **Microsoft: Jailbreaking is (mostly) simpler than you think** (March 2025)
  https://msrc.microsoft.com/blog/
- **AILuminate v0.5 Jailbreak Benchmark** (MLCommons, 2025)
  https://mlcommons.org/wp-content/uploads/2025/12/MLCommons-Security-Jailbreak-0.5.1.pdf

Awesome-list repos (rolled fresh, useful starting points):
- https://github.com/yueliu1999/Awesome-Jailbreak-on-LLMs
- https://github.com/chen37058/Red-Team-Arxiv-Paper-Update (near-daily updated)
- https://github.com/ThuCCSLab/Awesome-LM-SSP
- https://github.com/liuxuannan/Awesome-Multimodal-Jailbreak
- https://huggingface.co/datasets?search=jailbreak

---

## 6. Adjacent Themes Worth Bookmarking

- **Backdoor injection in LLMs** — TwinBreak *frames* safety as a backdoor; real backdoor-injection work is the conceptual mirror:
  - Stealthy and Persistent Unalignment via Backdoor Injection (arXiv:2312.00027)
  - BadEdit, Sleeper Agents (Hubinger et al., 2024), BEEAR defense (OpenReview 2024).
- **Mechanistic interpretability of safety neurons/heads** — section 2 papers above.
- **LLM unlearning for safety** — RMU/WMDP (https://www.wmdp.ai/), TOFU, MUSE.
- **Safety pretraining & abliteration** — community "abliterated" model releases on Hugging Face are direct descendants of TwinBreak-style work.
