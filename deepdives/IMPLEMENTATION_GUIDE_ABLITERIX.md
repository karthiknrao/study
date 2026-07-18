# Abliterix Implementation Guide

> **Purpose**: This document explains how the Abliterix codebase works and provides
> implementation-level detail sufficient for another LLM to re-implement the repository.
> It covers architecture, data flow, algorithms, module responsibilities, and the
> exact mathematical operations performed by each steering mode.

---

## 1. What Abliterix Does

Abliterix is an automated **abliteration** framework: it removes refusal behavior
from instruction-tuned language models while preserving as much of the base model's
capability as possible. It does this by:

1. Extracting per-layer **refusal directions** from paired harmless/harmful prompts.
2. Optionally cleaning those directions (SRA, projection, harmfulness decomposition).
3. Applying weight edits (LoRA, direct, angular, spherical, vector-field) to
   attention and MLP/MoE modules.
4. Searching the space of edit strengths with **Optuna TPE** to minimize a
   two-objective Pareto front: `refusal_rate` vs `KL_divergence`.

The result is a decensored model that can be exported, uploaded, or chatted with.

---

## 2. Repository Layout

```
abliterix/
├── pyproject.toml              # uv-based packaging, extras: research, ui, bench, vllm
├── src/abliterix/
│   ├── __init__.py
│   ├── cli.py                  # Entry point: banner, device detection, orchestration
│   ├── settings.py             # Pydantic-settings config (TOML + CLI + env)
│   ├── types.py                # Enums and small dataclasses
│   ├── util.py                 # Memory, seeding, UI helpers
│   ├── data.py                 # Dataset loading and trial formatting
│   ├── analysis.py             # Residual geometry reports / PaCMAP plots
│   ├── vectors.py              # Steering-vector computation (mean, PCA, OT, SRA, SOM, SAE)
│   ├── optimizer.py            # Optuna TPE search loop
│   ├── core/
│   │   ├── engine.py           # SteeringEngine: model loading, generation, hidden states
│   │   ├── steering.py         # Apply steering vectors to weights (LoRA/direct/etc.)
│   │   ├── vllm_backend.py     # vLLM tensor-parallel backend + adapter serialization
│   │   ├── vllm_moe_editor.py  # In-place MoE router suppression via collective_rpc
│   │   ├── vllm_compat.py      # Version/env compatibility shims
│   │   ├── fp8_utils.py        # FP8 dequantization / materialization
│   │   └── sglang_backend.py   # SGLang backend
│   ├── eval/
│   │   ├── scorer.py           # TrialScorer: KL, coherence, multi-objective score
│   │   └── detector.py         # RefusalDetector: keyword + LLM judge
│   ├── sra.py                  # Surgical Refusal Ablation
│   ├── som.py                  # Self-organizing-map directions
│   ├── sae.py                  # Sparse autoencoder feature basis
│   ├── rdo.py                  # Gradient-based refusal direction optimization
│   ├── svf.py                  # Steering vector fields
│   ├── harmfulness.py          # Joint harmfulness ⊥ refusal decomposition
│   ├── cliff_head.py           # Reasoning-model head ablation
│   ├── weight_transforms.py    # Direct-mode transforms (ORBA, biprojected, Householder)
│   ├── mote.py                 # Mixture-of-Tunable-Experts inference-time hooks
│   ├── safex.py                # SAFEx stability-based MoE expert scoring
│   ├── iterative.py            # Iterative abliteration for hardened models
│   ├── pareto.py               # Pareto-front selection helpers
│   ├── interactive.py          # Post-search interactive menu
│   └── scripts/
│       └── dequant_fp8.py      # CLI for offline FP8→BF16 dequant
├── configs/                    # 150+ per-model TOML recipes
├── docs/                       # Architecture, methods, evaluation docs
├── tests/                      # pytest suite
└── benchmarks/                 # HonestAbliterationBench spec
```

---

## 3. Configuration System

Abliterix uses **Pydantic-Settings** with a custom source precedence:

1. `init_settings` (programmatic)
2. CLI flags (`--model.model-id Qwen/Qwen3-4B`)
3. Environment variables (`AX_*`)
4. dotenv / secrets
5. TOML file (default `abliterix.toml`, overridable via `--config` or `AX_CONFIG`)

Key config sections:

| Section | File | Purpose |
|---------|------|---------|
| `[model]` | `settings.py:ModelConfig` | Model id, dtype, quantization, backend (hf/vllm/sglang), TP, FP8 handling |
| `[inference]` | `InferenceConfig` | Batch size, max tokens, min tokens, offload |
| `[steering]` | `SteeringConfig` | Vector method, steering mode, projection, search ranges |
| `[optimization]` | `OptimizationConfig` | Optuna trials, warmup, checkpoint dir, seeds |
| `[kl]` | `KLConfig` | KL scale, token count, prune threshold, objective mode |
| `[detection]` | `DetectionConfig` | Keyword markers, LLM judge endpoint, batch/concurrency |
| `[experts]` | `ExpertConfig` | MoE safety-expert search bounds |
| `[iterative]` | `IterativeConfig` | Multi-pass abliteration |
| Data sources | top-level | `benign_prompts`, `target_prompts`, `benign_eval_prompts`, `target_eval_prompts` |

Implementation note: CLI shorthands in `cli.py` map `--model X` to
`--model.model-id X` and infer a trailing model id argument automatically.

---

## 4. Core Algorithm Pipeline

The high-level flow in `cli.run()`:

```
1. Parse config
2. Detect devices, set seeds
3. Load benign/target prompt datasets
4. Optionally extract hidden states via vLLM native or speculators fast path
5. Load HF model (unless fast path skipped it)
6. Auto-tune batch size / detect response prefix (HF backend)
7. Build RefusalDetector and TrialScorer
8. Compute steering vectors from residual streams
9. Optionally: SRA, harmfulness pair, iterative peel, SVF training, cliff-head ablation
10. Profile MoE safety experts if applicable
11. If backend is vLLM/SGLang:
    a. Build ProjectionCache from safetensors or HF model
    b. Unload HF model
    c. Load vLLM/SGLang TP backend
    d. Attach MoE/editor hooks
    e. Capture baseline via TP backend
12. Run Optuna search (optimizer.run_search)
13. Show interactive results or exit in non-interactive mode
```

---

## 5. SteeringEngine (`core/engine.py`)

`SteeringEngine` is the model wrapper. Responsibilities:

### 5.1 Model Loading

- Chooses `AutoModelForCausalLM` vs `AutoModelForImageTextToText` based on
  presence of `vision_config`.
- Iterates `dtype_fallback_order` until a load succeeds.
- Applies BitsAndBytes 4-bit/8-bit quantization or FP8 handling.
- Auto-detects native FP8 / MXFP4 from `quantization_config`.
- Patches MTP layer-type mismatches (`_patch_mtp_layer_types`).
- Patches MoE configs missing `intermediate_size` for FP8 quantizer.
- Handles custom encoder modules (e.g. DeepSeek-V4).

### 5.2 FP8 Handling (`core/fp8_utils.py`)

Three modes controlled by `model.fp8_handling`:

- **materialize**: Replace every FP8 `nn.Linear` / fused expert with BF16
  `Parameter`s. Required for direct-mode weight edits.
- **forward_dequant**: Monkey-patch `Linear.forward` to dequantize on-the-fly.
  Memory-efficient, LoRA-only.
- **offline**: Assume weights are already BF16 on disk.

### 5.3 Module Discovery

- `transformer_layers`: returns `m.model.layers` or `m.model.language_model.layers`,
  truncated to `num_hidden_layers`.
- `steerable_modules(layer_idx)`: returns dict `component_name -> [nn.Module]`.
  Discovers attention Q/K/V/O, MLA q_b_proj/kv_b_proj, MLP down_proj, per-expert
  down_proj, shared experts, SSM out_proj, conv out_proj, etc.
- `list_steerable_components()`: sorted union across layers; synthesizes
  `mlp.down_proj` for fused MoE weights so EGA gets a profile.
- `has_expert_routing()` / `_locate_router()` / `_locate_fused_weights()`: MoE support.

### 5.4 Adapter Initialization (`_init_adapters`)

- Maps every discovered steerable module to its full path.
- Wraps the model with PEFT `get_peft_model` using `LoraConfig`.
- Rank is 1 except for `weight_normalization="full"`, which uses rank 3.
- Caches `lora_B` weights for fast baseline restoration.

### 5.5 Generation and Hidden-State Extraction

Methods to implement:

- `_tokenize(messages) -> BatchEncoding`: apply chat template, left-pad.
- `_generate(messages, max_new_tokens) -> list[str]`: greedy HF generation.
- `generate_text_batched(messages, ...) -> list[str]`: batched generation.
- `generate_and_score_batched(messages, kl_token_count) -> (responses, logprobs)`:
  generates and returns top-K logprob tensors for the first `kl_token_count` tokens.
- `compute_logprobs_batched(messages) -> logprobs`: score a fixed prompt.
- `extract_hidden_states_batched(messages) -> Tensor (n, layers+1, hidden_dim)`:
  returns residual stream hidden states at every layer + embedding.

Important detail: index 0 of the hidden-state tensor is the **embedding output**;
  layer `i` corresponds to transformer layer `i-1`.

---

## 6. Steering-Vector Computation (`vectors.py`)

Input: `benign_states` and `target_states`, shape `(n, layers+1, hidden_dim)`.

Output: `vectors`, shape `(layers+1, hidden_dim)` for single-direction modes,
        or `(n_dirs, layers+1, hidden_dim)` for multi-direction / SOM / SAE /
        harmfulness-pair modes.

### 6.1 Single-Direction Methods

- **mean** (default):
  ```
  vectors = F.normalize(target.mean(0) - benign.mean(0), p=2, dim=1)
  ```
- **median_of_means**: split into 5 chunks, take median of per-chunk directions.
- **pca**: PCA-1 of centered target residuals.
- **optimal_transport**: PCA-Gaussian closed-form transport map per layer.

### 6.2 Projection

Two flags (mutually ordered):

- `orthogonal_projection=True`: project the refusal vector onto the benign mean
  and subtract it, then renormalize.
- `projected_abliteration=True` (grimjim): only remove the component of the
  refusal direction **orthogonal** to the benign mean, preserving helpfulness.

```
benign_dir = normalize(mean(benign_states), dim=1)
proj_scalar = sum(vectors * benign_dir, dim=1, keepdim=True)
vectors = normalize(vectors - proj_scalar * benign_dir, dim=1)
```

### 6.3 Multi-Direction Mode

When `n_directions > 1`, per-layer SVD on the difference matrix yields top-K
orthogonal components. Each is projected independently if projection flags are set.

### 6.4 SRA (`sra.py`)

1. Compute base refusal vector via `compute_steering_vectors(..., method=base_method)`.
2. Build concept atoms: top `n_atoms` PCA components of **benign** residuals per layer.
3. Residualize the refusal vector against the concept subspace with ridge regularization:
   ```
   coeffs = solve(C C^T + alpha I, C v)
   v_clean = v - C^T coeffs
   ```
4. Normalize.

### 6.5 SOM (`som.py`)

1. For each layer, train a 2-D Kohonen SOM on harmful residuals.
2. Compute per-node centroids.
3. Direction per node = `normalize(centroid - benign_mean)`.
4. Optional projection.

### 6.6 SAE (`sae.py`)

1. Load checkpoint; detect encoder/decoder keys.
2. Encode harmful and harmless activations at `sae_layer`.
3. Score features by `|mean(target_feats) - mean(benign_feats)|`.
4. Use top-K decoder columns as refusal directions.
5. At non-SAE layers, fall back to mean-diff.

### 6.7 RDO (`rdo.py`)

Learn a single unit vector `r` via AdamW on the HF model:

```
L = lambda_abl * CE(f_ablate(r)(harm),  "Sure, here")
  + lambda_add * CE(f_add(alpha*r)(safe), "I'm sorry, but I can't help with that.")
  + lambda_ret * KL(f_ablate(r)(safe) || f(safe))
```

`f_ablate` projects `r` out at every layer via forward hooks;
`f_add` adds `alpha*r` at a single layer. Broadcast the learned direction to
all layers.

### 6.8 Harmfulness Pair (`harmfulness.py`)

Returns stacked `(refusal, harmfulness)` vectors:

- refusal = mean-diff
- harmfulness = PCA-1 of centered harmful activations, orthogonalized against refusal
- harmfulness is 0.5× outside the configured mid-layer band `[0.3, 0.7]`

---

## 7. Applying Steering (`core/steering.py`)

`apply_steering(engine, steering_vectors, vector_index, profiles, config, ...)`
dispatches by `steering_mode`.

### 7.1 Common Search Parameters

For each steerable component the optimizer samples a `SteeringProfile`:

```
max_weight          # peak ablation strength
max_weight_position # layer index of peak
min_weight          # floor strength (fraction of max_weight)
min_weight_distance # layers over which kernel tapers from peak to floor
```

Per-layer strength is interpolated with a decay kernel:

- **linear**:   `strength = max + t*(min - max)`
- **gaussian**: `strength = min + (max - min) * exp(-2 t^2)`
- **cosine**:   `strength = min + (max - min) * 0.5*(1 + cos(pi*t))`

where `t = distance / min_weight_distance`.

### 7.2 LoRA Mode

For each module:

1. Dequantize 4-bit / 8-bit / FP8 weight to float32 (cached).
2. Optionally normalize rows (`weight_normalization`).
3. Compute rank-1 adapter:
   ```
   lora_A = (v @ W).view(1, -1)
   lora_B = (-strength * v).view(-1, 1)
   ```
4. If `weight_normalization="full"`, perform low-rank SVD of the full delta
   and keep top `full_norm_lora_rank` singular values.
5. Write `lora_A` and `lora_B` weights into the PEFT adapter.

### 7.3 Direct Mode

Modify base weights in-place:

```
W_new = W - strength * outer(v, W @ v)   # input-side, v matches in_features
W_new = W - strength * outer(v, v @ W)   # output-side, v matches out_features
```

Then preserve row norms if requested.

Advanced transforms (`weight_transforms.py`):

- **ORBA**: double Gram-Schmidt orthogonalize `v` against benign mean, then
  rank-1 ablate, preserving row norms.
- **biprojected**: decompose into per-row or per-column magnitude × direction,
  ablate the unit directions, renormalize, recombine.
- **householder**: `W_new = W - 2*strength * outer(v, v @ W)`.

### 7.4 Angular / Adaptive Angular Modes

Install forward hooks on every layer that rotate the residual stream within the
2-D plane spanned by the activation and the steering direction:

```
proj_scalar = h @ d
residual    = h - proj_scalar * d
b2          = residual / ||residual||
h_new = (cosθ*proj_scalar + sinθ*||residual||) * d
      + (-sinθ*proj_scalar + cosθ*||residual||) * b2
```

Adaptive mode only rotates activations positively aligned with `d`.

### 7.5 Spherical Mode

Rotate along the geodesic on the activation hypersphere (RMSNorm-aware).

### 7.6 Vector-Field Mode

Train small MLP concept scorers on benign/target states (`svf.py`). At inference,
the gradient `∇_h score(h)` provides a per-token steering direction.

### 7.7 MoE Steering

Two complementary mechanisms:

1. **Router suppression** (`_apply_moe_steering`, `core/vllm_moe_editor.py`):
   scale down router weight rows for top-N safety experts:
   ```
   router.weight[eid] *= scale
   scale = max(0, 1 + router_bias / 10)
   ```
2. **Expert-Granular Abliteration (EGA)**:
   project the refusal direction out of **every** expert's fused `down_proj`
   tensor, preserving expert-row norms.

Safety experts are identified by:

```
score(e) = P(e | target) - P(e | benign)
```
or the SAFEx variant (`safex.py`):
```
score(e) = (mu_target - mu_benign) - lambda * sigma_target
```

### 7.8 Cliff-Head Ablation (`cliff_head.py`)

For reasoning models:

1. Read `num_attention_heads` and `head_dim` from config.
2. For each `(layer, head)`, project the per-layer refusal vector through the
   head's `o_proj` column block and compute the L2 norm of `head_cols^T @ v`.
3. Ablate the top `top_k_frac` heads by scaling their `o_proj` columns toward zero.

---

## 8. Evaluation (`eval/scorer.py`, `eval/detector.py`)

### 8.1 TrialScorer

Captures baseline on the un-steered model:

- `baseline_logprobs`: top-K logprobs on benign eval prompts.
- `baseline_mean_length`, `baseline_stdev_length`: word-count statistics.
- `baseline_refusal_count`: refusals on target eval prompts.

Per trial:

- `measure_kl_and_coherence`: generate benign responses, compute KL vs baseline
  and response-length z-score.
- `evaluate_compliance`: count refusals on target prompts.
- `_compute_objectives`: return `(divergence_objective, compliance_objective)`.

```
compliance_objective = detected_refusals / baseline_refusal_count
divergence_objective = kl_divergence / kl.scale
# unless do_nothing_guard and kl < target, then tied to compliance
if length_deviation > 2:
    divergence_objective *= 1 + 0.1*(length_deviation - 2)
```

### 8.2 RefusalDetector

Two modes:

- **keyword**: case-insensitive substring matching on a bilingual marker list,
  plus positional handling of "sorry" and extra patterns.
- **LLM judge**: OpenAI-compatible `/chat/completions` API with a structured
  JSON response format classifying each response as `R` (refusal/degenerate) or
  `C` (compliance).

Degenerate outputs (long filler runs, repeated sentences/n-grams, low character
diversity) are always counted as refusals.

---

## 9. Optimization Loop (`optimizer.py`)

Optuna study:

- Directions: minimize both divergence and compliance objectives.
- Sampler: `TPESampler` with `n_startup_trials` random warmup,
  `n_ei_candidates=128`, `multivariate=True`.
- Storage: `JournalStorage` with `JournalFileBackend` for resume support.

Per trial sampling:

1. Optional categorical: `direct_transform`, `decay_kernel`, `steering_variant`.
2. Categorical `vector_scope`: "global" or "per layer".
3. Float `vector_index` in `[0.3..0.95]*last_layer`; set to `None` if per-layer.
4. For each steerable component sample:
   - `max_weight` (with optional auto-disable negative lower bound)
   - `max_weight_position`
   - `min_weight` (fraction of max_weight)
   - `min_weight_distance`
5. If MoE, sample `moe.n_suppress`, `moe.router_bias`, `moe.expert_ablation_weight`.

Trial execution:

- For vLLM + in-place editing: dispatch edits to TP workers via `collective_rpc`.
- For vLLM + LoRA: build adapter weights from `ProjectionCache`, save to `/dev/shm`.
- For HF: `restore_baseline()` then `apply_steering(...)`.
- Evaluate; early-prune if KL > `kl.prune_threshold`.
- In `finally`: restore router/editor weights and sampled config flags.

---

## 10. vLLM Backend (`core/vllm_backend.py`, `core/vllm_moe_editor.py`)

### 10.1 VLLMGenerator

- Loads the model with tensor parallelism.
- Auto-selects attention backend for MLA / sink-attention models.
- Supports FP8 quantization, KV cache dtype, chunked prefill, prefix caching.
- Implements the same generation API as `SteeringEngine`.

### 10.2 ProjectionCache

Pre-computes `v @ W` for every steerable module so LoRA weights can be built
without the HF model loaded. Supports building from:

- loaded HF model (`ProjectionCache.build`)
- safetensors files on disk (`ProjectionCache.build_from_safetensors`)

Adapter serialization writes safetensors to a temp path under `/dev/shm`.

### 10.3 MoE Router Suppression in vLLM

Because vLLM shards the model across TP workers, direct weight edits use
`llm_engine.collective_rpc`:

- `_worker_install_persistent_suppression`: register forward hooks once per
  worker lifetime (re-registration is silently skipped by Dynamo on compiled
  models, so hooks must be installed before the first forward).
- `_worker_set_suppression_plan`: update a per-worker mutable dict that the hook
  reads; subtracts a penalty from router logits for flagged experts.
- `_worker_clear_suppression_plan`: restore baseline behavior.

Alternative profiling: with `enable_return_routed_experts=True`, read per-token
expert IDs directly from `RequestOutput.outputs[0].routed_experts` and aggregate
driver-side.

---

## 11. SGLang Backend (`core/sglang_backend.py`)

A drop-in TP backend similar to vLLM. It handles LoRA adapter loading and
parallel generation. MoE in-place editing is not yet implemented for SGLang.

---

## 12. Interactive Flow and Reproducibility

After optimization, `interactive.py` presents a menu:

- Show Pareto front / best trials
- Save model (merges LoRA or exports direct edits)
- Upload to Hugging Face
- Chat interactively
- Run standard benchmarks
- Export `reproduce.json` manifest with config, environment, and best trial

`reproducibility.py` records git commit, dependency versions, CUDA driver, and
config so runs can be reproduced with `abliterix --reproduce path.json`.

---

## 13. How to Re-Implement This Repo

### 13.1 Minimal Viable Skeleton

Start with these files in order:

1. `pyproject.toml` with dependencies: `torch`, `transformers`, `accelerate`,
   `peft`, `bitsandbytes`, `optuna`, `datasets`, `huggingface-hub`, `pydantic-settings`,
   `rich`, `questionary`.
2. `src/abliterix/types.py` — enums and dataclasses.
3. `src/abliterix/settings.py` — full config model.
4. `src/abliterix/util.py` — seeding, memory, UI helpers.
5. `src/abliterix/data.py` — dataset loading.
6. `src/abliterix/core/engine.py` — model loading, hidden-state extraction,
   generation, LoRA adapter init.
7. `src/abliterix/vectors.py` — compute steering vectors.
8. `src/abliterix/core/steering.py` — apply steering vectors.
9. `src/abliterix/eval/detector.py` — refusal detection.
10. `src/abliterix/eval/scorer.py` — KL / coherence / objectives.
11. `src/abliterix/optimizer.py` — Optuna loop.
12. `src/abliterix/cli.py` — orchestration.

### 13.2 Critical Invariants

- Hidden-state tensor index 0 is the embedding; layer `i` is at index `i+1`.
- `steerable_modules` must discover all target modules for the architectures you
  intend to support. Test on Llama, Qwen, Mistral, Gemma, Phi, and any MoE models.
- Left-padding is required for decoder-only generation.
- Always restore baseline weights/adapters/router edits between trials.
- FP8 direct-mode editing requires materializing weights to BF16 first.
- vLLM in-place editing requires `enforce_eager=True` and persistent hooks
  installed before the first forward.

### 13.3 Algorithm Checklist

- [ ] Compute mean-diff or chosen vector method on `(benign, target)` residuals.
- [ ] Optionally project against benign mean (projected abliteration).
- [ ] Build per-component steering profiles with decay-kernel interpolation.
- [ ] For LoRA: compute `lora_A = v @ W`, `lora_B = -strength * v`.
- [ ] For direct: `W_new = W - strength * outer(v, v @ W or W @ v)`, preserve norms.
- [ ] For MoE: identify safety experts, optionally suppress router + EGA.
- [ ] Evaluate: KL vs baseline on benign prompts, refusals on target prompts.
- [ ] Optimize with Optuna minimizing `(KL, refusal_rate)`.

### 13.4 Testing Strategy

- Unit tests for each vector method with synthetic residual tensors.
- Unit tests for steering application on small models (e.g. `Qwen2-0.5B`).
- Unit tests for detector keyword and judge classification.
- Integration test on a tiny model end-to-end with a few trials.
- Mock-based tests for vLLM backend kwargs assembly and MoE editor plans.

---

## 14. Key Equations Summary

| Operation | Equation |
|-----------|----------|
| Mean-diff direction | `r_l = normalize(mean(h_target,l) - mean(h_benign,l))` |
| Orthogonal projection | `r⊥ = normalize(r - (r · μ̂) μ̂)` |
| Projected abliteration | `r_proj = r - (r · μ̂) μ̂` (same as ortho here) |
| LoRA A/B | `A = v^T W`, `B = -λ v` |
| Direct ablation (output-side) | `W' = W - λ v (v^T W)` |
| Direct ablation (input-side) | `W' = W - λ (W v) v^T` |
| Gaussian kernel | `λ(d) = min + (max-min) exp(-2 (d/D)^2)` |
| Compliance objective | `detected / baseline_detected` |
| Divergence objective | `KL / scale` |
| Router suppression scale | `scale = max(0, 1 + bias/10)` |
| SAFEx score | `(μ_t - μ_b) - λ σ_t` |
| SRA residualization | `v_clean = v - C^T (C C^T + αI)^{-1} C v` |

---

## 15. References

See `docs/architecture.md`, `docs/methods.md`, `docs/evaluation.md`, and
`docs/references.md` for paper references and BibTeX.

The codebase is a derivative of [Heretic](https://github.com/p-e-w/heretic)
(AGPL-3.0-or-later) with extensive extensions for optimization, MoE, backends,
and modern steering methods.
