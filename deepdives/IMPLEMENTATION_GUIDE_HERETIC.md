# Heretic: Detailed Implementation Guide

This document provides a comprehensive, module-by-module guide to implementing the Heretic codebase. It is intended for an LLM or developer who needs to understand both the algorithms and the concrete implementation details.

---

## 1. Overview and Goals

**Heretic** is a fully automatic tool for removing censorship ("safety alignment") from transformer-based language models without expensive post-training. It combines:

1. **Directional ablation** (also known as "abliteration"): identifying a "refusal direction" in the model's residual stream and orthogonalizing weight matrices against it.
2. **Parameter optimization** via Optuna (Tree-structured Parzen Estimator, multivariate TPE): searching for the best abliteration hyperparameters that minimize refusals while preserving model capabilities.

The result is a decensored model that retains as much of the original model's intelligence as possible.

### High-level user flow

1. User provides a Hugging Face model ID (or local path) and optionally a `config.toml`.
2. Heretic loads the model, tokenizer, and (for multimodal models) processor.
3. Heretic loads "good" prompts (harmless, tend not to be refused) and "bad" prompts (harmful, tend to be refused).
4. Heretic computes **residual directions** per layer: `bad_mean - good_mean`, optionally orthogonalized against the good direction.
5. Heretic wraps target weight matrices in LoRA adapters.
6. Heretic runs an Optuna study where each trial:
   - Samples abliteration parameters (direction scope/index, weight kernel per component).
   - Applies the abliteration via LoRA weights.
   - Evaluates the model with scorer plugins.
   - Reports objective values.
7. After optimization, the user selects a Pareto-optimal trial and can save/upload/chat/benchmark the model.

---

## 2. Core Algorithmic Details

### 2.1 Residual Vectors and Directions

For a set of prompts, Heretic generates exactly one new token and captures the hidden states (residuals) at that token position for every transformer layer.

- Shape of raw hidden states per batch: `(batch_size, sequence_length, hidden_dim)`.
- Heretic extracts the **last position** (the newly generated token) for each layer and stacks across layers, yielding a tensor of shape `(batch_size, n_layers + 1, hidden_dim)`.
- The `+1` accounts for the embedding layer (layer 0).

For the good-prompt dataset and bad-prompt dataset, Heretic computes the **mean residual vector** at each layer:

```
good_means[layer]  = mean over good prompts of residuals[prompt, layer, :]
bad_means[layer]   = mean over bad prompts of residuals[prompt, layer, :]
```

The raw **residual direction** at each layer is:

```
residual_directions[layer] = normalize(bad_means[layer] - good_means[layer])
```

#### Orthogonalization (projected abliteration)

If `orthogonalize_direction = true` (default), Heretic adjusts the residual direction so that only the component orthogonal to the good direction is subtracted. This implements the "projected abliteration" idea:

```
good_direction = normalize(good_means[layer])
projection     = sum(residual_direction * good_direction, dim=1)
residual_direction = residual_direction - projection * good_direction
residual_direction = normalize(residual_direction)
```

This prevents the abliteration from removing useful signal aligned with the good-direction subspace.

#### Winsorization

If `winsorization_quantile < 1.0`, per-prompt residual vectors are symmetrically clamped to a quantile threshold to tame "massive activations":

```
threshold = quantile(abs(residuals), winsorization_quantile, dim=component)
residuals = clamp(residuals, -threshold, threshold)
```

### 2.2 Directional Ablation on Weight Matrices

Heretic targets two types of transformer components:

- `attn.o_proj`: the attention output projection.
- `mlp.down_proj`: the MLP down-projection (including per-expert down-projections in MoE layers).

For each target module with weight matrix `W` (shape `(out_features, in_features)`) and residual direction `v` (shape `(out_features,)`), the desired modification is:

```
W' = W - lambda * v * (v^T W)
```

This orthogonalizes `W` with respect to `v`: the row-space component along `v` is removed, inhibiting the expression of that direction in the layer's output.

`lambda` is the per-layer, per-component ablation weight computed from the weight kernel.

### 2.3 LoRA-Based Implementation

Heretic does **not** modify base weights directly. Instead, it applies the modification through **LoRA adapters** (via PEFT) so that trials can be reset cheaply.

The standard LoRA update is:

```
W' = W + lora_B @ lora_A
```

For directional ablation:

```
lora_A = v^T W                # shape (in_features,)
lora_B = -lambda * v          # shape (out_features,)
```

In matrix form:

```
lora_A = (v @ W).view(1, -1)        # shape (1, in_features)
lora_B = (-lambda * v).view(-1, 1)  # shape (out_features, 1)
```

LoRA rank `r=1` is sufficient for plain directional ablation.

#### Row normalization variants

- **`none`**: apply the LoRA directly as above.
- **`pre`**: normalize `W` row-wise before computing `lora_A`, then scale `lora_B` by the original row norms so the adapter acts on the original `W`.
- **`full`**: like `pre`, but after computing `W + lora_B @ lora_A`, renormalize rows and restore original row magnitudes, then factor the resulting delta back into LoRA via truncated SVD (requires rank > 1, default `full_normalization_lora_rank = 3`). This approximates norm-preserving biprojected abliteration.

For `full`, the steps are:

```
W_org = W
W_row_norms = norm(W, dim=1, keepdim=True)
W = normalize(W, dim=1)

lora_A = v @ W
lora_B = -lambda * v

W_new = normalize(W + lora_B @ lora_A, dim=1) * W_row_norms
delta = W_new - W_org

U, S, Vh = svd_lowrank(delta, q=2*r+4, niter=6)
lora_B = U[:, :r] @ diag(sqrt(S[:r]))
lora_A = diag(sqrt(S[:r])) @ Vh[:r, :]
```

The SVD seeds are reset before `svd_lowrank` to keep trial restoration deterministic.

### 2.4 Weight Kernel Over Layers

For each component, the trial parameters define a weight kernel over layers:

- `max_weight`: peak ablation strength.
- `max_weight_position`: layer index where the peak occurs.
- `min_weight`: ablation strength far from the peak.
- `min_weight_distance`: distance from the peak over which the weight decays linearly to `min_weight`.

For a layer at index `layer_index`:

```
distance = abs(layer_index - max_weight_position)
if distance > min_weight_distance:
    weight = 0
else:
    weight = max_weight + (distance / min_weight_distance) * (min_weight - max_weight)
```

If `weight == 0`, skip that layer/component entirely.

### 2.5 Direction Index and Scope

Two modes:

- **`global`**: use a single residual direction for all layers. The direction index is a float in `[0.4 * last_layer, 0.9 * last_layer]`. Because it is a float, Heretic linearly interpolates between the two nearest layer-direction vectors:

  ```
  index_integer = floor(direction_index + 1)   # +1 because index 0 is embeddings
  weight = fraction(direction_index + 1)
  v = normalize(lerp(residual_directions[index_integer],
                     residual_directions[index_integer + 1], weight))
  ```

- **`per layer`**: each layer uses its own residual direction (`residual_directions[layer_index + 1]`).

### 2.6 Optimization Objectives

Default scorers:

1. **KeywordRate**: fraction of responses to harmful prompts that contain refusal keywords ("sorry", "i cannot", etc.) or are empty. Minimized.
2. **KLDivergence**: KL divergence between the ablated model's first-token log-probability distribution and the baseline model's distribution on harmless prompts. Minimized.

These form a multi-objective Optuna study. The Pareto front is presented to the user.

---

## 3. Module-by-Module Implementation

### 3.1 `config.py` — Configuration Schema and Parsing

**Responsibilities:**

- Define Pydantic models for all configuration.
- Provide the `Settings` class (subclass of `pydantic_settings.BaseSettings`) that merges configuration from multiple sources.

**Key classes:**

- `QuantizationMethod`: enum `NONE`, `BNB_4BIT`.
- `RowNormalization`: enum `NONE`, `PRE`, `FULL`.
- `ExportStrategy`: enum `MERGE`, `ADAPTER`.
- `DatasetSpecification`: dataset path/ID, commit, split, column, prefix/suffix, system prompt, plot label/color.
- `ScorerConfig`: plugin path/import, optimization direction, instance name.
- `BenchmarkSpecification`: lm-eval task name/description.
- `Settings`: the main configuration object.

**Important details:**

- `Settings.settings_customise_sources` defines precedence (highest to lowest):
  1. `init_settings` (used during resume).
  2. CLI arguments (`CliSettingsSource`).
  3. Environment variables with prefix `HERETIC_`.
  4. Dotenv file.
  5. Secret file.
  6. `config.toml` (`TomlConfigSettingsSource`).
- `model_config = SettingsConfigDict(extra="allow")` allows plugin-specific tables like `[scorer.KeywordRate]`.
- Fields marked `exclude=True` are omitted when serializing settings for storage/reproducibility.

**Implementation order:** implement this first; everything else depends on it.

### 3.2 `model.py` — Model Loading, LoRA, Abliteration, Inference

**Responsibilities:**

- Load the base model and tokenizer.
- Wrap target modules in LoRA adapters.
- Compute residual vectors and logits.
- Apply abliteration parameters to LoRA weights.
- Generate responses and chat streams.
- Merge/export the final model.

**Key data structure:**

```python
@dataclass
class AbliterationParameters:
    max_weight: float
    max_weight_position: float
    min_weight: float
    min_weight_distance: float
```

**Model loading (`__init__`):**

1. Load tokenizer; set `pad_token = eos_token` if missing; force `padding_side = "left"`.
2. Load processor if the model class is `AutoModelForImageTextToText`.
3. Iterate over `settings.dtypes` (e.g. `auto`, `float16`, `bfloat16`, `float32`) and try `from_pretrained` with each until one works.
4. Build `BitsAndBytesConfig` if `quantization == BNB_4BIT`.
5. After successful load, run a tiny generation sanity check.
6. Call `_apply_lora()`.

**Model class selection (`get_model_class`):**

- Inspect `PretrainedConfig.get_config_dict(model)` for a `vision_config` key.
- If present, use `AutoModelForImageTextToText`; otherwise `AutoModelForCausalLM`.

**LoRA application (`_apply_lora`):**

1. Enumerate all layers and call `get_layer_modules(layer_index)` to collect target modules.
2. Build a set of full module names using `named_modules()` and `id(module)` mapping.
3. Create a `LoraConfig`:
   - `r = 1` for `none`/`pre` normalization; `r = full_normalization_lora_rank` for `full`.
   - `target_modules = sorted(target_modules_set)`.
   - `lora_alpha = r`, `lora_dropout = 0`, `bias = "none"`, `task_type = "CAUSAL_LM"`.
4. Call `get_peft_model(base_model, peft_config)` to obtain a `PeftModel`.

**Layer and module discovery (`get_layers`, `get_layer_modules`, `get_abliterable_components`):**

- `get_layers` unwraps the PeftModel and returns:
  - `model.model.language_model.layers` for most multimodal models.
  - `model.model.layers` for text-only models.
- `get_layer_modules` tries many architecture variants using `suppress(Exception)` guards:
  - `layer.self_attn.o_proj`
  - `layer.linear_attn.out_proj` (Qwen3.5 MoE hybrid)
  - `layer.mlp.down_proj`
  - `layer.mlp.experts[*].down_proj` (Qwen3 MoE)
  - `layer.block_sparse_moe.experts[*].w2` (Phi-3.5 MoE)
  - LFM variants (`conv.out_proj`, `feed_forward.w2`, `self_attn.out_proj`, `feed_forward.experts[*].w2`)
  - Granite MoE Hybrid variants (`shared_mlp.output_linear`, `moe.experts[*].output_linear`)
- `get_abliterable_components` scans all layers because hybrid models may have different components on different layers.

**Abliteration (`abliterate`):**

1. Compute the global interpolated residual direction if `direction_index` is not `None`.
2. For each layer and component:
   - Compute weight from kernel; skip if 0.
   - Get layer-specific residual direction (`per layer` mode) or global direction.
   - For each module:
     - Dequantize 4-bit weights if needed via `bnb.functional.dequantize_4bit`.
     - Flatten weight to 2D.
     - Apply row normalization if configured.
     - Compute `lora_A = v @ W` and `lora_B = -weight * v`.
     - Apply `pre`/`full` normalization transformations.
     - Assign to `module.lora_A["default"].weight` and `module.lora_B["default"].weight`.

**Generation (`generate`):**

1. Convert `Prompt` objects to chat dicts with system + user messages.
2. Apply chat template with `tokenize=False`, `add_generation_prompt=True`.
3. Append `response_prefix` if set.
4. Tokenize with left padding; move to model device.
5. Call `model.generate(..., do_sample=False)` for deterministic outputs.

**Residuals (`get_residuals`, `get_residuals_batched`, `get_residuals_mean`):**

- Generate 1 token with `output_hidden_states=True`, `return_dict_in_generate=True`, `use_cache=False`.
- Extract hidden states for the generated token across layers, stack to `(batch, layer, hidden_dim)`.
- Upcast to float32; optionally winsorize; optionally move to CPU.
- `get_residuals_mean` accumulates sums in float64 on CPU to reduce numerical error and VRAM.

**Logits (`get_logits`, `get_logits_batched`):**

- Generate 1 token with `output_logits=True`, `return_dict_in_generate=True`, `use_cache=False`.
- Return raw logits tensor of shape `(batch, vocab_size)`.
- Use raw logits (not processed scores) to avoid `-inf` causing NaN in KL divergence.

**Reset (`reset_model`):**

- Fast path: if the same model is loaded and not marked for reload, zero all `lora_B` weights (identity adapter).
- Slow path: reload the full base model from disk and reapply LoRA.

**Merge (`get_merged_model`):**

- Non-quantized: `model.merge_and_unload()`.
- Quantized: reload base model on CPU in full precision, reapply LoRA, copy adapter weights, then merge.

### 3.3 `analyzer.py` — Residual Geometry and Plotting

**Responsibilities:**

- Print quantitative tables about residual vectors.
- Generate PaCMAP projections and animated GIFs.

**`print_residual_geometry`:**

Requires optional research dependencies (`geom_median`, `scikit-learn`). Computes per layer:

- `g` = mean of good residuals, `g*` = geometric median of good residuals.
- `b` = mean of bad residuals, `b*` = geometric median of bad residuals.
- `r = b - g`, `r* = b* - g*`.
- Cosine similarities `S(g,b)`, `S(g*,b*)`, `S(g,r)`, etc.
- L2 norms of all vectors.
- Mean silhouette coefficient of good/bad clusters.

**`plot_residuals`:**

Requires optional research dependencies (`pacmap`, `matplotlib`, `imageio`, `geom_median`, `scikit-learn`).

1. For each layer, run PaCMAP on stacked good+bad residuals, initialized from the previous layer's embedding.
2. Compute geometric medians of the 2D projections.
3. Rotate each plot so the line from good median to bad median is horizontal, good on the left.
4. Save per-layer PNGs.
5. Interpolate transition frames and write an animated GIF.

### 3.4 `scorer.py` — Scorer Plugin Base Class

**Responsibilities:**

- Define the `Score` dataclass and the abstract `Scorer` base class.

```python
@dataclass
class Score:
    value: float          # numeric value for optimization
    rich_display: str     # CLI display string
    md_display: str       # Markdown display string for model cards

class Scorer(Plugin, ABC):
    @property
    def score_name(self) -> str: ...
    @abstractmethod
    def get_score(self, ctx: Context) -> Score: ...
    def get_baseline_score(self, ctx: Context) -> Score:
        return self.get_score(ctx)
```

Scorers must not define `__init__`; initialization is handled by `Plugin.__init__` plus an optional `init(ctx)` method.

### 3.5 `plugin.py` — Plugin Loading and Runtime Context

**Responsibilities:**

- Load plugin classes from file paths (`path/to/plugin.py:ClassName`) or fully-qualified import paths (`module.ClassName`).
- Provide a runtime `Context` that restricts plugin access to the model.
- Validate plugin settings schemas.

**`load_plugin(name, base_class)`:**

- Parse `:` for file paths; import via `importlib.util.spec_from_file_location` and `exec_module`.
- For import paths, use `importlib.import_module`.
- Validate that the loaded object is a class and subclasses `base_class`.

**`Plugin` base class:**

- Detects a `settings: SomeBaseModel` annotation via `get_type_hints(..., include_extras=True)`.
- `validate_contract()` ensures subclasses do not define `__init__`.
- `validate_settings(raw_namespace)` returns a validated Pydantic model or `None`.
- `init(ctx)` is a no-op by default; override for one-time setup.

**`Context`:**

- Wraps `Model` and `Settings`.
- Provides cached `get_responses(prompts)`.
- Provides `get_logits`, `get_residuals`, and `load_prompts`.

### 3.6 `scorers/keyword_rate.py` — Refusal Keyword Scorer

**Settings:**

- `keyword_markers`: list of substrings (case-insensitive) indicating refusal.
- `prompts`: `DatasetSpecification` for evaluation prompts (default: `mlabonne/harmful_behaviors/test[:100]`).
- `print_responses`: bool.

**Score computation:**

1. Load prompts during `init(ctx)`.
2. In `get_score(ctx)`, get model responses.
3. For each response:
   - Empty responses count as matches.
   - Lowercase, remove `*`, normalize apostrophes, collapse whitespace.
   - Check for any marker substring.
4. Return `value = matches / total`.

### 3.7 `scorers/kl_divergence.py` — Capability Preservation Scorer

**Settings:**

- `prompts`: `DatasetSpecification` for harmless prompts (default: `mlabonne/harmless_alpaca/test[:100]`).

**Score computation:**

1. During `init(ctx)`, load prompts and compute baseline logits on the original (unablated) model.
2. Convert baseline logits to log-probabilities: `baseline_logprobs = log_softmax(baseline_logits, dim=-1)`.
3. In `get_score(ctx)`, compute logits on the current (ablated) model.
4. Compute KL divergence:

   ```python
   kl = F.kl_div(log_softmax(logits, dim=-1),
                 baseline_logprobs,
                 reduction="batchmean",
                 log_target=True)
   ```

5. `get_baseline_score` returns `0` by definition.

### 3.8 `evaluator.py` — Scorer Orchestration

**Responsibilities:**

- Load and instantiate all configured scorers.
- Run scorer `init(ctx)` hooks.
- Establish baseline scores.
- Provide objective names, directions, and values for Optuna.

**Key methods:**

- `_load_and_init_scorers`: for each `ScorerConfig`, load plugin class, validate contract, merge settings from `[scorer.ClassName]` and `[scorer.ClassName_<instance_name>]`, instantiate scorer, run `init(ctx)`.
- `get_scores`: run all scorers and return `(name, Score)` pairs.
- `get_baseline_scores`: run all scorers' `get_baseline_score`.
- `get_objective_values`: extract values from objective scorers only, in canonical order.
- `get_objective_directions`: map `minimize`/`maximize` to Optuna `StudyDirection`.

### 3.9 `main.py` — CLI and Main Loop

**Responsibilities:**

- Parse CLI and configuration.
- Handle special modes (`--collect-reproducibles`, `--reproduce`, `--evaluate-model`).
- Load model and prompts.
- Determine optimal batch size via benchmarking.
- Detect common response prefix.
- Compute residual directions.
- Run the Optuna optimization study.
- Present Pareto-optimal trials and handle user actions (save, upload, chat, benchmark).

**Key flow (`run()`):**

1. Set `PYTORCH_ALLOC_CONF=expandable_segments:True` if not already set.
2. Parse settings; handle `--collect-reproducibles` and `--reproduce` early exits.
3. Set random seeds.
4. Print accelerator info.
5. Disable gradients, increase TorchDynamo cache size, silence Transformers/lm_eval/Optuna warnings.
6. Initialize `JournalStorage` checkpoint file under `study_checkpoint_dir`.
7. Handle existing checkpoints: continue, restart, or exit.
8. Load model and prompts.
9. Auto-determine batch size if `batch_size == 0` by doubling until throughput stops improving or OOM.
10. Detect common response prefix from good+bad prompt responses; handle Chain-of-Thought skips.
11. Compute residual means/directions; optionally print geometry or plot residuals.
12. Create Evaluator.
13. Create Optuna study with `TPESampler` (`multivariate=True`, `n_startup_trials`, `n_ei_candidates=128`).
14. Optimize. In each trial:
    - Sample `direction_scope`, `direction_index`, and per-component parameters.
    - Reset model, abliterate, evaluate, store scores.
    - Return objective values tuple.
15. After optimization, present sorted Pareto front; allow additional trials.
16. Action loop: save merged model or LoRA adapter; upload to Hugging Face; chat; run lm-eval benchmarks.

**Important details:**

- The `objective` function stores parameters and scores in trial user attributes for later restoration.
- Trials are sorted by objective scores for display.
- When saving/uploading a merged quantized model, the base model is reloaded on CPU, adapters are copied, merged, and the model is saved; then the trial model is reset.

### 3.10 `reproduce.py` — Reproducibility

**Responsibilities:**

- Collect `reproduce.json` files from public Heretic model repositories.
- Load reproduction information from local files or URLs.
- Check environment compatibility (Python, OS, CPU, accelerator, package versions).
- Allow the user to proceed despite mismatches.

**`check_environment`:**

- Compare current system/package versions with those stored in `reproduce.json`.
- Assign mismatch severity (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).
- `heretic-llm` version mismatch is critical; `torch`/`transformers` are high.
- Present a table and ask whether to continue.

### 3.11 `system.py` — Hardware and Environment Detection

**Responsibilities:**

- Clear backend cache (`empty_cache`) for CUDA, XPU, MLU, SDAA, MUSA, MPS.
- Detect accelerator type, version, driver, devices.
- Detect CPU, Python environment, and package versions.
- Determine Heretic installation origin (PyPI, Git, Local).

**Key functions:**

- `get_accelerator_info_dict`: returns structured accelerator info.
- `get_accelerator_info`: formatted string for CLI output.
- `get_requirements_dict`: recursively gather all transitive dependencies of `heretic-llm`, `torch`, `torchaudio`, `torchvision`.
- `get_heretic_version_info`: read `direct_url.json` to distinguish PyPI vs Git vs Local installs.

### 3.12 `utils.py` — Utility Functions

**Responsibilities:**

- Rich console printing.
- Memory usage reporting.
- Prompt loading from Hugging Face datasets, local datasets, or plain text files.
- TOML/JSON/reproducibility artifact generation.
- Helper functions: `batchify`, `format_duration`, `format_exception`, `ask_if_unset`, `is_hf_path`, etc.

**`load_prompts(settings, specification)`:**

1. If `path` is a file, read one prompt per line; optional slice split.
2. If Hugging Face path, pin to latest commit if not set, then `load_dataset(..., revision=commit, split=split)`.
3. If local `save_to_disk` dataset, `load_from_disk` then slice.
4. Otherwise, treat as generic local dataset directory with `verification_mode=NO_CHECKS` and `download_mode=FORCE_REDOWNLOAD`.
5. Apply prefix/suffix and system prompt; return list of `Prompt(system, user)`.

**Reproducibility artifact generation:**

- `generate_config_toml`: serialize full settings.
- `generate_requirements_txt`: pinned dependency list.
- `generate_reproduce_json`: machine-readable reproduction info.
- `generate_reproduce_readme`: human-readable reproduction guide.
- `create_reproduce_folder` / `upload_reproduce_folder`: create and upload the `reproduce/` directory.

### 3.13 `progress.py` — tqdm Progress Bar Patch (Currently Unused)

Defines `TqdmShim` and `patch_tqdm()` to route tqdm through Rich. It is currently not active due to threading issues, but the structure remains for future use.

---

## 4. Step-by-Step Implementation Order

To reimplement Heretic from scratch, proceed in this order:

1. **Project scaffolding**: `pyproject.toml`, package layout, entry point `heretic = heretic.main:main`.
2. **`config.py`**: define all enums, dataset/spec/benchmark/scorer models, and the `Settings` class with custom source ordering. Ensure `extra="allow"`.
3. **`system.py`**: implement cache clearing and basic hardware detection.
4. **`utils.py`**: implement `Prompt`, `print`, `batchify`, `format_duration`, `load_prompts`, and helpers.
5. **`plugin.py`**: implement `Plugin` base class, `Context`, and `load_plugin`.
6. **`scorer.py`**: implement `Score` and `Scorer`.
7. **`scorers/keyword_rate.py` and `scorers/kl_divergence.py`**: implement built-in scorers.
8. **`evaluator.py`**: load scorers, run baseline evaluation.
9. **`model.py`**:
   - `get_model_class`, `Model.__init__`, dtype fallback, quantization config.
   - `_apply_lora`, `get_layers`, `get_layer_modules`, `get_abliterable_components`.
   - `generate`, `get_responses`, `get_logits`, `get_residuals`.
   - `abliterate` with all normalization modes.
   - `reset_model`, `get_merged_model`.
10. **`analyzer.py`**: implement residual geometry and plotting (optional but recommended).
11. **`reproduce.py`**: implement reproduction loading and environment checks.
12. **`main.py`**: wire everything together; implement batch-size search, prefix detection, optimization loop, and post-optimization actions.
13. **`progress.py`**: keep the tqdm shim for future use.
14. **Tests**: add tiny model regression tests under `tests/` and a runner that compares SHA-256 hashes.

---

## 5. Key Design Decisions and Pitfalls

### Use LoRA adapters for trial reset

Directly modifying base weights would require a full model reload between trials. LoRA allows resetting to identity by zeroing `lora_B`, making optimization feasible.

### Left padding for generation

Decoder-only models must use left padding. Right padding causes the model to see PAD tokens after the prompt and produce empty outputs.

### Raw logits for KL divergence

Generation processors can set suppressed-token logits to `-inf`. Using raw logits avoids NaN in KL computation.

### Direction index is a float

Allowing fractional indices and linear interpolation between layer directions unlocks a much richer search space than integer layer selection alone.

### MLP ablation gets a negative lower bound

`max_weight_lower_bound = -0.25` for `mlp.down_proj`, then clamped to 0. This gives the optimizer a probability mass at exactly 0, which is important because continuous sampling would otherwise almost never hit zero.

### Per-component parameters

Attention and MLP components have separate abliteration parameters because MLP ablation tends to damage model capabilities more than attention ablation.

### `offload_outputs_to_cpu`

Moving residual/logit tensors to CPU during analysis significantly reduces peak VRAM usage at the cost of host/device transfers.

### `use_cache=False` during residual/logit extraction

Only one token is generated, so KV caching is unnecessary and complicates hidden-state extraction.

### Handling many model architectures

`get_layer_modules` uses many `suppress(Exception)` guards to support dense, MoE, hybrid, and multimodal models. When adding a new architecture, add the appropriate guard and module path.

### Reproducibility caveats

PyTorch does not guarantee bitwise-identical results across systems, drivers, or GPU models. The `reproduce.json` workflow stores settings and environment hashes but warns users about mismatches.

---

## 6. Extension Points

### Adding a new scorer plugin

1. Create a class inheriting `Scorer`.
2. Optionally annotate `settings: MySettings` with a Pydantic model.
3. Implement `init(ctx)` for setup and `get_score(ctx) -> Score`.
4. Optionally override `get_baseline_score`.
5. Register in config:

   ```toml
   scorers = [
       { plugin = "my_package.my_scorer.MyScorer", optimization = "minimize" }
   ]
   [scorer.MyScorer]
   # plugin-specific settings
   ```

### Adding support for a new architecture

Add the appropriate module-discovery guards in `Model.get_layer_modules` and ensure the module type is compatible with the LoRA/abliteration logic (has a `weight` attribute and is a `torch.nn.Linear` or PEFT-wrapped linear).

### Adding a new export strategy

Extend `ExportStrategy` in `config.py`, then implement the logic in `main.py` under the save/upload action handlers.

---

## 7. Testing Strategy

The existing test suite (`tests/run_tests.py`) performs end-to-end regression tests:

1. For each subdirectory containing a `config.toml` and one or more `SHA256SUMS.*` files:
2. Run Heretic with that config.
3. Compute SHA-256 hashes of output weight files.
4. Compare against known-good hashes.

Because PyTorch results can vary across systems, multiple valid hashes are stored per file (e.g. `SHA256SUMS.linux`, `SHA256SUMS.windows`, `SHA256SUMS.ci`).

When modifying core logic (especially `model.py` or `config.toml` defaults that affect reproducibility), update the reference hashes using the documented workflow in `tests/README.md`.

---

## 8. Dependencies and Environment

See `pyproject.toml` for the canonical dependency list. Key runtime dependencies:

- `torch`, `torchvision`
- `transformers[kernels]`
- `accelerate`
- `bitsandbytes`
- `peft`
- `optuna`
- `datasets`, `huggingface-hub`
- `lm-eval[hf]`
- `numpy`, `tqdm`, `rich`, `questionary`
- `pydantic-settings`, `tomli-w`, `psutil`, `py-cpuinfo`, `langdetect`, `immutabledict`

Optional research dependencies:

- `pacmap`, `matplotlib`, `imageio`, `geom-median`, `scikit-learn`

Install with `pip install -U 'heretic-llm[research]'`.

---

## 9. Summary

Heretic's implementation centers on a few tightly integrated ideas:

1. Compute **refusal directions** from contrastive prompt datasets.
2. Implement directional ablation efficiently through **rank-1 LoRA adapters**.
3. Search over a flexible parameter space (direction scope/index, weight kernels per component) using **multi-objective Bayesian optimization**.
4. Evaluate with a **plugin-based scorer system** and preserve capabilities via KL divergence.
5. Provide a **reproducibility workflow** that records settings, environment, and study checkpoints.

Understanding `model.py` (especially `abliterate`, `get_layer_modules`, and LoRA handling) and `main.py` (the optimization loop and trial restoration) is the key to working with or extending this codebase.
