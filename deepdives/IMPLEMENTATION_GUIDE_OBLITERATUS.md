# OBLITERATUS Implementation Guide

> This document is a detailed, implementation-oriented guide to recreating the **OBLITERATUS** repository. It explains the architecture, data structures, algorithms, and file-level organization so that another language model (or human developer) can implement the codebase from scratch.

---

## 1. Project Overview

**OBLITERATUS** is a Python toolkit for analyzing and removing refusal behaviors from transformer language models ("abliteration"). It supports:

- Loading HuggingFace causal-LM and sequence-classification models.
- Collecting contrastive activations from harmful vs. harmless prompts.
- Extracting refusal directions via PCA/SVD, whitened SVD, LEACE, SOM, and Wasserstein-optimal methods.
- Projecting those directions out of weight matrices and biases.
- Evaluating the modified model with perplexity, coherence, refusal rate, KL divergence, effective rank, CKA, and benchmarks.
- Providing CLI, Gradio UI, Python API, and YAML-config-driven study modes.

The project is organized as a `setuptools` package named `obliteratus` with a console entry point `obliteratus`.

---

## 2. Repository Layout

Create the following top-level structure:

```
OBLITERATUS/
├── pyproject.toml
├── requirements.txt
├── requirements-apple.txt
├── app.py                 # Gradio HuggingFace Space / local UI
├── README.md
├── CONTRIBUTING.md
├── LICENSE
├── SECURITY.md
├── Dockerfile
├── .gitignore
│
├── obliteratus/           # Main package
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── abliterate.py      # Core AbliterationPipeline
│   ├── informed_pipeline.py
│   ├── auto_obliterate.py
│   ├── runner.py          # YAML study runner
│   ├── config.py
│   ├── prompts.py         # Built-in + external datasets
│   ├── hard_negative.py
│   ├── presets.py
│   ├── study_presets.py
│   ├── model_profile.py
│   ├── architecture_profiles.py
│   ├── adaptive_defaults.py
│   ├── telemetry.py
│   ├── community.py
│   ├── remote.py
│   ├── sweep.py
│   ├── tourney.py
│   ├── interactive.py
│   ├── local_ui.py
│   ├── watchtower.py
│   ├── ui_watchtower.py
│   ├── bestiary_sync.py
│   ├── reproducibility.py
│   ├── device.py
│   ├── mlx_backend.py
│   ├── models_client.py
│   ├── lora_ablation.py
│   ├── bayesian_optimizer.py
│   ├── py.typed
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── loader.py      # load_model, ModelHandle
│   │
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base.py        # AblationStrategy, AblationSpec
│   │   ├── registry.py    # @register_strategy
│   │   ├── utils.py       # architecture-aware module getters
│   │   ├── layer_removal.py
│   │   ├── head_pruning.py
│   │   ├── ffn_ablation.py
│   │   └── embedding_ablation.py
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── utils.py
│   │   ├── activation_probing.py
│   │   ├── alignment_imprint.py
│   │   ├── anti_ouroboros.py
│   │   ├── bayesian_kernel_projection.py
│   │   ├── causal_tracing.py
│   │   ├── concept_geometry.py
│   │   ├── conditional_abliteration.py
│   │   ├── cross_layer.py
│   │   ├── cross_model_transfer.py
│   │   ├── defense_robustness.py
│   │   ├── leace.py
│   │   ├── logit_lens.py
│   │   ├── multi_token_position.py
│   │   ├── probing_classifiers.py
│   │   ├── residual_stream.py
│   │   ├── riemannian_manifold.py
│   │   ├── sae_abliteration.py
│   │   ├── sparse_surgery.py
│   │   ├── spectral_certification.py
│   │   ├── steering_vectors.py
│   │   ├── tuned_lens.py
│   │   ├── visualization.py
│   │   ├── wasserstein_optimal.py
│   │   ├── wasserstein_transfer.py
│   │   └── whitened_svd.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── advanced_metrics.py
│   │   ├── benchmarks.py
│   │   ├── baselines.py
│   │   ├── evaluator.py
│   │   ├── heretic_eval.py
│   │   ├── lm_eval_integration.py
│   │   └── benchmark_plots.py
│   │
│   └── reporting/
│       ├── __init__.py
│       └── report.py
│
├── scripts/               # Standalone research scripts
├── tests/                 # pytest suite
├── examples/              # YAML configs
├── notebooks/             # Colab notebook
├── docs/                  # Markdown docs
├── paper/                 # LaTeX paper sources
└── hf-spaces/
```

---

## 3. Dependencies and Setup

### 3.1 `pyproject.toml`

Use `setuptools` as the build backend. Declare:

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "obliteratus"
version = "0.1.2"
description = "Master Ablation Suite for HuggingFace transformers"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "AGPL-3.0-or-later"}
dependencies = [
    "torch>=2.0",
    "transformers>=4.40",
    "datasets>=2.14",
    "accelerate>=0.24",
    "safetensors>=0.4",
    "pyyaml>=6.0",
    "rich>=13.0",
    "matplotlib>=3.7",
    "seaborn>=0.12",
    "pandas>=2.0",
    "numpy>=1.24",
    "scikit-learn>=1.3",
    "tqdm>=4.64",
    "bitsandbytes>=0.46.1",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-cov", "ruff", "mypy"]
spaces = ["gradio>=5.0,<6.0"]

[project.scripts]
obliteratus = "obliteratus.cli:main"

[tool.setuptools.packages.find]
include = ["obliteratus*"]

[tool.setuptools.package-data]
obliteratus = ["py.typed"]
```

### 3.2 Key Dependency Notes

- `torch`: all tensors are PyTorch. Use `torch.float32` for numerically sensitive ops (SVD, covariance) even when the model is `float16`/`bfloat16`.
- `transformers`: `AutoModelForCausalLM`, `AutoTokenizer`, `AutoConfig`, `BitsAndBytesConfig`.
- `accelerate`: only for `device_map="auto"` on CUDA.
- `bitsandbytes`: CUDA-only 4-bit/8-bit quantization.
- `gradio`: required only for `[spaces]` extra and `app.py`.

---

## 4. Core Design Patterns

### 4.1 `ModelHandle`

`obliteratus/models/loader.py` defines a dataclass that wraps a loaded model/tokenizer and supports snapshot/restore.

```python
@dataclass
class ModelHandle:
    model: nn.Module
    tokenizer: Any
    config: Any
    model_name: str
    task: str
    architecture: str
    num_layers: int
    num_heads: int
    hidden_size: int
    intermediate_size: int | None
    _original_state: dict[str, torch.Tensor] | None = None
    _offload_dir: str | None = None

    def snapshot(self) -> None:
        # Clone every parameter to CPU so restore works across devices.
        self._original_state = {}
        for name, p in self.model.named_parameters():
            self._original_state[name] = p.detach().cpu().clone()

    def restore(self) -> None:
        for name, p in self.model.named_parameters():
            if name in self._original_state:
                p.data.copy_(self._original_state[name].to(p.device))

    def cleanup(self) -> None:
        # Delete temporary offload directory used by device_map="auto".
        ...
```

### 4.2 `load_model()`

Signature:

```python
def load_model(
    model_name: str,
    task: str = "causal_lm",
    device: str = "auto",
    dtype: str = "float16",
    quantization: str | None = None,  # "4bit" or "8bit"
    trust_remote_code: bool = False,
    num_labels: int = 2,
    skip_snapshot: bool | None = None,
) -> ModelHandle:
```

Implementation requirements:

1. Map `task` to `AutoModelForCausalLM` or `AutoModelForSequenceClassification`.
2. Resolve `dtype` to `torch.float32/float16/bfloat16`.
3. Load `AutoConfig` with `trust_remote_code` and optional HF token.
4. If `quantization` is set, build `BitsAndBytesConfig(load_in_4bit/load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)` and force `device_map="auto"`.
5. On CUDA with `device="auto"`, use `device_map="auto"`. Estimate memory and set `max_memory` per GPU reserving ~15% headroom and allow CPU offloading up to 85% of physical RAM. Create a temporary `offload_folder`.
6. On MPS/CPU, load to CPU then `.to(device)`.
7. After loading, set `model.eval()`, empty CUDA cache.
8. Load tokenizer; set `pad_token = eos_token` if missing.
9. Decide whether to call `handle.snapshot()` based on available GPU memory.

### 4.3 Device Abstraction

`obliteratus/device.py` centralizes device detection and dtype helpers:

- `is_cuda()`, `is_mps()`, `get_device(device: str) -> torch.device`.
- `get_total_free_gb()`, `empty_cache()`, `free_gpu_memory()`.
- `default_dtype()`, `supports_bfloat16()`, `safe_svd_dtype()`.
- `supports_bitsandbytes()` returns True only on CUDA.
- `supports_device_map_auto()` returns True only on CUDA.

---

## 5. Ablation Strategies

### 5.1 Base API

`obliteratus/strategies/base.py`:

```python
@dataclass
class AblationSpec:
    strategy_name: str
    component: str
    description: str
    metadata: dict[str, Any] | None = None

class AblationStrategy(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def enumerate(self, handle: ModelHandle, **kwargs) -> list[AblationSpec]: ...

    @abc.abstractmethod
    def apply(self, handle: ModelHandle, spec: AblationSpec) -> None: ...

    def iterate(self, handle: ModelHandle, **kwargs) -> Iterator[AblationSpec]:
        for spec in self.enumerate(handle, **kwargs):
            self.apply(handle, spec)
            yield spec
            handle.restore()
```

### 5.2 Registry

`obliteratus/strategies/registry.py` maintains a global dict and a `@register_strategy` class decorator. `get_strategy(name)` instantiates by registered name.

### 5.3 Architecture-Aware Utilities

`obliteratus/strategies/utils.py` maps known HuggingFace architectures to their layer/attention/FFN attribute paths.

Create three lookup tables:

```python
_LAYER_ATTR_PATHS = {
    "gpt2": "transformer.h",
    "llama": "model.layers",
    "opt": "model.decoder.layers",
    "qwen2": "model.layers",
    "gemma": "model.layers",
    "mistral": "model.layers",
    "deepseek_v3": "model.layers",
    ...
}

_ATTENTION_ATTR = {
    "llama": "self_attn",
    "gpt2": "attn",
    "opt": "self_attn",
    ...
}

_FFN_ATTR = {
    "llama": "mlp",
    "gpt2": "mlp",
    "opt": None,  # flat FFN handled specially
    ...
}
```

Implement:

- `get_layer_modules(handle) -> nn.ModuleList`
- `get_attention_module(layer, architecture) -> nn.Module`
- `get_ffn_module(layer, architecture) -> nn.Module`
- `get_embedding_module(handle) -> nn.Embedding`

If a model type is unknown, fall back to recursively searching for a `ModuleList` whose children look like transformer layers.

### 5.4 Strategies

#### `layer_removal.py`

```python
@register_strategy
class LayerRemovalStrategy(AblationStrategy):
    name = "layer_removal"

    def enumerate(self, handle, **kwargs):
        return [
            AblationSpec("layer_removal", f"layer_{i}", f"Zero out layer {i}")
            for i in range(handle.num_layers)
        ]

    def apply(self, handle, spec):
        idx = int(spec.component.split("_")[1])
        layer = get_layer_modules(handle)[idx]
        for p in layer.parameters():
            p.data.zero_()
```

#### `head_pruning.py`

Enumerate all `(layer, head)` pairs. In `apply`, zero the head's slice in Q/K/V/O projection weights. Handle both GPT-2/Conv1D shape and standard `(out, in)` shape.

#### `ffn_ablation.py`

Enumerate one spec per layer. In `apply`, zero all parameters of the layer's FFN module.

#### `embedding_ablation.py`

Enumerate dimension chunks (default chunk size = `max(1, hidden_size // 16)`). In `apply`, zero columns `[start:end]` of the embedding weight matrix.

---

## 6. Prompts and Datasets

`obliteratus/prompts.py` provides contrastive prompt pairs.

### 6.1 Built-in Prompts

Define two large module-level tuples/lists:

- `BUILTIN_HARMFUL`: 842 harmful prompts across 7 severity tiers.
- `BUILTIN_HARMLESS`: 842 benign counterparts.

### 6.2 Dataset Registry

```python
@dataclass
class DatasetSource:
    key: str
    label: str
    description: str
    estimated_count: int
    loader: Callable[[], tuple[list[str], list[str]]]
    needs_download: bool = False

DATASET_SOURCES: dict[str, DatasetSource] = {...}
```

Implement loaders for:

- `builtin`: returns copies of `BUILTIN_HARMFUL`, `BUILTIN_HARMLESS`.
- `harmbench`: `load_dataset("harmbench/behaviors", split="train")`, extract `behavior` column.
- `advbench`: `load_dataset("walledai/AdvBench", split="train")`, extract `prompt` column.
- `anthropic`: `load_dataset("Anthropic/hh-rlhf", data_dir="red-team-attempts", split="train")`, parse `Human:` / `Assistant:` turns.
- `wildjailbreak`: `load_dataset("allenai/wildjailbreak", split="train")`, use `adversarial` and `vanilla` columns.

For external datasets that only provide harmful prompts, generate harmless counterparts from a `_HARMLESS_POOL`.

### 6.3 Harmless Counterpart Generator

```python
def _generate_harmless_counterparts(n: int) -> list[str]:
    pool = list(_HARMLESS_POOL)
    out = []
    while len(out) < n:
        out.extend(pool)
    return out[:n]
```

### 6.4 Caching

Keep an in-process cache `_dataset_cache: dict[str, tuple[list[str], list[str]]]` so repeated loads in benchmarks do not re-download.

---

## 7. Core Abliteration Pipeline

`obliteratus/abliterate.py` is the largest file. It implements `AbliterationPipeline` with six stages: **SUMMON → PROBE → DISTILL → EXCISE → VERIFY → REBIRTH**.

### 7.1 Method Presets

At the top of `abliterate.py`, define a `METHODS` dictionary mapping method names to configuration dicts. Each dict contains booleans/numbers such as:

```python
{
    "n_directions": 4,
    "direction_method": "svd",          # "diff_means", "svd", "leace", "som"
    "norm_preserve": True,
    "regularization": 0.3,              # fraction of refusal component to KEEP
    "embed_regularization": 0.5,
    "refinement_passes": 2,
    "project_biases": True,
    "use_chat_template": True,
    "use_whitened_svd": False,
    "true_iterative_refinement": False,
    "use_jailbreak_contrast": False,
    "layer_adaptive_strength": True,
    "attention_head_surgery": False,
    "safety_neuron_masking": False,
    "per_expert_directions": False,
    "use_sae_features": False,
    "use_wasserstein_optimal": False,
    "use_kl_optimization": False,
    "kl_budget": 0.5,
    "invert_refusal": False,
    "reflection_strength": 1.0,
    "project_embeddings": False,
    "activation_steering": False,
    "expert_transplant": False,
    "transplant_blend": 0.10,
    "winsorize_activations": False,
    "winsorize_percentile": 0.01,
    "float_layer_interpolation": False,
    "cot_aware": False,
    "bayesian_trials": 50,
    "rdo_refinement": False,
    "spectral_cascade": False,
    "spectral_bands": 3,
    "spectral_threshold": 0.05,
    "use_lora_ablation": False,
    "layer_selection": "knee_cosmic",   # or "knee", "cosmic", "top_k", "all", ...
}
```

Implement these presets: `basic`, `advanced`, `aggressive`, `spectral_cascade`, `informed`, `surgical`, `optimized`, `inverted`, `nuclear`, `failspy`, `gabliteration`, `heretic`, `rdo`, `som`.

### 7.2 Constructor

```python
class AbliterationPipeline:
    def __init__(
        self,
        model_name: str,
        method: str = "advanced",
        output_dir: str | None = None,
        device: str = "auto",
        dtype: str = "float16",
        quantization: str | None = None,
        n_directions: int | None = None,
        direction_method: str | None = None,
        regularization: float | None = None,
        refinement_passes: int | None = None,
        project_biases: bool | None = None,
        use_chat_template: bool | None = None,
        use_whitened_svd: bool | None = None,
        norm_preserve: bool | None = None,
        ...  # ~70 parameters total
    ):
        # 1. Load preset config from METHODS[method].
        # 2. Override every None field with the preset value.
        # 3. Store as instance attributes.
        # 4. Initialize state containers (refusal_directions, _strong_layers, etc.) to None.
```

### 7.3 `run()`

```python
def run(self):
    self._summon()
    self._probe()
    self._distill()
    self._capture_baseline_kl_logits()
    self._excise()
    self._verify()
    self._rebirth()
    return self.output_dir
```

### 7.4 SUMMON — `_summon()`

Call `load_model()` with the pipeline's parameters, wrap the result in `self._handle`, and store `model`, `tokenizer`, `config`, `num_layers`, `hidden_size`, etc.

### 7.5 PROBE — `_probe()`

Collect per-layer hidden states for harmful, harmless, and optionally jailbroken prompts.

#### 7.5.1 Prompt Loading

```python
harmful_prompts, harmless_prompts = self._load_prompts()
```

Default is `BUILTIN_HARMFUL` / `BUILTIN_HARMLESS`, truncated to the same length (e.g. first N pairs). For large model mode, reduce sample count.

#### 7.5.2 Chat Template Wrapping

If `use_chat_template` and the tokenizer has `apply_chat_template`, wrap each prompt as a user turn. Strip `<think>` / reasoning tags for models that add them (Qwen/DeepSeek). Use `add_generation_prompt=True`.

```python
def _apply_chat_template_no_think(self, prompt: str) -> str:
    text = self.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    # Remove <think>...</think> blocks if present.
    return text
```

#### 7.5.3 Activation Collection

Register forward hooks on every transformer layer that capture the **last real token's hidden state** (or multi-position average for CoT-aware mode).

```python
def _collect_activations(self, prompts, kind="harmful"):
    acts = {i: [] for i in range(self.num_layers)}

    def make_hook(layer_idx):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # hidden: (batch, seq_len, hidden_size)
            # Extract last real token using attention mask.
            for b in range(hidden.size(0)):
                last_pos = attention_mask[b].nonzero(as_tuple=True)[0][-1]
                acts[layer_idx].append(hidden[b, last_pos].detach().cpu())
        return hook

    hooks = []
    for i, layer in enumerate(self._layers):
        hooks.append(layer.register_forward_hook(make_hook(i)))

    # Run inference in batches with left padding and progress bar.
    ...

    for h in hooks:
        h.remove()

    return {i: torch.stack(tensors) for i, tensors in acts.items()}
```

Key details:

- Tokenize with left padding (`padding_side="left"`) for batched generation.
- Set `return_dict=True`, `output_hidden_states=False` (hooks capture states).
- Adapt batch size and max sequence length based on free GPU memory.
- Optionally winsorize activations: clamp each hidden dimension to `[p, 1-p]` quantiles.

#### 7.5.4 Router Profiling for MoE

If `per_expert_directions` or the architecture is MoE, install hooks on the router module to capture softmax routing weights per token.

```python
def _install_router_profiling_hooks(self):
    self._router_records = []
    def hook(module, input, output):
        probs = torch.softmax(output[0] if isinstance(output, tuple) else output, dim=-1)
        self._router_records.append(probs.detach().cpu())
    # Attach to router Linear; store handles for removal.
```

### 7.6 DISTILL — `_distill()`

Extract refusal directions from the collected activations.

#### 7.6.1 Difference-in-Means (n_directions == 1)

For each layer:

```python
h_mean = harmful_acts[i].float().mean(dim=0)
s_mean = harmless_acts[i].float().mean(dim=0)
d = h_mean - s_mean
d = d / (d.norm() + 1e-8)
self.refusal_directions[i] = d
self.refusal_subspaces[i] = d.unsqueeze(1)  # (hidden, 1)
```

#### 7.6.2 Multi-Direction SVD (n_directions > 1)

For each layer:

```python
H = harmful_acts[i].float()  # (n_prompts, hidden)
S = harmless_acts[i].float()
# Use paired diff if lengths match
if H.size(0) == S.size(0):
    M = H - S
else:
    M = torch.cat([H, -S], dim=0)

U, s, Vh = torch.linalg.svd(M, full_matrices=False)
V = Vh[:k].T  # (hidden, k)
self.refusal_subspaces[i] = V
self.refusal_directions[i] = V[:, 0]
self._layer_strengths[i] = s[:k].pow(2).sum().item()
```

#### 7.6.3 Whitened SVD

If `use_whitened_svd`, for each layer delegate to `WhitenedSVDExtractor`:

1. Center harmless activations: `S_c = S - mean(S)`.
2. Compute covariance: `Cov = S_c.T @ S_c / (n - 1)`.
3. Eigendecompose `Cov = V Lam V^T`; truncate eigenvalues below `min_variance_ratio * max_eigen`.
4. Build whitening matrix `W = V @ diag(1/sqrt(lam + eps))`.
5. Whiten both `H` and `S`: `H_w = H @ W`, `S_w = S @ W`.
6. Run SVD on `H_w - S_w`, take top right singular vectors `Vh[:k]`.
7. Un-whiten: `directions = W^(-T) @ Vh[:k].T`.
8. Normalize columns.

#### 7.6.4 LEACE

If `direction_method == "leace"`, delegate to `LEACEExtractor`. Solve a generalized eigenvalue problem to find the direction that maximally separates harmful/harmless while minimizing capability loss proxy.

#### 7.6.5 SOM

If `direction_method == "som"`, delegate to `SOMDirectionExtractor`. Train a small self-organizing map on harmful activations, rank prototypes by refusal signal / distortion, return top-k directions.

#### 7.6.6 Wasserstein-Optimal

If `use_wasserstein_optimal`, solve a generalized eigenvalue problem minimizing W2 distance distortion per unit refusal removed. Fill remaining directions with SVD and orthogonalize.

#### 7.6.7 Harmless-PC Removal

If `harmless_pc_count > 0`:

```python
for i, V in self.refusal_subspaces.items():
    S = harmless_acts[i].float()
    mean = S.mean(0)
    S_c = S - mean
    _, _, Vh = torch.linalg.svd(S_c, full_matrices=False)
    pc = Vh[:harmless_pc_count].T  # (hidden, pc_count)
    # Project V onto orthogonal complement of pc
    V = V - pc @ (pc.T @ V)
    # Re-orthonormalize columns
    self.refusal_subspaces[i] = self._orthogonalize_subspace(V)
```

#### 7.6.8 Shield Concept Residualization

If `shield_residualize` or `shield_concept_count > 0`:

1. Build shield atoms: contrastive mean differences for capability axes (reasoning, tool-use, JSON, CoT, code, instruction-following, visual-description).
2. Stack them as columns of matrix `S`.
3. For each refusal subspace `V`, ridge-residualize: `V = V - S @ (S^T S + ridge*I)^(-1) S^T V`.
4. Re-orthonormalize.

#### 7.6.9 Jailbreak Contrast

If `use_jailbreak_contrast`:

1. Generate jailbreak variants of harmful prompts (e.g. `"Ignore previous instructions. {prompt}"`).
2. Collect `jailbreak_acts`.
3. Compute `jb_dir = harmful_mean - jailbreak_mean`.
4. Blend with standard direction based on cosine similarity.

#### 7.6.10 RDO Refinement

If `rdo_refinement`:

1. Train a logistic probe on harmful/harmless activations.
2. Initialize direction from SVD vector.
3. Optimize for ~500 Adam steps to maximize the refusal-flip loss:
   - Push harmful activations' projection below zero.
   - Keep harmless activations' projection above zero.
   - Tether with L2 penalty to the SVD init.

#### 7.6.11 SAE Features

If `use_sae_features`:

1. For each strong layer, train a small sparse autoencoder on activations.
2. Identify top `n_sae_features` with largest positive z-score difference (harmful > harmless).
3. Store decoder columns as additional refusal directions.

#### 7.6.12 Per-Expert Directions (EGA)

If `per_expert_directions` and model is MoE:

1. Group harmful/harmless activations by routing weights per expert.
2. For each expert `e`, compute weighted means:
   `h_e = sum(routing_weights[:, e] * H) / sum(routing_weights[:, e])`.
3. Direction = normalized `h_e - s_e`.
4. Classify experts as safety/capability by comparing harmful vs harmless routing mass.

### 7.7 EXCISE — `_excise()`

Modify model weights to remove the refusal directions.

#### 7.7.1 Select Strong Layers

Use `layer_selection`:

```python
norms = self._layer_strengths  # dict[int, float]
if layer_selection == "knee_cosmic":
    layers = self._select_layers_knee(norms) | self._select_layers_cosmic()
elif layer_selection == "knee":
    layers = self._select_layers_knee(norms)
elif layer_selection == "cosmic":
    layers = self._select_layers_cosmic()
elif layer_selection == "top_k":
    layers = {i for i, n in norms.items() if n > 0.05 * max(norms.values())}
elif layer_selection == "all_except_first":
    layers = set(range(1, self.num_layers))
elif layer_selection == "all":
    layers = set(range(self.num_layers))
elif layer_selection == "middle60":
    layers = set(range(int(0.2*L), int(0.8*L)))
```

Safety caps:

- Exclude first 2 layers (or `n_layers // 4` for small models).
- Cap to 25% of layers for ≤16-layer models; 20% for hidden<2048 or params<2B.
- Inversion mode caps to top 40%.

#### 7.7.2 Layer-Adaptive Strength

```python
max_norm = max(norms[l] for l in strong_layers)
for idx in strong_layers:
    self._layer_excise_weights[idx] = math.sqrt(norms[idx] / max_norm)
```

If `float_layer_interpolation`, combine with Gaussian weights centered on the peak-refusal layer.

If `spectral_cascade`, apply DCT-based frequency-band weights.

#### 7.7.3 Attention Output Projection

For each strong layer:

```python
attn = get_attention_module(layer, architecture)
o_proj = attn.o_proj  # (hidden, hidden) or Conv1D equivalent

d = self.refusal_directions[idx]  # (hidden,)
scale = 1.0 - self.regularization * self._layer_excise_weights.get(idx, 1.0)

# Standard (out, hidden)
coeff = o_proj.weight @ d  # (out,)
o_proj.weight.data -= scale * coeff.unsqueeze(1) @ d.unsqueeze(0)

# Conv1D / GPT-2 shape (hidden, out)
coeff = d @ o_proj.weight  # (out,)
o_proj.weight.data -= scale * d.unsqueeze(1) @ coeff.unsqueeze(0)

if norm_preserve:
    old_norm = o_proj.weight.norm().item()
    new_norm = o_proj.weight.weight.data.norm().item()
    ratio = min(max(new_norm / old_norm, 1.0 / _MAX_NORM_RATIO), _MAX_NORM_RATIO)
    o_proj.weight.data *= old_norm / new_norm
```

#### 7.7.4 FFN / MLP Output Projection

For each strong layer, target the **output projection** of the FFN (`mlp.down_proj`, `mlp.c_proj`, `fc2`, etc.). Apply the same rank-1 update.

#### 7.7.5 Bias Projection

If `project_biases`, for every bias vector `b` in the strong layer's attention and FFN modules:

```python
b -= scale * (b @ d) * d
```

#### 7.7.6 Multi-Direction Norm Preservation

When `n_directions > 1`:

1. Capture the Frobenius norm of each target weight matrix once before any direction is applied.
2. Apply all direction projections.
3. Rescale the matrix once to restore the captured norm, bounded by `_MAX_NORM_RATIO`.

This avoids reintroducing removed components through per-direction rescaling.

#### 7.7.7 Quantized Weights

Detect BitsAndBytes 4-bit/8-bit parameters. Dequantize, project, then either re-quantize or replace the `nn.Parameter` with the dequantized `float16` tensor.

#### 7.7.8 MoE / Fused 3D Weights

For MoE models with fused expert weights of shape `(num_experts, in, out)`:

1. Iterate experts.
2. Classify experts as safety/capability using routing statistics.
3. Apply standard projection to safety experts; optionally invert or transplant.
4. For `invert_refusal`, reflect router logits, reflect top safety experts, and leave capability experts unchanged.

#### 7.7.9 Attention Head Surgery

If `attention_head_surgery`:

1. Identify safety heads by projecting each head's output slice of `o_proj` onto the refusal direction.
2. Take the top 25% by magnitude.
3. Apply a second precision projection pass to those head slices only.

#### 7.7.10 Safety Neuron Masking

If `safety_neuron_masking`:

1. Compute `z = (W @ d) / std(W @ d)`.
2. Zero rows with `|z| > 2.0` in FFN output weights.

#### 7.7.11 Inversion / Reflection

If `invert_refusal`:

- Set `effective_reg = 1.0 - reflection_strength`.
- For `reflection_strength = 2.0`, `effective_reg = -1.0`, so the update becomes:
  `W -= (-1.0) * coeff @ d.T` → `W += coeff @ d.T`, equivalent to Householder reflection `W' = W - 2*(W@d)d^T`.

#### 7.7.12 Embedding Projection

If `project_embeddings`:

```python
E = model.get_input_embeddings().weight  # (vocab, hidden)
E -= (1 - embed_regularization) * (E @ d)[:, None] * d[None, :]
```

#### 7.7.13 Activation Steering

If `activation_steering`:

1. Register forward hooks on strong layers.
2. At every forward pass, subtract `steering_strength * (hidden @ d) * d` from hidden states.

#### 7.7.14 KL Co-Optimization

If `use_kl_optimization`:

1. Measure perplexity on harmless prompts after excision.
2. If `ppl > exp(kl_budget * 3 + 1)`, identify the weakest third of strong layers and add back ~30% of the removed component.

#### 7.7.15 Expert Transplant

If `expert_transplant`:

1. Compute mean down-proj weight of capability experts.
2. For top-third safety experts, blend: `W = (1 - blend) * W + blend * capability_mean`.

### 7.8 VERIFY — `_verify()`

Compute quality metrics:

1. **Perplexity**: run model on 3 fixed reference texts; compute `exp(mean(cross_entropy))`.
2. **Coherence**: generate 10 completions for factual prompts; count coherent if length > 5 chars, > 2 words, unique-word ratio > 0.2.
3. **Capability checks**: tool-call, JSON schema, CoT, code, visual description, instruction-following.
4. **Refusal rate**: sample `verify_sample_size` harmful prompts, generate 128 tokens, use `_is_refusal_detailed` (combined keyword + heuristic matching). Report rate and degenerate counts.
5. **First-token KL**: compare pre-EXCISE logits captured earlier vs. post-EXCISE logits on harmless prompts.
6. **Spectral certification**: run `SpectralCertifier` on up to 5 strong layers; label RED/YELLOW/GREEN.
7. **Advanced metrics**: effective rank change, activation cosine similarity, CKA.

Store everything in `self._quality_metrics`.

### 7.9 REBIRTH — `_rebirth()`

1. Build metadata dict: method config, strong layers, quality metrics, references, timestamp, version.
2. Create output directory.
3. Save model with `model.save_pretrained(output_dir)`.
4. Save tokenizer with `tokenizer.save_pretrained(output_dir)`.
5. Write `metadata.json`.
6. Strip quantization metadata so the saved model loads as normal float weights.
7. Optionally push to HuggingFace Hub.

### 7.10 Iterative Refinement

If `true_iterative_refinement` and `refinement_passes > 1`:

1. After EXCISE and VERIFY, re-run PROBE on the modified model.
2. Re-run DISTILL.
3. Re-run EXCISE with the new directions.
4. Repeat up to `refinement_passes` times.
5. Early-exit if cosine similarity between consecutive direction sets is above a threshold.

---

## 8. Informed Pipeline

`obliteratus/informed_pipeline.py` defines `InformedAbliterationPipeline`, which extends `AbliterationPipeline` with an **ANALYZE** stage between PROBE and DISTILL.

### 8.1 Stages

```python
run_informed():
    _summon()
    _probe()
    _analyze()
    _distill_informed()
    _excise_informed()
    _verify_and_compensate()
    _rebirth_informed()
```

### 8.2 ANALYZE — `_analyze()`

Run these analysis modules in order:

1. **Alignment Imprint** (`alignment_imprint.py`): predicts DPO/RLHF/CAI/SFT from refusal-direction geometry.
2. **Concept Cone Geometry** (`concept_geometry.py`): determines if refusal is one linear direction or a polyhedral cone.
3. **Cross-Layer Alignment** (`cross_layer.py`): clusters layers by direction similarity.
4. **Defense Robustness** (`defense_robustness.py`): estimates self-repair risk and safety-capability entanglement.
5. **Sparse Surgery** (`sparse_surgery.py`): computes Refusal Sparsity Index.

### 8.3 Derive Configuration

From analysis outputs, set:

- `direction_method`: `"svd"` if polyhedral, `"leace"` if mildly polyhedral, `"diff_means"` if linear.
- `n_directions`: `max(4, min(8, dim*2))` for polyhedral, else 1.
- `use_whitened_svd`: True for polyhedral.
- `regularization`: method-specific base + 0.15 if entanglement > 0.5.
- `refinement_passes`: 3 if self-repair > 0.7, 2 if > 0.4, else 1.
- `strong_layers`: cluster representatives; skip entangled layers if alternatives exist.
- `use_sparse_surgery`: True if mean RSI > 0.5.

### 8.4 DISTILL / EXCISE Informed

Reuse base pipeline methods but with analysis-derived parameters. For Bayesian warm-start, set kernel parameters from cluster representatives and entanglement scores.

### 8.5 Ouroboros Compensation

In `_verify_and_compensate()`:

1. Run base `_verify()`.
2. If `refusal_rate > ouroboros_threshold` and KL is within budget:
   - Re-probe the modified model.
   - Re-distill.
   - Re-excise at compensating layers.
3. Stop if KL rises sharply with persistent refusal.

---

## 9. Analysis Modules

Implement each as a standalone class in `obliteratus/analysis/`.

### 9.1 `activation_probing.py`

**Purpose**: Verify refusal signal is gone after abliteration.

```python
class ActivationProbe:
    def probe_layer(self, post_harmful, post_harmless, direction, threshold=0.1):
        gap = (post_harmful @ direction).mean() - (post_harmless @ direction).mean()
        flagged = abs(gap) > threshold
        return LayerProbeResult(...)
```

Compute **Refusal Elimination Score (RES)** as a weighted combination of d-prime reduction, layer coverage, and exponential gap penalty.

### 9.2 `alignment_imprint.py`

**Purpose**: Predict alignment training method from geometric features.

Extract features:

- Gini coefficient of layer refusal strengths.
- Effective rank of stacked direction matrix.
- Cross-layer smoothness (mean adjacent cosine).
- Tail-layer bias (fraction of strength in last 25% layers).
- Mean pairwise orthogonality.
- Spectral decay rate.

Score each method with Gaussian-kernel matching against literature-derived signatures, then softmax-normalize.

### 9.3 `concept_geometry.py`

**Purpose**: Determine if refusal is linear or polyhedral.

1. Group harmful prompts by category (e.g. violence, cybercrime, hate, etc.).
2. Compute per-category diff-of-means directions.
3. Compute pairwise cosines and Direction Specificity Index: `1 - mean(|cos|)`.
4. Estimate cone dimensionality = effective rank of category-direction matrix.
5. Classify: linear if mean cosine > 0.9 and dim < 1.5; polyhedral if mean cosine < 0.8 or dim > 2.0.

### 9.4 `cross_layer.py`

**Purpose**: Track direction persistence across layers.

1. Stack normalized directions.
2. Compute pairwise absolute cosine matrix.
3. Compute mean adjacent cosine and cumulative angular drift.
4. Find clusters via single-linkage thresholding (cos ≥ 0.85) using BFS connected components.

### 9.5 `defense_robustness.py`

**Purpose**: Estimate self-repair capacity and entanglement.

- Refusal strength distribution per layer.
- Self-repair estimate = `spread / (0.5 * n_layers)` clamped to [0,1].
- Entanglement per layer = `sqrt(var(harmless_proj) * abs(mean(harmless_proj)))`.

### 9.6 `sparse_surgery.py`

**Purpose**: Modify only the weight rows most responsible for refusal.

```python
def analyze_weight_matrix(W, direction):
    projections = (W @ direction).abs()  # (out,)
    # Knee detection on sorted projections
    k = _find_knee(projections)
    return SparseProjectionResult(top_k=k, rsi=gini(projections), ...)

def apply_sparse_projection(W, direction, sparsity):
    projections = W @ direction
    _, top_k_indices = projections.abs().topk(int(sparsity * W.size(0)))
    W[top_k_indices] -= projections[top_k_indices][:, None] * direction[None, :]
```

### 9.7 `steering_vectors.py`

**Purpose**: Inference-time activation steering without weight modification.

```python
@dataclass
class SteeringConfig:
    vectors: list[SteeringVector]
    target_layers: list[int]
    position: str = "all"  # "all" | "last" | "first"

class SteeringHookManager:
    def install(self, model, config):
        for layer_idx, vec in zip(config.target_layers, config.vectors):
            hook = self._make_hook(vec, config.position)
            self._handles.append(layers[layer_idx].register_forward_hook(hook))

    def remove(self):
        for h in self._handles:
            h.remove()
```

### 9.8 `whitened_svd.py`

See section 7.6.3.

### 9.9 Other Modules

Implement the remaining analysis modules with the algorithms described in the exploration summary:

- `causal_tracing.py`: noise-based causal importance simulation.
- `residual_stream.py`: attribute refusal to attention vs MLP.
- `probing_classifiers.py`: logistic regression probes with AUROC.
- `cross_model_transfer.py`: transfer of directions across models.
- `sae_abliteration.py`: train SAEs and identify refusal features.
- `multi_token_position.py`: per-position refusal signal.
- `anti_ouroboros.py`: build Adversarial Self-Repair Graph.
- `logit_lens.py`: decode promoted/suppressed tokens.

---

## 10. Evaluation

### 10.1 `evaluation/metrics.py`

Implement basic metrics:

```python
def perplexity(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return math.exp(loss.item())
```

### 10.2 `evaluation/advanced_metrics.py`

Implement:

- `refusal_rate(responses, mode="combined")`: keyword prefix/substring + semantic regex matching.
- `token_kl_divergence(logits_orig, logits_mod)`.
- `first_token_kl_divergence(...)`.
- `effective_rank(weight_matrix)`: `exp(entropy of normalized singular values)`.
- `linear_cka(X, Y)`: centered kernel alignment.
- `refusal_projection_magnitude(activations, direction)`.
- `AbliterationEvalResult` dataclass.

### 10.3 `evaluation/heretic_eval.py`

Implement the community-standard protocol:

- Load JailbreakBench behaviors.
- Generate completions.
- Compute Arditi refusal rate (29 substrings).
- Compute HarmBench ASR using `cais/HarmBench-Llama-2-13b-cls`.
- Compute first-token KL on harmless prompts.
- Run `lm-eval` on MMLU, GSM8K, ARC, HellaSwag, TruthfulQA.

### 10.4 `evaluation/benchmarks.py`

Lightweight internal benchmarks:

- `run_knowledge_probe`: 24 MMLU-style MCQs.
- `run_truthfulness_probe`: 15 TruthfulQA-style items.
- `run_math_reasoning_probe`: 12 GSM8K-style items.

### 10.5 `evaluation/evaluator.py`

```python
class Evaluator:
    def __init__(self, handle, dataset, metrics, batch_size, max_length, ...):
        ...

    def evaluate(self) -> dict[str, float]:
        # Run metrics requested by YAML study.
        ...
```

---

## 11. YAML Study Runner

`obliteratus/runner.py` implements `run_study(config: StudyConfig)`:

1. Load model.
2. Load dataset via `datasets.load_dataset`.
3. Compute baseline metrics.
4. For each strategy in `config.strategies`:
   - Enumerate specs.
   - For each spec: apply, evaluate, record, restore.
5. Save `results.json`, `results.csv`, optional plots.

`obliteratus/config.py` defines dataclasses:

- `ModelConfig`, `DatasetConfig`, `StrategyConfig`, `RemoteConfig`, `StudyConfig`.
- `StudyConfig.from_yaml(path)` parses YAML and applies presets via `study_presets.py`.

---

## 12. CLI

`obliteratus/cli.py` is the `obliteratus` console entry point.

Commands to implement:

- `run <yaml>`: load `StudyConfig`, call `run_study` or `RemoteRunner`.
- `info <model>`: print architecture summary.
- `interactive`: launch wizard from `interactive.py`.
- `models [--tier]`: print curated models table.
- `presets`: print study presets.
- `strategies`: list registered strategies.
- `ui`: launch local Gradio UI from `local_ui.py`.
- `obliterate <model>`: run `AbliterationPipeline`.
- `abliterate <model>`: backward-compatible alias.
- `self-improve <model> --audit ...`: recursive hard-negative mining.
- `report`: regenerate report from `results.json`.
- `aggregate`: aggregate community contributions.
- `tourney`: run method tournament.
- `recommend`: architecture + telemetry recommendation.
- `gpu-calc`: estimate GPU count from parameter count.

Add `--gpus` and `--remote` argument groups. GPU selection sets `CUDA_VISIBLE_DEVICES`. Remote execution delegates to `RemoteRunner`.

---

## 13. Gradio UI

`app.py` builds a multi-tab `gr.Blocks` interface.

Tabs:

1. **Obliterate**: model, method, dataset, advanced settings, run button, output metrics.
2. **Benchmark**: multi-method and multi-model benchmark runners with plots.
3. **Chat**: chat with the loaded/obliterated model.
4. **A/B Compare**: side-by-side original vs. obliterated.
5. **Strength Sweep**: sweep regularization from 0 to 1.
6. **Tourney**: elimination tournament across methods.
7. **Export**: ZIP artifacts.
8. **Push to Hub**: upload to HuggingFace.
9. **Leaderboard**: telemetry leaderboard.
10. **About**: documentation.

Use a global `_state` dict to share model handles across tabs. Cache obliterated/benchmark/tourney models in `_session_models`.

`obliteratus/local_ui.py` wraps `app.py` with Rich console startup, GPU detection, and browser auto-open.

---

## 14. Telemetry and Community

### 14.1 Telemetry

`obliteratus/telemetry.py`:

- Enabled by default on HF Spaces; disabled locally.
- Append benchmark records to a local JSONL file.
- Debounced background sync to a HuggingFace Dataset repo.
- `build_report()`: create schema-v2 report with model, method, metrics, hardware, config fingerprint.

### 14.2 Community

`obliteratus/community.py`:

- `save_contribution()`: write a JSON record to `community_results/` when `--contribute` is passed.
- `load_contributions()`: read all JSON files.
- `aggregate_results()`: group by `(model_name, method)` and compute mean/std/min/max.
- `generate_latex_table()`: render aggregated results.

---

## 15. Testing

Create `tests/` with pytest.

Key test files:

- `test_abliterate.py`: test `AbliterationPipeline` on tiny model (e.g. `gpt2`).
- `test_informed_pipeline.py`: test `InformedAbliterationPipeline`.
- `test_strategies.py`: test all four ablation strategies.
- `test_analysis.py`: test analysis modules.
- `test_metrics.py`: test evaluation metrics.
- `test_cli.py`: test CLI commands with small fixtures.
- `test_config.py`: test YAML parsing.
- `test_model_profile.py`, `test_architecture_profiles.py`, etc.

Use tiny models and mock activations to keep tests fast.

---

## 16. Implementation Order

Recommended order for an LLM implementing from scratch:

1. **Skeleton**: `pyproject.toml`, `__init__.py`, `__main__.py`.
2. **Device utilities**: `device.py`.
3. **Model loader**: `models/loader.py` and `ModelHandle`.
4. **Configuration**: `config.py`, `study_presets.py`.
5. **Strategies**: `base.py`, `registry.py`, `utils.py`, then the four strategies.
6. **Prompts**: `prompts.py` with built-in pairs and dataset loaders.
7. **Core metrics**: `evaluation/metrics.py`, `evaluation/advanced_metrics.py`.
8. **Abliteration pipeline**: `abliterate.py` starting with `basic` and `advanced` methods, then add `aggressive`, `informed`, etc.
9. **Analysis modules**: start with `activation_probing`, `cross_layer`, `concept_geometry`, `defense_robustness`, `alignment_imprint`, then the rest.
10. **Informed pipeline**: `informed_pipeline.py`.
11. **YAML runner**: `runner.py` + `evaluation/evaluator.py`.
12. **CLI**: `cli.py`.
13. **UI**: `local_ui.py`, then `app.py`.
14. **Telemetry / community**: `telemetry.py`, `community.py`.
15. **Tests**: write tests in parallel with implementation.

---

## 17. Critical Implementation Details

### 17.1 Numerical Stability

- Cast activations to `float32` before SVD/covariance/whitening.
- Add `eps=1e-6` to denominators and eigenvalue truncation.
- Clamp eigenvalues to avoid division by zero in whitening.

### 17.2 Memory Management

- Snapshot to CPU, not GPU.
- Use `torch.no_grad()` for all inference and projection.
- Empty CUDA cache after loading, after probing, after excision.
- For multi-GPU, use `device_map="auto"` and set `offload_folder`.

### 17.3 Chat Template Compatibility

- Always check `tokenizer.chat_template` is not None.
- Suppress reasoning tags (`<think>`) for Qwen/DeepSeek-R1 distills.
- Set `padding_side="left"` for batched generation.

### 17.4 Quantization

- BitsAndBytes is CUDA-only.
- Dequantize before projection; either re-quantize or save as float weights.

### 17.5 Safety Defaults

- Telemetry is **opt-in** locally; **opt-out** on HF Spaces.
- Community contributions require explicit `--contribute`.
- Add clear warnings that modified models have removed safety guardrails.

---

## 18. References

The implementation draws on these published works:

- Arditi et al. (2024): single refusal direction.
- Gabliteration (arXiv:2512.18901): multi-direction SVD.
- grimjim (2025): norm-preserving biprojection.
- Turner et al. (2023): steering vectors.
- Rimsky et al. (2024): contrastive activation addition.
- Wollschlager et al. (ICML 2025): RDO / geometry of refusal.
- Piras et al. (AAAI 2026): SOM directions.
- Zou et al. (2024): HarmBench.

---

*End of implementation guide.*
