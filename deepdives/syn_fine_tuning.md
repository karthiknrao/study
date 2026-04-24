# Synthetic Data for Fine-Tuning LLMs — Deep Dive

## Why Synthetic Data?

- **Cost**: Human annotation is expensive ($0.10–$1.00+ per sample for complex tasks). Synthetic data generation via frontier models costs a fraction.
- **Scale**: Generate millions of examples on demand.
- **Control**: Precisely specify distribution, difficulty, and diversity of examples.
- **Privacy**: No PII or licensing concerns with generated data.
- **Bootstrapping**: Cold-start problems (no labeled data) become solvable.

## Core Approaches

### 1. Prompt-Based Generation
Use a strong model (GPT-4, Claude) to generate `(input, output)` pairs from a schema or template.

```
System: You are an expert SQL writer.
User: Generate a natural language question and the corresponding SQL query
for a table schema with columns: orders(id, customer_id, total, date)
```

**When to use**: Well-defined task domains, structured outputs.

### 2. Self-Instruct / Evol-Instruct
Start with a seed set of ~175 human-written tasks, then use the LLM to:
- Generate new instructions similar to the seeds
- Filter out low-quality or near-duplicate examples
- Evolve instructions to increase complexity

This is the backbone of Alpaca, WizardLM, and similar projects.

### 3. Constitutional AI / Critique-and-Revise
Generate data, then have the model critique its own output and revise:

```
Step 1: Generate initial response
Step 2: Critique the response against principles
Step 3: Revise based on critique
```

Produces higher-quality training data at the cost of more inference compute.

### 4. Distillation
Use a strong teacher model's outputs to train a smaller student model. Key variants:
- **Logit distillation**: Match the full output distribution
- **Response distillation**: Match only the final text (most common for synthetic data)
- **Preference distillation**: Generate pairs, rank them, use for DPO/RLHF

### 5. Trajectory / Reasoning Synthesis
For chain-of-thought fine-tuning:
- Generate multiple reasoning paths for a problem
- Verify correctness (execution, formal verification, unit tests)
- Keep only verified correct trajectories

Used extensively in math (MetaMath, MathCoder) and code (CodeAlpaca, WizardCoder).

## Quality Control Pipeline

This is the hardest and most important part:

```
Raw Generation → Filtering → Deduplication → Verification → Curation
```

| Stage | Techniques |
|-------|-----------|
| **Filtering** | Length heuristics, perplexity scoring, classifier-based quality filters, regex patterns |
| **Deduplication** | MinHash, exact match, embedding similarity (cosine > 0.9 → drop) |
| **Verification** | Unit tests (code), execution checkers (SQL), formal verifiers (math), LLM-as-judge |
| **Curation** | Diversity maximization via clustering, difficulty stratification, rejection sampling |

## Avoiding Model Collapse

The biggest risk: training on model-generated data causes degradation over generations.

**Symptoms**:
- Loss of tail distribution (rare patterns vanish)
- Repetition loops increase
- Diversity drops with each generation

**Mitigations**:
- Mix synthetic data with real human data (never train on 100% synthetic)
- Use a *different* (stronger) model to generate than the one you're fine-tuning
- Add controlled noise/perturbation
- Verify outputs against ground truth where possible
- Monitor diversity metrics (n-gram entropy, distinct-n, embedding spread)

## Fine-Tuning Methods with Synthetic Data

| Method | Data Format | Best For |
|--------|------------|----------|
| **SFT** (Supervised FT) | `(input, output)` pairs | Teaching new tasks, style transfer |
| **DPO** | `(input, chosen, rejected)` pairs | Alignment, reducing hallucination |
| **RLHF** | Reward model + PPO | Complex preference alignment |
| **ORPO / KTO** | Simpler preference variants | When full RLHF is overkill |

### SFT Pipeline Example

```python
# 1. Generate synthetic data
prompts = load_seed_prompts("seeds.jsonl")
synthetic_data = []

for prompt in prompts:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}]
    )
    synthetic_data.append({
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response.content[0].text}
        ]
    })

# 2. Filter
filtered = [d for d in synthetic_data if quality_filter(d)]

# 3. Deduplicate
deduped = deduplicate_by_embedding(filtered, threshold=0.92)

# 4. Format for training
with jsonlines.open("train.jsonl", "w") as f:
    for item in deduped:
        f.write(item)
```

### DPO Pipeline (Preference Data)

```python
# Generate chosen/rejected pairs
for prompt in prompts:
    # Generate 4-8 candidates
    candidates = [generate(prompt) for _ in range(4)]
    # Rank them (LLM-as-judge or reward model)
    ranked = rank_by_quality(prompt, candidates)
    dpo_data.append({
        "prompt": prompt,
        "chosen": ranked[0],   # best response
        "rejected": ranked[-1] # worst response
    })
```

## Practical Recipes by Domain

### Code Generation
- Generate problem descriptions → solutions → test cases
- Run test cases to verify; discard failing pairs
- Vary difficulty: trivial → easy → medium → hard → competition
- Include edge cases explicitly in generation prompts

### Math / Reasoning
- Start with problems that have verifiable answers
- Generate multiple solution paths per problem
- Keep only paths that reach the correct answer
- Use rejection sampling at scale

### Domain Adaptation (Medical, Legal, etc.)
- Generate from domain corpora (textbooks, papers)
- Use domain experts to validate a subset (compute agreement)
- Use validated subset as few-shot examples for more generation

### Tool Use / Function Calling
- Define tool schemas
- Generate realistic user queries requiring tool use
- Execute tools, capture results, generate follow-up responses
- This creates multi-turn tool-use trajectories

## Key Metrics to Track

- **Data quality**: LLM-as-judge scores, human eval on samples
- **Diversity**: Distinct-n, embedding entropy, cluster count
- **Difficulty**: Distribution of problem complexity
- **Contamination**: Overlap with common eval benchmarks (leakage = invalid results)
- **Model performance**: Hold-out eval set (must be human-verified, not synthetic)

## Tools & Frameworks

| Tool | Purpose |
|------|---------|
| **Magpie** | Generate alignment data from open-source LLMs |
| **DataDreamer** | End-to-end synthetic data generation pipeline |
| **DistilLabel** | Label data using LLMs with quality controls |
| **Cleanlab** | Automated data quality scoring |
| **Instructor** (Python) | Structured output extraction from LLMs |

## Common Pitfalls

1. **Eval contamination**: Your synthetic data accidentally contains benchmark questions → inflated scores
2. **Bias amplification**: Generator biases get baked into the fine-tuned model
3. **Format overfitting**: Model learns the output format but not the underlying task
4. **Homogenization**: All outputs converge to a narrow style
5. **Overfitting synthetic distribution**: Model performs well on synthetic-like inputs but poorly on real-world distribution shift

## Recommended Reading

- **Self-Instruct** (Wang et al., 2023) — bootstrapping instruction data
- **Alpaca** (Taori et al., 2023) — practical SFT from synthetic data
- **WizardLM / Evol-Instruct** (Xu et al., 2023) — complexity evolution
- **LIMA** (Zhou et al., 2023) — 1000 good examples > 100K mediocre ones
- **Phi series** (Microsoft) — "textbooks are all you need" approach
- **Direct Preference Optimization** (Rafailov et al., 2023) — simpler alignment
