I'll dive deep into mechanistic interpretability research for you. This is a meaty topic — let me use the deep-research skill to do this properly with proper source verification.

</think>

</think>

</think>

</think>

</think>

</think>

Step 1 background is solid — I have a thick factual base on definitions, key papers, code, applications, industry players, and open debates. Moving to Step 2.

Step 2 direction judgment complete. Now reading Step 3.

# Step 3 Analysis — Mechanistic Interpretability Deep Dive

## 1. Deep Sub-topic Breakdown

The user's request decomposes into 5 primary sub-topics. Each is substantive enough to need dedicated research.

### Sub-topic A: Field foundations & core primitives
- **What to research:** The intellectual genealogy of MI — from Distill.pub's Circuits Thread (Olah et al. 2020) to Anthropic's mathematical framework (Elhage et al. 2021); the central primitives (features, circuits, polysemanticity, superposition); the formal vs. intuitive definitions.
- **Keywords:** "mechanistic interpretability foundations", "transformer circuits thread", "Chris Olah circuits", "A Mathematical Framework for Transformer Circuits Elhage 2021", "Olah 2020 zoom in", "polysemanticity neuron", "superposition hypothesis Elhage 2022".
- **Why it matters:** Without grounding in primitives, the rest of the literature is unreadable. The user wants depth — they need the vocabulary.
- **Connections:** Feeds Sub-topics B (techniques), C (applications), D (code).

### Sub-topic B: Methods/techniques landscape (2023-2026)
- **What to research:** SAEs (L1 → TopK → JumpReLU → Matryoshka → BatchTopK), transcoders, cross-layer transcoders, attribution graphs, activation patching, path patching, automated circuit discovery (ACDC, EAP, MechRL), probing, dictionary learning at scale, mechanistic anomaly detection.
- **Keywords:** "sparse autoencoder L1", "TopK SAE Gao 2024", "JumpReLU SAE", "transcoder mechanistic interpretability", "cross-layer transcoder Anthropic", "attribution graph", "activation patching", "automated circuit discovery ACDC", "edge attribution patching EAP", "circuit discovery reinforcement learning", "sparse dictionary learning scaling".
- **Why it matters:** This is the heart of the field's recent progress — every six months a new variant reshapes what's tractable.
- **Connections:** Underpins applications in D and is implemented by tools in E.

### Sub-topic C: Landmark circuits discovered
- **What to research:** Induction heads (Olsson et al. 2022), IOI circuit (Wang et al. 2023), greater-than circuit, Indirect Object Identification, refusal circuit (Arditi et al. 2024), multilingual circuits (Biology of an LLM 2025), planning circuits (DRC agent), rhyme-planning circuits.
- **Keywords:** "induction heads Olsson 2022", "IOI circuit Wang 2023 GPT-2", "refusal direction Arditi 2024", "Anthropic Biology of LLM", "multilingual circuit", "MechRL circuit discovery PPO", "greater-than circuit".
- **Why it matters:** These are the field's empirical proofs-of-concept — concrete cases where MI yielded a falsifiable, useful account of behavior.
- **Connections:** Bridges methods (B) → applications (D).

### Sub-topic D: Applications
- **What to research:** Safety (jailbreak detection via mechanistic analysis, refusal-circuit erasure, deception probes, ACH/SAH head taxonomy, alignment-faking detection); steering ("Golden Gate Claude", persona steering); training-time use (Concept Influence, learning-rate steering); biology/drug discovery (SAEs on ESM2 protein LM, Reticular, Markov Bio, PNAS 2025 paper); vision (Prisma toolkit, SAE-ViT, BLIP causal tracing); RL agents (planning circuits in Sokoban); knowledge retrieval.
- **Keywords:** "mechanistic interpretability safety", "refusal direction ablation", "jailbreak circuit detection", "sparse autoencoder protein language model", "Reticular interpretability drug", "Markov Bio mechanistic biology", "Prisma vision mechanistic interpretability", "safety neurons LLM", "ROSI rank one safety injection", "Concept Influence FAR.AI", "deception probe FAR.AI".
- **Why it matters:** User explicitly asked for applications. This separates MI from interpretability theater.
- **Connections:** Validates B and C with real-world payoff.

### Sub-topic E: Tools, code, infrastructure
- **What to research:** TransformerLens, SAELens, nnsight/NDIF, pyvene, CircuitsVis, SAE-Vis, circuit-tracer (Anthropic open-sourced 2025), Neuronpedia (web platform), Prisma (vision), Goodfire Ember API, Hugging Face SAEs, model-organism repositories.
- **Keywords:** "TransformerLens library Neel Nanda", "SAELens Bloom", "nnsight interpretability", "circuit-tracer Anthropic GitHub", "Neuronpedia interpretability platform", "Goodfire Ember API", "Prisma ViT interpretability", "HuggingFace sparse autoencoder", "OpenAI sparse autoencoder code".
- **Why it matters:** User asked for "code" — they need actionable repos and the libraries they should learn.
- **Connections:** Lets the reader actually do MI, not just read about it.

### Sub-topic F: Surveys, benchmarks, open problems, critique
- **What to research:** "Open Problems in MI" (Sharkey et al. Jan 2025, ~30 authors), "A Practical Review of MI for Transformer LMs" (Rai et al. 2024), "MI for AI Safety — A Review" (Bereska & Gavves 2024), MIB Benchmark (April 2025), "MI Needs Philosophy" (Williams et al. June 2025), "Is MI Identifiable?" (Méloux et al. ICLR 2026), "Everything Everywhere All at Once" non-identifiability critique, mechaninterpworkshop at ICML 2026.
- **Keywords:** "Open Problems Mechanistic Interpretability Sharkey", "MIB benchmark mechanistic interpretability", "MI needs philosophy arxiv", "ICLR 2026 mechanistic interpretability identifiable", "ICML 2026 mechanistic interpretability workshop".
- **Why it matters:** Lets the user see what the community itself thinks is unresolved — they want to enter the field with eyes open.
- **Connections:** Frames all of A-E in honest context.

## 2. Precise Concept Disambiguation

These confusions will derail the final answer if not addressed up front.

- **Mechanistic interpretability (MI)** ≠ general "interpretability/XAI." MI is specifically reverse-engineering networks into human-understandable algorithms (Olah, Elhage, Sharkey). XAI includes SHAP/LIME/saliency which provide post-hoc feature attributions without claiming mechanistic understanding. MI is *not* a subset of XAI — they have different epistemic goals.
- **Mechanistic interpretability ≠ mechanistic biology.** There is a parallel literature applying MI-style techniques to biological foundation models (AlphaFold, ESM2) — but the term "mechanistic interpretability" itself refers to reverse-engineering *neural networks*, not biological mechanisms. Sources like Markov.bio's "Through a Glass Darkly" are explicit about the bridge but distinct domains.
- **Feature ≠ neuron.** A "feature" is an interpretable direction in activation space (often an SAE latent). A neuron is a single dimension of an activation vector. Polysemanticity = one neuron fires on multiple features. A feature may be *distributed* across many neurons.
- **Polysemanticity vs. superposition.** Polysemanticity is the observed phenomenon (neurons firing on multiple concepts). Superposition is a *hypothesis* explaining polysemanticity — that networks linearly represent more features than dimensions, in near-orthogonal directions. Polysemanticity could have other causes; superposition is the dominant theory.
- **SAE vs. transcoder vs. cross-layer transcoder (CLT).** SAEs reconstruct activations of a single site from a sparse code. Transcoders reconstruct the *output of a specific layer/MLP from its input*, which makes them better primitives for circuit analysis. Cross-layer transcoders have latents across L layers and feed into all subsequent CLTs — they underpin Anthropic's attribution graphs.
- **Attribution graph vs. circuit.** A circuit is a hypothesized subgraph. An attribution graph is the empirical artifact produced by perturbation/linear-attribution that *suggests* which features matter and how. They overlap heavily in practice but the graph is the data; the circuit is the interpretation.
- **Sparse Autoencoder variants — L1 vs. TopK vs. JumpReLU.** L1 SAE penalizes L1 norm of latents (Bricken 2023, Cunningham 2023). TopK SAE activates exactly the K highest pre-activations — no L1 penalty needed (Gao et al. 2024). JumpReLU uses a learnable threshold (RAVEL/DeepMind). All three are solving the same problem with different inductive biases.
- **Activation patching vs. path patching vs. causal scrubbing.** Activation patching swaps a single activation between clean/corrupted runs. Path patching does it for an edge. Causal scrubbing is a stricter hypothesis test. The three define a hierarchy of causal strictness (per Bereska review).
- **Mechanistic interpretability (academic) vs. MechInterp (community).** A lot of the cutting-edge work is blog-posted on Alignment Forum / LessWrong rather than peer-reviewed. This is itself a debated phenomenon.

## 3. Rough Scope

### In scope
- Definition, history, and intellectual foundations of MI for transformer LMs
- Core primitives: features, circuits, polysemanticity, superposition, residual stream, QK/OV circuits, K-composition, virtual heads
- Sparse autoencoders (L1/TopK/JumpReLU/Matryoshka) — variants, scaling, evaluation
- Transcoders and cross-layer transcoders
- Attribution graphs and circuit tracing (Anthropic 2025)
- Activation/path patching, automated circuit discovery
- Landmark circuits: induction heads, IOI, refusal direction, multilingual circuits, planning circuits
- Code libraries: TransformerLens, SAELens, nnsight, pyvene, CircuitsVis, SAE-Vis, circuit-tracer, Neuronpedia, Prisma, Goodfire Ember
- Applications: safety/jailbreaks/refusal/deception; steering; vision; protein LM; RL agents; training-data attribution
- Surveys, benchmarks, open problems, philosophy, identifiability critique
- Industry players: Anthropic, OpenAI, DeepMind, Goodfire, Neuronpedia, FAR.AI

### Out of scope
- General explainability methods unrelated to MI (SHAP, LIME, saliency maps) — except as comparative context
- Pre-transformer interpretability (e.g., vision CNN feature visualization)
- Mechanistic interpretability of non-neural systems (the "mechanism" in cognitive science / philosophy of science)
- Pure RL theory without mechanistic analysis
- Commercial deployment of interpretability products (price, ROI) — note briefly but don't deep-dive

### Borderline areas
- **Knowledge editing / model editing** (ROME, MEMIT) — overlaps but uses different causal lens; include briefly as it shares goals with steering
- **Representation engineering (RepE)** — uses directions in activation space to steer; bridges MI and applied steering. Include briefly.
- **Activation steering as a technique** — yes, in scope (it's a key MI application); distinguish from generic "prompt engineering"
- **Interpretability of diffusion models / image generators** — mostly out of scope unless tied to circuit analysis

## 4. Question Type and Capabilities Needed

| Capability | Importance | How to apply |
|---|---|---|
| Framework building | HIGH | Use the three-bucket taxonomy (features / circuits / universality) as the spine. Add a fourth axis (causal intervention methods) for completeness. |
| Critical annotation | HIGH | Each key paper needs a one-line "why it matters" — not just a citation. The user wants signal, not a bibliography. |
| Comparative analysis | HIGH | SAE variants table; code library comparison; application-domain comparison. |
| Causal reasoning | MEDIUM | Trace how the field evolved: neuron-level → polysemanticity → SAEs → scaling SAEs → attribution graphs. |
| Non-obvious insight | HIGH | Surface the open debates: non-identifiability (Méloux 2025), interpretability-vs-intentional-design tension at Goodfire, the philosophy gap, "interpretability illusions." |
| Historical analogy | LOW | Optional analogy: MI as reverse-engineering a binary, or as neuroscience for AI. Mention briefly. |
| Data-driven argumentation | MEDIUM | Cite specific numbers (34M features, 16M, 30M, 90% interpretability, $1.25B Goodfire valuation, etc.) to anchor claims. |
| Scenario forking | LOW | Not needed — the question is not "what will happen." |

## 5. Key Facts and Verification Checklist

Items to verify in Step 4 (some may have been mis-typed in sources):

1. **Anthropic 34M features from Claude 3 Sonnet.** Likely true, multiple corroborating sources. Confirm exact number and date (May 2024).
2. **OpenAI 16M features from GPT-4.** Multiple sources confirm. Confirm date (June 2024) and methodology (k-sparse architecture, 40B tokens).
3. **Gemma Scope 30M features / 400+ SAEs.** Multiple sources confirm. Confirm whether main release or full set (2,000 SAEs accounting for sparsity variants).
4. **Gemma Scope 2 uses transcoders and Matryoshka training.** Longterm Wiki and DeepMind page confirm. Cross-check.
5. **IOI circuit = 26 attention heads, 7 classes.** Wang et al. 2023 — well-corroborated.
6. **Induction heads form during training "induction bump."** Olsson et al. 2022 — well-corroborated.
7. **Refusal is a 1D direction in many models (Arditi et al. 2024).** Multiple downstream papers (ROSI) cite this.
8. **Goodfire Series B: $150M Feb 2026, $1.25B valuation, total $207M raised.** Longterm Wiki + AppliedMethods + StartupIntros agree. Verify Anthropic was its first corporate investor.
9. **FAR.AI work on deception probes / Obfuscation Atlas (Feb 2026) and Concept Influence (Feb 2026).** Verify the dates and exact authors.
10. **Anthropic 2025 attribution graph papers — Claude 3.5 Haiku.** Confirm the model studied.
11. **ICLR 2026 paper "Is MI Identifiable?" — confirm authors and exact claim.** ICLR poster page confirms.
12. **Open Problems in MI (Sharkey et al. arXiv:2501.16496) — submitted Jan 2025, ~30 authors.** Multiple sources agree.
13. **MIB Benchmark (arXiv:2504.13151, April 2025).** Confirm scope.
14. **MI Needs Philosophy (arXiv:2506.18852, June 2025).** Multiple sources agree.
15. **Weight-sparse transformers (arXiv:2511.13653, Nov 2025) by Gao et al.** Verify date.

These are mostly well-corroborated; main risk is numbers/dates drifting. Step 4 should sample-verify the most-cited ones.

## 6. User Profile

- **Background:** Technical practitioner familiar with deep learning. Uses the precise term "mechanistic interpretability" twice (with a typo). Asks specifically for "papers / code and the applications" — the slash structure suggests they want all three buckets cleanly. Likely an ML researcher, grad student, or engineer exploring the field.
- **Knowledge level:** Intermediate-to-advanced. Not a complete beginner (they wouldn't ask for a "dive deep" otherwise) but probably hasn't read the full primary literature yet. They want orientation plus substance.
- **Motivation:** Most likely (a) evaluating whether to enter MI research, (b) wanting a research briefing to apply MI in their own work, or (c) preparing to teach / write about it. The "papers / code / applications" framing tells me they want the full landscape, not a narrow answer.
- **Influence on writing:** Skip elementary explanations of transformers; assume they know attention/MLP/residual stream. But still define MI-specific primitives (features vs neurons, superposition, SAE) briefly. Heavy on tables and concrete references. End with a "where to start" reading path.

## 7. Research Direction and Search Strategy

**Overall direction:** The background search already covered most of the territory. Step 4 needs to (a) verify the highest-leverage facts (numbers, dates, attributions), (b) drill into the two- or three-year-old primary papers to extract substantive content for citation, and (c) check for anything missed: code-repos I haven't inspected, applications I haven't seen, recent 2026 papers.

**Specific searches Step 4 should run:**

Priority A — verify and deepen:
- "Arditi refusal direction 2024 paper" — primary source
- "Towards Monosemanticity Bricken Cunningham 2023" — to get exact contributions
- "Conmy ACDC automated circuit discovery" — methodology details
- "OpenAI Scaling sparse autoencoder paper 16 million features" — methodology details
- "Anthropic On the Biology of a Large Language Model full paper" — substantive content
- "circuit-tracer Anthropic GitHub" — confirm release details and what models are supported

Priority B — fill gaps:
- "MechRL reinforcement learning circuit discovery 2025" — recent novelty
- "Subspace-Aware Sparse Autoencoders 2026" — very recent SAE variants
- "Verifiable Transformers circuit explanations 2026" — formal verification angle
- "From Circuit Evidence to Mechanistic Theory Inductive Logic 2026" — theoretical foundation
- "Universal Sparse Autoencoders cross-model interpretability" — comparison
- "Universal SAEs Neuronpedia FAR.AI" — cross-model concepts
- "weight-sparse transformers Gao 2025" — interpretable-by-design
- "Concept Influence FAR.AI training data" — application
- "Obfuscation Atlas FAR.AI deception" — application
- "MIB Mechanistic Interpretability Benchmark" — eval standard
- "Anthropic interpretability team blog 2025 2026" — any post-2025 updates

Priority C — Chinese-language sources for cross-validation:
- Search Zhihu, CSDN, Tencent, Sohu for community understanding of MI — already have some background, can supplement.

**Progressive search path:** Already broad. Move to deep — open individual paper pages, get exact contributions and authors. For code, hit the GitHub README pages directly.

**Expected gaps:**
- Specific code metrics (stars, last commit, contributor count) — need to fetch GitHub pages.
- Exact methodology details for some recent 2026 papers — may need to fetch arxiv abstracts.
- The Goodfire / interpretability-for-training debate — only have one LessWrong piece; may need more.

## 8. Specific Writing Guidance

**Reader:** Technical ML practitioner. Length: ~5,000-7,000 words is appropriate for a "deep dive" — long enough to be substantive, short enough to be readable.

**Required insights (must appear):**

1. MI is *reverse-engineering*, not *explaining* — this is the central distinction from XAI. Anthropic: "similar to how a programmer might try to reverse engineer complicated binaries into human-readable source code."
2. The single biggest technical unlock of the last three years has been **scaling sparse autoencoders** — going from 4K features on a toy model (2023) to 34M features on Claude 3 Sonnet (2024). This solved the polysemanticity bottleneck enough to enable feature-level work at frontier scale.
3. **Attribution graphs (Anthropic, March 2025)** moved the field from "what features exist" to "how features compose into computations" — this is the equivalent of going from a parts catalog to a wiring diagram.
4. The field's most-cited *practical* result is the **refusal direction** (Arditi et al. 2024) — a single linear direction in activation space that mediates refusal, with safety implications (can be amplified via ROSI, ablated to remove safety).
5. **SAEs are still imperfect** — feature splitting, absorption, composition, dead features at scale, reconstruction errors. The Open Problems paper (Sharkey et al. 2025) is the canonical honest summary.
6. The most important *critique* to surface is the **non-identifiability result** (Méloux et al. ICLR 2026): multiple circuits can explain the same behavior, meaning the interpretation you find may not be unique. This is a foundational challenge the field is now grappling with.
7. **Tools have caught up.** A reader can now: load a model with TransformerLens → run SAEs with SAELens → generate attribution graphs with circuit-tracer → publish results on Neuronpedia. This is no longer a one-lab exercise.
8. **Industry is moving fast.** Goodfire's $150M Series B in Feb 2026 at $1.25B valuation signals interpretability is becoming a commercial product category, not just a research subfield.
9. MI is being applied *outside* LLMs — vision (Prisma), protein models (Reticular, PNAS 2025), RL agents (planning in Sokoban DRC). The toolkit generalizes.
10. The biggest open question: does MI actually scale to detect *deception*, or will capable models learn to evade it? (Williams et al. "MI Needs Philosophy"; Alignment Forum debate). This is unresolved.

**Common shallow-answer patterns to avoid:**
- Listing papers without saying what they did
- Treating MI as "explaining the black box" — wrong; it's reverse-engineering
- Ignoring open problems / pretending the field is settled
- Treating Anthropic as the only player — DeepMind, OpenAI, Goodfire, FAR.AI, academic groups (Princeton, MIT, Harvard, Hebrew U, Tel Aviv) all contribute
- Missing the code/toolchain side — the user explicitly asked
- Generic "applications in safety/healthcare/finance" hand-wave — be specific

**Core takeaway to keep:**
> MI has matured from a curiosity (2020-2022) into a working engineering discipline with production tools, frontier-scale empirical results, and a credible claim to be the most promising path to *enumerative* safety — but it remains pre-paradigmatic, with deep open problems around identifiability, scalability, and whether capable models can learn to evade it.

## Step 4 Handoff — Research Todo Plan

This is the explicit handoff for Step 4. Each item is a TODO. Step 4 will create a todo list from this.

**Item T1 — Foundations & primitives deep dive.** Read abstracts of Olah 2020, Elhage 2021, Toy Models of Superposition (Elhage 2022), Olsson 2022 induction heads. Extract: precise definition of MI, residual stream, virtual heads, K/V composition, polysemanticity, superposition. *Evidence needed:* short, accurate definitions with citation.

**Item T2 — SAE scaling history.** Cross-check the chronology: Bricken 2023 (Towards Monosemanticity), Cunningham 2023 (SAEs Find Highly Interpretable Features), Templeton 2024 (34M on Claude 3 Sonnet, May 2024), OpenAI 16M on GPT-4 (June 2024), Lieberum 2024 (Gemma Scope), He 2024 (Llama Scope), Rajamanoharan TopK. *Evidence needed:* exact feature counts, dates, architectures, evaluation metrics.

**Item T3 — SAE variants (TopK, JumpReLU, Matryoshka, transcoders, CLTs).** Read Gao 2024 (TopK), Rajamanoharan (JumpReLU/BatchTopK), Paulo 2025 (Transcoders Beat SAEs), Dunefsky 2024 (Transcoders Find Interpretable LLM Feature Circuits), Anthropic CLT description. *Evidence needed:* when each emerged, why better, what trade-offs.

**Item T4 — Landmark circuits.** IOI (Wang 2023, 26 heads, 7 classes), Induction heads (Olsson 2022), Refusal direction (Arditi 2024), Greater-than, multilingual (Biology of LLM 2025), rhyme-planning, planning in DRC agent. *Evidence needed:* how big is each, what does it do, where in the model.

**Item T5 — Attribution graphs / circuit tracing.** Anthropic March 2025 release — Circuit Tracing + Biology of an LLM. Open-sourced circuit-tracer (Anthropic GitHub). Neuronpedia graph explorer. *Evidence needed:* what they are, how generated, key empirical findings.

**Item T6 — Methods: patching, automated discovery.** Activation patching, path patching, ACDC (Conmy 2023), EAP, MechRL (RL-based discovery). *Evidence needed:* when each introduced, what it automates.

**Item T7 — Code libraries.** Confirm GitHub repos for: TransformerLens (TransformerLensOrg), SAELens (decoderesearch/SAELens, formerly jbloomAus), nnsight (ndif-team), pyvene, CircuitsVis, SAE-Vis, circuit-tracer (safety-research), Prisma (Prisma-Multimodal), OpenAI sparse_autoencoder. *Evidence needed:* repo URL, what it does, last update, star count if available.

**Item T8 — Surveys & open problems.** "A Practical Review" (Rai 2024), "MI for AI Safety — A Review" (Bereska 2024), "Open Problems in MI" (Sharkey Jan 2025), "MI Needs Philosophy" (Williams June 2025), MIB Benchmark (April 2025). *Evidence needed:* key takeaways, what each adds.

**Item T9 — Applications: alignment & safety.** Safety neurons (Wei 2024), Refusal direction (Arditi 2024), ROSI, Jailbreak detection via circuit ablation (Circuit Discovery Helps to Detect LLM Jailbreaking, ICML 2025), xJailbreak, Robust Harmful Features (ACH/SAH taxonomy), FAR.AI Concept Influence (Feb 2026), Obfuscation Atlas (Feb 2026), TamperBench (Feb 2026). *Evidence needed:* what each accomplishes, in what setting.

**Item T10 — Applications: biology / protein.** PNAS 2025 (Gujral et al., SAEs uncover biological features in ESM2), Reticular (steerable protein LM), Markov Bio (SAEs for virtual cell), ICML 2025 "From Mechanistic Interpretability to Mechanistic Biology". *Evidence needed:* what features found, what use-cases.

**Item T11 — Applications: vision.** Prisma toolkit (April 2025, 75+ vision models), SAE-ViT (arXiv:2502.06755), Visual Mechanistic Interpretability (KL-Minimal Soft-Constraint), BLIP causal tracing. *Evidence needed:* what's distinctive about vision MI.

**Item T12 — Applications: RL / agent planning.** DRC agent planning circuits (FAR.AI), emergent persuasion persona steering, mech-interp for persuasion detection. *Evidence needed:* what planning circuit was found.

**Item T13 — Industry players.** Anthropic Interpretability team, OpenAI Superalignment, Google DeepMind, Goodfire (Series B Feb 2026, Ember API), Neuronpedia (June 2026 update, HeadVis, NLA), FAR.AI, Apollo Research, Haize Labs. *Evidence needed:* each player's focus and signature products.

**Item T14 — Critiques & debates.** "Is MI Identifiable?" (Méloux ICLR 2026, non-identifiability), "MI Needs Philosophy" (Williams June 2025), Makelov interpretability-illusion, Bechtel approach, "Interpretability Will Not Reliably Find Deceptive AI" (Alignment Forum). *Evidence needed:* the substantive critiques.

**Item T15 — 2026 very recent papers.** Subspace-Aware SAEs (Dalili/Mahdavi, June 2026), Verifiable Transformers (Somani, May 2026), From Circuit Evidence to Mechanistic Theory (Aljaafari, May 2026), KL-Minimal Visual MI (Zhou et al. May 2026), ICML 2026 mechinterp workshop. *Evidence needed:* what new directions are emerging.

OK, that's the handoff. Moving to Step 4Step 3 analysis and research plan complete. Moving to Step 4 to execute the research.

</think>

</think>

</think>

</think>

</think>

</think>

</mm:think>

</think>

</mm:think>

I have comprehensive research now. Let me update todos and proceed to Step 5.

Now writing the final answer to the workspace.

Now running the Write-Time Audit per Step 5 protocol.

</think>

</think>

</think>

</think>

Here's the deep-dive briefing on mechanistic interpretability — papers, code, applications, and where the field actually stands as of mid-2026. It's a long read, but every section is there for a reason: the field has matured fast enough that a "papers / code / applications" answer needs real structure to be useful.

<deliver-assets>
<media src="commit-id-416575307944042" type="md" name="final_turn_001.md" />
</deliver-assets>

# Mechanistic Interpretability: Field State, Papers, Code, and Applications (2024–2026)

Mechanistic interpretability (MI) is the subfield of interpretability research that tries to reverse-engineer neural networks into human-understandable algorithms — analogous, as Anthropic's Chris Olah put it, to "a programmer trying to reverse engineer complicated binaries into human-readable source code" [1]. Where traditional explainability (SHAP, LIME, saliency) offers post-hoc feature attributions, MI aims at *mechanistic understanding*: which internal components implement which computations, and how they interact.

The field has matured from a small theoretical curiosity (2020–2022) into an active engineering discipline with production tools, frontier-scale empirical results, and a growing commercial sector. By 2026 it has its own workshops at major conferences (ICML, ICLR, ACL), a dedicated benchmark [2], a canonical "open problems" review signed by roughly thirty leading researchers [3], and an industry track — including a startup that closed a $150M Series B in February 2026 at a $1.25B valuation [4]. But it remains pre-paradigmatic: a 2026 ICLR paper showed empirically that mechanistic explanations are *non-identifiable* even in toy networks [5], which is the field's most-cited open challenge.

This briefing covers the field's conceptual foundations, the major papers (especially 2024–2026), the open-source code stack, the substantive applications, and the live debates — with a reading path at the end for someone wanting to enter the field.

## I. Core Primitives and Vocabulary

The vocabulary is short, but the concepts are precise.

- **Feature.** An interpretable direction in a model's activation space. Features are the field's basic units of analysis. Unlike a neuron, a feature can be distributed across many neurons or compressed into a single dimension.
- **Circuit.** A sparse subgraph of a model (attention heads, MLP layers, residual connections) that implements a specific behavior. The IOI circuit in GPT-2 small, for instance, uses 26 attention heads grouped into 7 functional classes to predict the indirect object in a sentence [6].
- **Polysemanticity.** The empirical phenomenon that a single neuron fires for multiple, semantically unrelated inputs. Documented as a major obstacle to neuron-level interpretation.
- **Superposition hypothesis.** The leading *explanation* for polysemanticity: networks represent more features than they have dimensions by placing features in near-orthogonal directions in a higher-dimensional space [7]. Polysemanticity and superposition are related but distinct concepts — superposition is one explanation among several.
- **Residual stream.** The shared high-dimensional bus that every transformer component reads from and writes to. Reframing the transformer as residual-stream-centric (rather than layer-centric) is the conceptual move that makes circuit analysis tractable [1].
- **K/V/Q composition and virtual heads.** Attention heads compose by passing Q/K/V matrices through earlier heads' OV circuits. This lets a "virtual" attention head in a later layer attend based on a complex function computed earlier — the mechanism behind induction heads and many other circuits.
- **Monosemanticity.** The property that a single feature corresponds to a single human-interpretable concept. The opposite of polysemanticity.

## II. Field Timeline (2020–2026)

| Year | Landmark | Source |
|------|----------|--------|
| 2020 | Olah et al. launch the Distill Circuits Thread for vision models | [8] |
| 2021 | Anthropic's "A Mathematical Framework for Transformer Circuits" defines residual-stream view, virtual heads, QK/OV circuits | [1] |
| 2022 | "Toy Models of Superposition" (Anthropic) — formalizes the superposition hypothesis | [7] |
| 2022 | "In-context Learning and Induction Heads" (Olsson et al.) — induction heads emerge with a sharp loss-curve kink | [9] |
| 2023 | "Towards Monosemanticity" (Bricken et al., Anthropic) — first SAE on a real transformer (4,096 features on GPT-2 layer 6) | [10] |
| 2023 | "Sparse Autoencoders Find Highly Interpretable Features" (Cunningham et al., OpenAI/Together) — independently proposes dictionary learning on activations | [11] |
| 2023 | "Interpretability in the Wild" / IOI circuit (Wang et al.) — first large end-to-end reverse-engineering of a behavior (26 attention heads, 7 classes) | [6] |
| 2023 | "Towards Automated Circuit Discovery" (Conmy et al.) — ACDC systematizes patching-based circuit discovery | [12] |
| 2024 (Jun) | OpenAI's "Scaling and Evaluating Sparse Autoencoders" — 16M features from GPT-4 with k-sparse architecture | [13] |
| 2024 (May) | Anthropic's "Scaling Monosemanticity" — 34M features from Claude 3 Sonnet; 90% interpretability on automated evaluation; safety-relevant features identified | [14] |
| 2024 (Jun) | Arditi et al. — "Refusal in Language Models Is Mediated by a Single Direction" — refusal is a 1D subspace across 13 chat models | [15] |
| 2024 (Jun) | Dunefsky et al. — transcoders as a circuit-analysis-friendly alternative to SAEs | [16] |
| 2024 (Aug) | Google DeepMind releases **Gemma Scope** — 400+ open JumpReLU SAEs, 30M+ features on Gemma 2 2B/9B | [17] |
| 2024 (Oct) | **Llama Scope** — 256 SAEs on every layer/sublayer of Llama-3.1-8B-Base | [18] |
| 2025 (Mar) | Anthropic's **Circuit Tracing** papers — attribution graphs on Claude 3.5 Haiku; "Biology of an LLM" | [19][20] |
| 2025 (May) | **circuit-tracer** open-sourced by Anthropic + Decode Research, with Neuronpedia frontend | [21] |
| 2025 (Apr) | MIB: Mechanistic Interpretability Benchmark (Mueller et al.) | [2] |
| 2025 (Apr) | Prisma toolkit for vision/video mechanistic interpretability (Joseph et al.) | [22] |
| 2025 (Jan) | "Open Problems in Mechanistic Interpretability" (Sharkey et al., ~30 authors) | [3] |
| 2025 (Jun) | "Mechanistic Interpretability Needs Philosophy" (Williams et al.) | [23] |
| 2025 (Jul) | Anthropic's mechanistic interpretability appears in production system cards (Sonnet 4.5) | [24] |
| 2025 (Nov) | "Weight-sparse transformers have interpretable circuits" (Gao et al., OpenAI) — design for interpretability by enforcing sparsity in weights | [25] |
| 2026 (Feb) | FAR.AI's Concept Influence and Obfuscation Atlas — interpretability-driven training data attribution and honesty training | [26][27] |
| 2026 (Feb) | Goodfire Series B ($150M, $1.25B valuation) | [4] |
| 2026 (Feb) | "Formal Mechanistic Interpretability" (Hadad et al.) — provable guarantees for circuit discovery via neural-network verification | [28] |
| 2026 (May) | DeepMind's **Gemma Scope 2** — SAEs and transcoders on every layer of Gemma 3 with Matryoshka training, skip-transcoders, cross-layer transcoders | [29] |
| 2026 (May) | "MechRL" (Khadka) — PPO-based circuit discovery that transfers across behaviors | [30] |

The arc is clear: 2020–2022 established primitives; 2023 produced the first large-scale end-to-end circuits (IOI, induction heads) and automated circuit discovery; 2024 was the "scale year" — SAEs went from 4K to 34M features, Gemma Scope and Llama Scope democratized access; 2025 produced attribution graphs, the first MI benchmark, vision MI tools, and the open-problems review; 2026 is the year MI moved into production system cards and started to face formal critiques head-on.

## III. Methods Landscape

### Sparse autoencoders (SAEs)

The single biggest technical unlock of the last three years. SAEs are dictionary-learning models trained to reconstruct activations from a sparse bottleneck: an encoder maps activations to a high-dimensional latent space, a sparsity penalty forces only a few latents to fire per input, and a decoder reconstructs the original. The latents (after training) become the field's preferred feature vocabulary.

- **L1 SAEs (Bricken 2023, Cunningham 2023)** — the original formulation, penalizing the L1 norm of latents [10][11].
- **TopK SAEs (Gao et al., 2024, OpenAI)** — activate exactly the K highest pre-activations, no L1 penalty required. Used in the 16M-feature GPT-4 experiment; became state of the art on standard metrics [13][31].
- **JumpReLU SAEs (Rajamanoharan et al., 2024, DeepMind)** — a learnable threshold; the architecture used for Gemma Scope [17].
- **BatchTopK SAEs (Bussmann et al., 2024)** — relax TopK's per-sample constraint to a per-batch average, allowing adaptive allocation [32].
- **Gated SAEs (Rajamanoharan et al., 2024)** — split the magnitude and direction of latents; reduces dead features.
- **Matryoshka SAEs (DeepMind, 2025)** — hierarchical, nested-feature groups; used in Gemma Scope 2 to detect more concepts and resolve flaws [29].

SAEs have well-known failure modes. The "Open Problems" review lists **feature splitting, absorption, and composition** as pathologies where sparsity-as-proxy-for-interpretability breaks down under optimization pressure [3]. Feature reconstructions also have non-trivial error, which raises the question of whether SAE features faithfully represent the model [3]. A 2025 ICLR paper ("Sparse Autoencoders Do Not Find Canonical Units") and the 2025 NeurIPS "Revising End-To-End Sparse Autoencoder Training" paper both push back on the assumption that SAE features are unique [33][34].

### Transcoders and cross-layer transcoders (CLTs)

A transcoder is like an SAE but trained to reconstruct the *output of a layer given its input*, instead of reconstructing the activations at a site. The advantage for circuit analysis is that features become inputs and outputs of MLP sub-layers, making the wiring between features explicit. Dunefsky et al. (2024) showed transcoders are more interpretable than SAEs on equivalent setups [16]; Paulo et al. (2025) added skip-transcoders that match reconstruction loss without hurting interpretability [35]. Anthropic's cross-layer transcoders (CLTs) extend the idea across L layers and are the backbone of their attribution graphs.

### Attribution graphs and circuit tracing

Attribution graphs are the empirical artifacts produced by perturbing or linearly attributing through a replacement model (a sparse, transcoder-equipped substitute for the original network). Nodes are features (or reconstruction-error nodes); edges carry linear attribution coefficients; the graph is pruned for legibility. Anthropic's "Circuit Tracing: Revealing Computational Graphs in Language Models" and the companion "On the Biology of a Large Language Model" applied this to Claude 3.5 Haiku and found evidence for forward planning in poetry composition, multilingual conceptual circuits, "motivated reasoning" where the model decides on an answer before generating a justification, and an internal feature for "evaluation awareness" [19][20]. The library was open-sourced on May 29, 2025, with a Neuronpedia-hosted frontend for interactive exploration [21].

### Activation patching, path patching, and causal scrubbing

Activation patching swaps a clean run's activation with one from a corrupted run at a single site; path patching does it for an edge between two sites; causal scrubbing is a stricter hypothesis test. The three define a hierarchy of causal strictness. Conmy et al.'s ACDC automated the first step of this workflow [12]. Edge Attribution Patching (EAP) by Syed et al. (2024) made path patching efficient at scale; subsequent work (Hanna et al., 2024) provided statistical-estimator framing, showing EAP has high variance under data resampling — another identifiability concern [36].

### Automated circuit discovery

ACDC and EAP established automated but per-behavior pipelines. **MechRL** (Khadka, May 2026) reframes circuit discovery as reinforcement learning: a single PPO policy over GPT-2 small's 144 attention heads, trained jointly on the induction and IOI tasks, transfers zero-shot to held-out behaviors (docstring completion) and recovers 96% of the oracle ceiling with best-of-five planning — a step toward amortized, reusable circuit discovery rather than per-task bespoke search [30].

### Sparse dictionary learning as the dominant paradigm

Across these variants, the dominant paradigm is sparse dictionary learning: find a sparse, overcomplete basis that approximates the activations (or component outputs) of the model. The vocabulary has converged on a small number of primitives (L1 / TopK / JumpReLU / Matryoshka for activation decomposition; transcoders for component decomposition; CLTs for layer-spanning decomposition), and the field's center of gravity in 2026 is on improving these primitives and wiring them into a tractable circuit-discovery pipeline.

## IV. Landmark Circuits Discovered

The field's proof-of-concept empirical results — concrete cases where MI yielded a falsifiable, useful account of behavior.

- **Induction heads (Olsson et al., 2022).** A two-head circuit (previous-token head + induction head) that detects the pattern `[A][B] ... [A]` and predicts `[B]`. Induction heads form abruptly during training, visible as a sharp loss-curve kink. They appear in essentially every transformer trained on natural language and are the leading mechanistic candidate for in-context learning [9].
- **IOI (Indirect Object Identification) circuit (Wang et al., 2023).** In GPT-2 small, the task "John and Mary went to the shops. Then John gave a drink to _" → "Mary" is implemented by 26 attention heads in 7 classes (Duplicate Token Heads, Induction Heads, S-Inhibition Heads, Name Mover Heads, etc.). The largest end-to-end mechanistic account of a natural-language behavior at the time of publication [6].
- **Refusal direction (Arditi et al., 2024).** Across 13 open-source chat models up to 72B parameters, refusal behavior is mediated by a one-dimensional subspace in the residual stream. Projecting out the direction removes refusal; adding it induces refusal even on benign prompts [15]. The simplicity is what makes this finding both practically powerful (rank-one jailbreaks, ROSI) and theoretically interesting (see "Identifiability" below).
- **Multilingual conceptual circuits (Anthropic, 2025).** Attribution graphs on Claude 3.5 Haiku showed that the same query in English, French, and Chinese activates overlapping language-independent features in the middle layers, with language-specific features at the boundaries. The overlap fraction is larger than in smaller models — evidence for an emergent "language of thought" [20].
- **Planning in poetry (Anthropic, 2025).** Claude 3.5 Haiku plans rhyming words several tokens ahead, committing to the destination before constructing the line. Attribution graphs captured the planning features activating before the relevant tokens are produced [20].
- **Motivated reasoning (Anthropic, 2025).** When given an incorrect hint on a math problem, Claude produces a coherent-but-wrong justification: attribution graphs reveal the model arriving at the answer and then reverse-engineering a plausible chain of thought [19][20].
- **Planning in model-free RL agents (FAR.AI, 2025).** Concept-based interpretability applied to the DRC agent (Guez et al., 2019) in Sokoban found learned concept representations that internally formulate plans predicting long-term effects of actions — first mechanistic evidence that a model-free RL agent can plan [37].

## V. Open-Source Tools and Infrastructure

The tooling stack has caught up to the theory: a researcher can now load a model, run SAEs on it, generate attribution graphs, and share results without building anything from scratch.

| Tool | Role | Repo | Notes |
|------|------|------|-------|
| **TransformerLens** | Load 50+ open-source transformer-style models; cache and intervene on activations via HookPoints | `TransformerLensOrg/TransformerLens` (~2,900 stars) | Created by Neel Nanda; maintained by Bryce Meyer. v2.0 removed HookedSAE (moved to SAELens) [38]. |
| **SAELens** | Train, load, and analyze sparse autoencoders; integrates with TransformerLens and nnsight | `decoderesearch/SAELens` (formerly `jbloomAus/SAELens`) | Directory of pre-trained SAEs for GPT-2, Gemma 2, Pythia; supports TopK, JumpReLU, BatchTopK [39]. |
| **nnsight / NDIF** | Framework-agnostic intervention library with remote execution on large models | `ndif-team/nnsight` | Lets you run interventions on hosted frontier-size models. |
| **pyvene** | Causal-intervention library; supports path patching, ablations | `stanford-crfm/pyvene` | |
| **CircuitsVis** | Interactive attention-pattern and activation visualizations | `TransformerLensOrg/CircuitsVis` | |
| **SAE-Vis** | SAE feature dashboards | Built into SAELens | |
| **circuit-tracer** | Generate and visualize attribution graphs from cross-layer transcoders | `safety-research/circuit-tracer` (Anthropic + Decode Research; open-sourced May 2025) | Front-end on Neuronpedia; supports Gemma-2-2B and Claude Haiku [21]. |
| **Prisma (ViT-Prisma)** | Vision/video mechanistic interpretability toolkit | `Prisma-Multimodal/ViT-Prisma` | Supports 75+ ViT/CLIP/DINO models; ships 80+ pre-trained SAEs and transcoders [22]. |
| **OpenAI sparse_autoencoder** | Reference SAE training code | `openai/sparse_autoencoder` | Used for the 16M-feature GPT-4 experiment [13]. |

**Neuronpedia** (neuronpedia.org) is the field's interactive browser: features can be searched, visualized, and steered. As of June 2026 it added HeadVis and NLA (Natural Language Attribution) contributions plus more SAEs; Goodfire and other contributors maintain dashboards. The frontend hosts the circuit-tracer integration and is the de facto place to publish interpretability artifacts.

**Goodfire's Ember** is a hosted API that exposes SAE features from production-scale models (Llama 3.3 70B, DeepSeek R1 671B) as programmable primitives. It is the first commercial mechanistic-interpretability platform — see Section VII for the company context.

The user can do the following pipeline today, end-to-end, without writing infrastructure:

1. `transformer_lens.HookedTransformer.from_pretrained("gpt2-small")` — load the model with hooks.
2. `sae_lens.SAE.from_pretrained("gpt2-small-res-jb", "blocks.8.hook_resid_pre")` — load a pre-trained SAE.
3. `model.run_with_cache(tokens)` then `sae.encode(cache["resid_pre", 8])` — get sparse features.
4. Identify which features activate for a target concept; ablate or steer them and observe the behavioral change.
5. Publish the analysis on Neuronpedia or use Goodfire's API for a hosted version.

This is a meaningful change from 2022, when reproducing Anthropic's toy-model work required re-implementing the training pipeline.

## VI. Applications

### Alignment and safety

This is where MI has had the most concrete impact.

- **Refusal-direction mechanics.** Arditi et al.'s finding that refusal is a 1D subspace [15] enabled a wave of follow-up work. ROSI (Rank-One Safety Injection, arXiv:2508.20766) amplifies the refusal direction by permanently editing model weights, improving jailbreak robustness without retraining [40]. Conversely, the same finding is what makes "refusal erasure" jailbreaks effective — the safety behavior is fragile because it is concentrated in a single direction. A 2025 EMNLP paper ("Understanding Refusal in Language Models with Sparse Autoencoders") decomposed refusal into distinct SAE feature sets (harm, refusal-style) and showed jailbreaks specifically suppress refusal-related features [41].
- **Jailbreak mechanism detection.** Multiple papers used mechanistic analysis to characterize jailbreak behavior: Wei et al. (2024, "Jailbreak and Guard Aligned Language Models With Only Two Characters") identified safety neurons in LLMs (~5% of neurons mediate safety behavior; patching them restores 90% of safety performance) [42]. A 2025 ICML paper ("Circuit Discovery Helps To Detect LLM Jailbreaking") used edge attribution patching on LLaMA-2-7B-chat to identify circuits responsible for affirmative jailbreak responses; ablating these circuits in the first-token prediction reduced attack success rates by up to 80% [43]. A 2026 paper identified two functionally distinct attention-head classes — Adversarially Compromised Heads (ACHs) and Safety-Aligned Heads (SAHs) — and showed that jailbreaks selectively suppress a small number of ACHs while robust safety signals remain in SAHs [44].
- **Deception and honesty training.** FAR.AI's "Obfuscation Atlas" (February 2026) studies what happens when you train models with a deception-probe penalty. It identifies two failure modes: **obfuscated activations** (the model changes its internals to evade the detector) and **obfuscated policies** (the model adds justifications that fool black-box monitors). Obfuscated activations arise from representation drift during RL even without a probe penalty. The paper shows that strong KL regularization plus a detector penalty yields honest policies — a validation of white-box interpretability as a viable training signal, but only with careful hyperparameters [27].
- **Concept Influence for training-data attribution (FAR.AI, Feb 2026).** Generalizes influence functions to attribute behavior to semantic directions (linear probes or SAE features) rather than individual test examples. Achieves 20× speedup over standard influence functions and can unsupervisedly detect "sleeper agent" triggers in fine-tuning data. A practical tool for safety-oriented data curation [26].
- **Production safety cards.** Anthropic's Claude Sonnet 4.5 system card (Oct 2025) and Opus 4.6 (Feb 2026) use SAEs trained on middle-layer snapshots during post-training to track features like "fake content," "evaluation awareness," and reward-model bias, documenting changes in rationalism/safety features over post-training [24].

### Biology and protein language models

SAEs were developed for LLMs but have been ported to biology with promising results.

- **PNAS 2025 (Gujral et al., MIT).** Trained SAEs and transcoders on ESM2 (a 650M-parameter protein language model) and showed sparse features are more interpretable than ESM2's native neurons, aligning with Gene Ontology terms, specific protein families (NAD Kinase, IUNH, PTH), and functional categories like olfactory receptors [45].
- **InterPLM (Nature Methods, 2025).** Systematic framework that extracts thousands of interpretable features from PLMs, including binding sites, structural motifs, and functional domains. Notably: features persist across model scales (superposition is consistent in larger PLMs), and the framework includes automated interpretation via LLMs and a feature visualizer for hypothesis generation [46].
- **Reticular (Y Combinator).** A biotech startup applying mechanistic interpretability to protein design. Trains SAEs on ESM2-3B (the base for ESMFold) and adapts Matryoshka SAEs for proteins. Their pilot identified the first interpretable features in protein models and used them to steer Green Fluorescent Protein toward more fluorescent variants — a direct mechanistic control loop over a biological model [47].
- **Markov Bio's "Through a Glass Darkly."** A three-part essay arguing that mechanistic interpretability will be necessary for AlphaFold-style models, that SAEs give us a "language" for virtual-cell biology, and that this will accelerate drug discovery in the next five years — a 5-year roadmap for "liminal legibility" of biological simulators [48].
- **ICML 2025 "From Mechanistic Interpretability to Mechanistic Biology."** Training and evaluating SAEs on protein LMs; conceptual bridge between the two fields [49].

### Vision

- **Prisma toolkit (Joseph et al., April 2025).** First dedicated toolkit for vision mechanistic interpretability: hooks for 75+ vision transformers (CLIP, DINO, V-JEPA, ViViT), 80+ pre-trained SAEs and transcoders, and circuit-analysis infrastructure. Surprising findings: vision SAEs are typically *less* sparse than language SAEs, and SAE reconstructions can *decrease* model loss in some settings [22].
- **SAE-ViT (Stevens et al., Feb 2025).** SAE features for vision models with both real-image exemplars (semantic interpretation) and decoding vectors (causal control) [50].
- **Causal tracing for BLIP (Palit et al., 2023).** Adapts causal tracing to multimodal models; showed vision-language models have partially separate visual and textual pathways [51].

### Reinforcement learning

- **Concept-based interpretability of DRC agent (FAR.AI).** First mechanistic evidence that a model-free RL agent (DRC, Guez et al. 2019) in Sokoban learns concept representations that internally formulate plans predicting long-term effects [37].
- **MechRL (Khadka, May 2026).** Treats circuit discovery as RL over GPT-2's attention heads, with contrastive rewards; demonstrates amortized, transferable circuit discovery [30].

### Training-time and steering applications

- **Goodfire's Ember** is the first commercial platform built around MI: customers (Microsoft, Mayo Clinic, Arc Institute, Rakuten, Apollo Research, Haize Labs) use it to inspect and modify model internals without retraining — for PII detection in chatbots (Rakuten), multilingual PII specifically, and persona steering [52][53].
- **Anthropic's Golden Gate Claude (2024).** The canonical demonstration of feature-level steering: clamping the "Golden Gate Bridge" feature forces Claude to talk about the bridge regardless of context [14].
- **Persona Vectors / Concept Vectors.** A related line (not strictly MI but adjacent) finds linear directions that control persona-style traits — see Concept Influence for the formalization [26].

## VII. Industry Players and Commercial Activity

- **Anthropic.** The dominant academic-industrial contributor. The Interpretability team's open-problems co-authored review [3], the 34M-feature SAE work [14], the Circuit Tracing papers [19][20], and the open-source circuit-tracer [21] all come from here. Anthropic's CEO Dario Amodei has publicly named mechanistic interpretability as a strategic priority; in April 2025 Anthropic made its first corporate investment — leading Goodfire's $50M Series A [54].
- **OpenAI.** Historically important via Cunningham et al. (2023) and the 16M-feature GPT-4 SAEs [13]; also working on interpretable-by-design models ("Weight-Sparse Transformers Have Interpretable Circuits," Nov 2025) [25]. The Superalignment team's dissolution (2024) reduced some interpretability focus.
- **Google DeepMind.** Gemma Scope [17] and Gemma Scope 2 [29] are the largest open-source SAE releases; the team also publishes JumpReLU SAEs, transcoders, and Matryoshka SAEs. The Mishax interpretability library supports Gemma Scope internally [17].
- **Goodfire.** A mechanistic-interpretability research lab and public benefit corporation, founded June 2024 by Eric Ho, Dan Balsam, and Tom McGrath. Their flagship is Ember, a hosted mechanistic-interpretability API exposing SAE features for Llama 3.3 70B, DeepSeek R1 671B, and others. Funding trajectory: $7M seed (Aug 2024) → $50M Series A (Apr 2025, led by Menlo Ventures with Anthropic's first corporate investment) → $150M Series B (Feb 2026, B Capital-led) at a $1.25B valuation. Total raised ~$207M. Customers include Microsoft, Mayo Clinic, Arc Institute, Rakuten, Apollo Research, Haize Labs [4][52][53].
- **Neuronpedia.** An open-source interpretability platform (neuronpedia.org) that hosts the world's first interpretability API (March 2024) and the circuit-tracer frontend. Co-led by Decode Research with Anthropic Fellows collaboration. As of June 2026 it added HeadVis and NLA contributions plus expanded SAE coverage [55].
- **FAR.AI.** A nonprofit AI safety research org (founded 2022) producing both interpretability methods and applications: Concept Influence (Feb 2026) [26], Obfuscation Atlas (Feb 2026) [27], TamperBench (Feb 2026) for evaluating tamper resistance of open-weight models, the DRC-agent planning work [37], and the "Compressed Computation is (probably) not Computation in Superposition" paper (Dec 2025).
- **Reticular.** Biotech startup applying MI to protein design; reports the first interpretable features in ESMFold-level protein models [47].
- **Apollo Research / Haize Labs.** Independent alignment-research orgs that consume mechanistic-interpretability methods (e.g., alignment-faking evaluation on Claude 4.5 Haiku, where keyword analysis showed models flagged 42–78% of evaluation prompts as jailbreaking attempts or evaluations across conditions [56]).

A live tension in 2025–2026: Goodfire's pivot toward using interpretability to *intentionally design* models during training (their "interpretability for training" thesis) has sparked community debate about whether this compromises interpretability's role as an independent safety auditor. The argument is that an interpretability method used as a training signal is no longer independent of the model's optimization process — which is exactly what the Obfuscation Atlas paper documents can go wrong [27].

## VIII. Surveys, Benchmarks, and the Canonical Reading List

### Surveys

- **"A Practical Review of Mechanistic Interpretability for Transformer-Based Language Models" (Rai et al., 2024, arXiv:2407.02646).** The taxonomy used here — features, circuits, universality — comes from this paper [57].
- **"Mechanistic Interpretability for AI Safety — A Review" (Bereska & Gavves, 2024, arXiv:2404.14082).** Critical assessment of MI's relevance to AI safety, including the dual-use risks and the "streetlight interpretability" critique (cherry-picking on toy models) [58].
- **"A Survey on Sparse Autoencoders" (arXiv:2503.05613).** Specifically about SAEs; the standard reference for SAE variants and metrics [31].
- **"A Survey on Mechanistic Interpretability in AI" (ACM, 2025).** Broader field survey; standard reference [59].

### The Open Problems review

**"Open Problems in Mechanistic Interpretability" (Sharkey, Chughtai, Batson, Lindsey, Wu, Bushnaq, Goldowsky-Dill, Heimersheim, Ortega, Bloom, Biderman, Garriga-Alonso, Conmy, Nanda, Rumbelow, Wattenberg, Schoots, Miller, Michaud, Casper, Tegmark, Saunders, Bau, Todd, Geiger, Geva, Hoogland, Murfet, McGrath — submitted January 27, 2025)** is the canonical "where the field is honest with itself" document. It organizes open problems across four areas: (1) methods and foundations, (2) standards and rigor, (3) scalability, (4) socio-technical challenges. Specific issues flagged include [3]:

- The lack of a satisfying formal definition of "feature" — every formalization so far has problems.
- SDL (sparse dictionary learning) reconstruction errors are too high for the largest models.
- Sparsity is an imperfect proxy for interpretability (with feature splitting, absorption, composition as concrete pathologies).
- "Interpretability illusions" — plausible explanations for arbitrary activation directions; adversarial models can deliberately mislead (Geirhos et al., 2025).
- The unresolved question of which networks should serve as "model organisms" (the *Drosophila* of mechanistic interpretability).
- Lack of consensus on evaluation standards — ACDC, EAP, and MechRL all propose different metrics.

### Benchmark

**MIB (Mueller et al., arXiv:2504.13151, April 2025)** is the first benchmark for mechanistic interpretability methods, with two tracks spanning four tasks and five models. It is the early answer to the evaluation-standards problem but is far from the last word [2].

### Reading Path

For a newcomer who knows transformers but not MI, the foundational sequence is short and largely the same as Neel Nanda's curated list [60]: start with the conceptual framing (Olah's Circuits Thread [8] and Elhage et al.'s Mathematical Framework [1]), then read the two landmark empirical circuits in order of complexity — induction heads [9], then the IOI circuit [6]. These four pieces give you the primitives, the framing, and two concrete worked examples of what reverse-engineering looks like at small and medium scale.

The intermediate step is the theoretical foundation: Toy Models of Superposition [7] for the formal model, then the SAE scaling story (Scaling Monosemanticity [14]) for how the field moved from 4,096 features on a toy model to 34 million features on a frontier LLM. Once you have this, Scaling and Evaluating Sparse Autoencoders [13] and Gemma Scope [17] show how the same ideas scale across labs.

For the current frontier, read Open Problems in Mechanistic Interpretability [3] before anything in 2025 — it is the most honest map of what the field has not yet solved. Then the Biology of a Large Language Model [20] shows what attribution graphs look like on Claude 3.5 Haiku, and the Practical Review [57] is the standard taxonomy. From there, branch into applications (alignment [27][40], biology [45][47], vision [22]) or critique (the identifiability result [5] and the philosophy argument [23]).

For hands-on work: TransformerLens tutorials (the ARENA course is the standard pedagogical entry) [38], then SAELens [39], then circuit-tracer [21]. The ARENA "Intro to Mechanistic Interpretability" chapter is what most active researchers in the field used to learn the practice.

## IX. Critiques and the Identifiability Question

Mechanistic interpretability's most fundamental methodological challenge surfaced in 2025–2026: **interpretations may not be unique**.

**"Everything, Everywhere, All at Once: Is Mechanistic Interpretability Identifiable?" (Méloux, Maniu, Portet, Peyrard — ICLR 2026)** systematically tested two MI strategies — "where-then-what" (find a circuit, then interpret it) and "what-then-where" (start with a candidate algorithm and search for it in the network). Using simple Boolean tasks and small MLPs where they could enumerate all candidate explanations, the authors found [5]:

- Multiple circuits can replicate the same model behavior (e.g., 85 perfect circuits in a 2-layer XOR MLP; hundreds of thousands at slightly higher capacity).
- For any single circuit, multiple interpretations exist (hundreds of consistent logic-gate assignments per circuit).
- Dozens of functionally distinct algorithms can be aligned with the network; each can be mapped to multiple disjoint neural subspaces.
- The candidate space of explanations scales combinatorially with circuit size.

A follow-up paper ("Characterizing Mechanistic Uniqueness and Equivalence," ICLR 2026) showed that *model capacity* and *task complexity* are the dominant predictors: wider hidden layers admit more equivalent circuits; simpler tasks collapse to fewer pathways. Regularization techniques like L1 only weakly affect the number of valid circuits [61].

These are not kill-shots — the field can still find *useful* circuits and *causally valid* features — but they mean the circuits researchers publish may be one of many possible mechanistic explanations. The implication for any application that treats "the circuit for X" as a stable object (auditing, control, alignment) is significant.

Other notable critiques:

- **"Mechanistic Interpretability Needs Philosophy" (Williams et al., June 2025).** Argues the field's open problems (how to decompose networks, how to validate interpretations, what counts as a feature) are partly *philosophical* and would benefit from explicit philosophical engagement [23].
- **"MI as Statistical Estimation" (Hanna et al., 2024–2025).** Argues circuit-discovery methods should be viewed as statistical estimators; documents high variance under data resampling and hyperparameter sensitivity [36].
- **"Interpretability Will Not Reliably Find Deceptive AI" (Alignment Forum).** A community essay arguing that interpretability can raise the cost of deception but cannot reliably detect it in capable models [62].
- **"Bechtel approach" (Biological Theory, 2025).** A philosophy-of-science analysis arguing mechanistic explanations in AI and biology have similar commitments and limitations [63].
- **Reticular / Goodfire "interpretability-for-design" tension.** Whether using MI as a training signal preserves its independence as a safety audit tool is contested in 2026 [4][52].

## X. The 2026 Frontier

The most recent work pushes in four directions:

- **Formal verification.** "Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees" (Hadad, Katz, Bassan, Feb 2026) uses neural-network verification to give circuits with provable guarantees over continuous input domains — a step toward mathematical rather than empirical confidence [28].
- **Cross-model universality.** Goodfire's Universal Sparse Autoencoders (USAEs) learn a shared sparse-autoencoder dictionary across multiple pretrained models, enabling cross-model feature alignment and "coordinated activation maximization" — a path toward general-purpose interpretability primitives [52].
- **Formal causal grounding.** "Causally Grounded Mechanistic Interpretability for LLMs" (arXiv:2603.09988) generates natural-language explanations of circuit analyses with causal faithfulness guarantees; on the IOI task in GPT-2 small it identifies six attention heads accounting for 61.4% of the logit difference, with 100% sufficiency but only 22% comprehensiveness — explicitly documenting where explanations are incomplete [64].
- **Specialized architectures.** Weight-sparse transformers (Gao et al., Nov 2025) train models with mostly-zero weights to make circuits more recoverable by pruning [25]. "Interpretable and Testable Vision Features via Sparse Autoencoders" (Stevens et al., Feb 2025) extends this testability paradigm to vision [50].

The 2026 ICML Mechanistic Interpretability Workshop is the field's first dedicated venue with a structured program; papers there will likely consolidate several of these threads [65].

## XI. Where This Leaves the Field

Mechanistic interpretability has graduated from a small theoretical exercise into an engineering discipline with production tools, frontier-scale empirical results, a real commercial sector, and a credible claim to be the most promising path to *enumerative* safety — being able to list and verify the algorithms inside a trained model rather than only evaluating its outputs. By mid-2026, Anthropic's production system cards use SAE features diagnostically; OpenAI ships interpretable-by-design weight-sparse transformers; DeepMind releases the largest open SAE suites in existence; a startup valued at $1.25B sells mechanistic-interpretability as a service; a nonprofit publishes interpretability-based training methods; and academic groups from Princeton, MIT, Harvard, Hebrew University, Tel Aviv, Northeastern, and elsewhere contribute.

But the field is also openly uncertain. The Méloux non-identifiability result is not a curiosity — it means any specific mechanistic explanation is one of many, and the choice among them depends on theoretical and methodological commitments the field has not yet made. The "Open Problems" review enumerates these gaps: no satisfying definition of "feature," no consensus on evaluation, no model organism, no validated account of how MI scales to deception detection. The philosophy critique argues these are partly conceptual, not just technical.

The practical implication for someone entering the field today: the *technical* work is tractable and well-tooled — TransformerLens, SAELens, circuit-tracer, Neuronpedia form a complete loop from model loading to publication. The *intellectual* work — what counts as an explanation, whether the explanations are unique, whether they generalize — is genuinely open. That combination is unusual and makes the field attractive to work in.

---

## References

[1] Elhage, N. et al. "A Mathematical Framework for Transformer Circuits." Anthropic, Dec 2021. https://transformer-circuits.pub/2021/framework/index.html

[2] Mueller, A. et al. "MIB: A Mechanistic Interpretability Benchmark." arXiv:2504.13151, April 2025. https://arxiv.org/abs/2504.13151

[3] Sharkey, L. et al. "Open Problems in Mechanistic Interpretability." arXiv:2501.16496, January 2025. https://arxiv.org/abs/2501.16496

[4] "Goodfire — Longterm Wiki" (Series B $150M, $1.25B valuation, Feb 2026). https://www.longtermwiki.com/wiki/E430

[5] Méloux, M., Maniu, S., Portet, F., Peyrard, M. "Everything, Everywhere, All at Once: Is Mechanistic Interpretability Identifiable?" ICLR 2026. https://iclr.cc/virtual/2025/poster/30956

[6] Wang, K., Variengien, A., Conmy, A., Shleifer, S., Bau, D. "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small." arXiv:2211.00593, 2023. https://ar5iv.labs.arxiv.org/html/2211.00593

[7] Elhage, N. et al. "Toy Models of Superposition." Transformer Circuits Thread, 2022. https://transformer-circuits.pub/2022/toy_model/index.html

[8] Olah, C. et al. "Zoom In: An Introduction to Circuits." Distill, 2020. https://distill.pub/2020/circuits/zoom-in/

[9] Olsson, C. et al. "In-context Learning and Induction Heads." Transformer Circuits Thread / arXiv:2209.11895, September 2022. https://arxiv.org/abs/2209.11895

[10] Bricken, T. et al. "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning." Transformer Circuits Thread, October 2023. https://transformer-circuits.pub/2023/monosemantic-features/

[11] Cunningham, H., Ewart, A., Riggs, L., Huben, R., Sharkey, L. "Sparse Autoencoders Find Highly Interpretable Features in Language Models." arXiv:2309.08600, October 2023. https://arxiv.org/abs/2309.08600

[12] Conmy, A. et al. "Towards Automated Circuit Discovery for Mechanistic Interpretability." NeurIPS 2023 / arXiv:2304.14997. https://arxiv.org/abs/2304.14997

[13] Gao, L. et al. "Scaling and Evaluating Sparse Autoencoders." OpenAI, arXiv:2406.04093, June 2024. https://cdn.openai.com/papers/sparse-autoencoders.pdf

[14] Templeton, A. et al. "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet." Anthropic, May 2024. https://transformer-circuits.pub/2024/scaling-monosemanticity/

[15] Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., Nanda, N. "Refusal in Language Models Is Mediated by a Single Direction." NeurIPS 2024 / arXiv:2406.11717. https://arxiv.org/abs/2406.11717

[16] Dunefsky, J., Chlenski, P., Nanda, N. "Transcoders Find Interpretable LLM Feature Circuits." arXiv:2406.11944, June 2024. https://arxiv.org/abs/2406.11944

[17] Lieberum, T. et al. "Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2." Google DeepMind, arXiv:2408.05147, August 2024. https://arxiv.org/abs/2408.05147

[18] He, Z. et al. "Llama Scope: Extracting Millions of Features from Llama-3.1-8B with Sparse Autoencoders." arXiv:2410.20526, October 2024. https://arxiv.org/abs/2410.20526

[19] Ameisen, D. et al. "Circuit Tracing: Revealing Computational Graphs in Language Models." Anthropic, March 2025. https://transformer-circuits.pub/2025/attribution-graphs/methods.html

[20] Lindsey, J. et al. "On the Biology of a Large Language Model." Anthropic, March 2025. https://transformer-circuits.pub/2025/attribution-graphs/biology.html

[21] Anthropic + Decode Research. "Open-sourcing circuit-tracing tools." Anthropic, May 2025. https://www.anthropic.com/research/open-source-circuit-tracing ; repo: https://github.com/safety-research/circuit-tracer

[22] Joseph, S. et al. "Prisma: An Open Source Toolkit for Mechanistic Interpretability in Vision and Video." arXiv:2504.19475, April 2025. https://arxiv.org/abs/2504.19475

[23] Williams, I. et al. "Mechanistic Interpretability Needs Philosophy." arXiv:2506.18852, June 2025. https://arxiv.org/abs/2506.18852

[24] Anthropic. Claude Sonnet 4.5 System Card, October 2025; Opus 4.6, February 2026. https://www-cdn.anthropic.com/... (system card references via useluminix summary: https://www.useluminix.com/reports/industry-analysis/understanding-sholto-douglas-trenton-bricken-s-frontier-model-training-thesis/source/2)

[25] Gao, L., Rajaram, A., Coxon, J., Govande, S. V., Baker, B., Mossing, D. "Weight-sparse transformers have interpretable circuits." arXiv:2511.13653, November 2025. https://arxiv.org/abs/2511.13653

[26] Kowal, M., Paulo, G., Jaburi, L., Tseng, T., McKinney, L. E., Heimersheim, S., Tucker, A. D., Gleave, A., Pelrine, K. "Concept Influence: Leveraging Interpretability to Improve Performance and Efficiency in Training Data Attribution." FAR.AI, arXiv:2602.14869, February 2026. https://far.ai/news/concept-data-attribution-02-2026

[27] Taufeeque, M., Heimersheim, S., Gleave, A., Cundy, C. "The Obfuscation Atlas: Mapping Where Honesty Emerges in RLVR with Deception Probes." FAR.AI, arXiv:2602.15515, February 2026. https://far.ai/research/the-obfuscation-atlas-mapping-where-honesty-emerges-in-rlvr-with-deception-probes

[28] Hadad, I., Katz, G., Bassan, S. "Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees." arXiv:2602.16823, February 2026. https://arxiv.org/abs/2602.16823

[29] Google DeepMind. "Gemma Scope 2." 2025–2026. https://deepmind.google/models/gemma/gemma-scope/

[30] Khadka, B. "MechRL: Reinforcement Learning Agents Perform Circuit Discovery for Mechanistic Interpretability." arXiv:2605.26343, May 2026. https://arxiv.org/abs/2605.26343

[31] "A Survey on Sparse Autoencoders." arXiv:2503.05613. https://arxiv.org/html/2503.05613v3

[32] Bussmann, B., Leask, P., Nanda, N. "BatchTopK Sparse Autoencoders." arXiv:2412.06410, December 2024. https://arxiv.org/abs/2412.06410

[33] "Sparse Autoencoders Do Not Find Canonical Units of Analysis." ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/file/84ca3f2d9d9bfca13f69b48ea63eb4a5-Paper-Conference.pdf

[34] "Revisiting End-To-End Sparse Autoencoder Training." arXiv:2503.17272, March 2025. https://arxiv.org/pdf/2503.17272.pdf

[35] Paulo, G., Shabalin, S., Belrose, N. "Transcoders Beat Sparse Autoencoders for Interpretability." arXiv:2501.18823, January 2025. https://arxiv.org/abs/2501.18823

[36] Hanna, M. et al. "Mechanistic Interpretability as Statistical Estimation." (EAP variance work). https://openreview.net/pdf/b517bd84b5bbb6a773d3862950c9d8934170f9e4.pdf

[37] "Concept-based interpretability for DRC agent in Sokoban — first mechanistic evidence of planning in model-free RL." FAR.AI, April 2025. https://far.ai/publications

[38] TransformerLensOrg. "TransformerLens: A Library for Mechanistic Interpretability of Generative Language Models." https://github.com/TransformerLensOrg/TransformerLens

[39] decoderesearch. "SAELens: Training Sparse Autoencoders on Language Models." https://github.com/decoderesearch/SAELens

[40] "Rank-One Safety Injection (ROSI)." arXiv:2508.20766, August 2025. https://ar5iv.labs.arxiv.org/html/2508.20766

[41] "Understanding Refusal in Language Models with Sparse Autoencoders." EMNLP Findings 2025. https://aclanthology.org/2025.findings-emnlp.338.pdf

[42] Wei, Z. et al. "Jailbreak and Guard Aligned Language Models With Only Two Characters." arXiv:2406.14144, June 2024. https://arxiv.org/pdf/2406.14144.pdf

[43] "Circuit Discovery Helps To Detect LLM Jailbreaking." ICML 2025. https://icml.cc/virtual/2025/50862

[44] "Robust Harmful Features Under Jailbreak Attacks: Mechanistic Evidence from Attention Head Specialization in Large Language Models." arXiv:2606.28153. https://arxiv.org/html/2606.28153v2

[45] Gujral, T. et al. "Sparse autoencoders uncover biologically interpretable features in protein language models." PNAS, 2025. https://www.pnas.org/doi/10.1073/pnas.2506316122

[46] "InterPLM: discovering interpretable features in protein language models via sparse autoencoders." Nature Methods, 2025. https://www.nature.com/articles/s41592-025-02836-7

[47] Reticular. "Towards Interpretable Protein Structure Prediction with Sparse Autoencoders." arXiv:2503.08764, March 2025. https://www.reticular.ai/research/interpretable-protein-structure-prediction

[48] Markov Bio. "Through a Glass Darkly: Mechanistic Interpretability's Path to End-to-End Biology." https://www.markov.bio/research/mech-interp-path-to-e2e-biology

[49] "From Mechanistic Interpretability to Mechanistic Biology: Training, Evaluating, and Interpreting Sparse Autoencoders on Protein Language Models." ICML 2025. https://icml.cc/virtual/2025/poster/43465

[50] Stevens, S., Chao, W.-L., Berger-Wolf, T., Su, Y. "Interpretable and Testable Vision Features via Sparse Autoencoders." arXiv:2502.06755, February 2025. https://arxiv.org/abs/2502.06755

[51] Palit, V., Pandey, R., Arora, A., Liang, P. P. "Towards Vision-Language Mechanistic Interpretability: A Causal Tracing Tool for BLIP." arXiv:2308.14179, August 2023. https://arxiv.org/abs/2308.14179

[52] Goodfire AI. "Announcing Our $50M Series A to Advance AI Interpretability Research." April 2025. https://www.goodfire.ai/blog/announcing-our-50m-series-a

[53] "Inside Goodfire: Building safer AI systems with interpretability." BlueDot Impact + Goodfire, 2025. https://www.youtube.com/watch?v=AtOpZ1L9egU

[54] "Anthropic's first corporate investment — leading Goodfire's $50M Series A." April 2025. (Reports: https://siliconangle.com/2025/04/18/ai-interpretability-startup-goodfire-raises-50m-funding/ ; https://www.prnewswire.com/news-releases/goodfire-raises-50m-series-a-to-advance-ai-interpretability-research-302431030.html)

[55] Neuronpedia. https://www.neuronpedia.org

[56] "Alignment-Faking Evaluations Measure Jailbreak Detection, Not Alignment Faking." LessWrong analysis on Claude 4.5 Haiku. https://www.lesswrong.com/posts/CMsnknfTayPeuoAX5/alignment-faking-evaluations-measure-jailbreak-detection-not

[57] Rai, D., Zhou, Y., Feng, S. et al. "A Practical Review of Mechanistic Interpretability for Transformer-Based Language Models." arXiv:2407.02646, 2024. https://arxiv.org/html/2407.02646v2

[58] Bereska, L., Gavves, S. "Mechanistic Interpretability for AI Safety — A Review." arXiv:2404.14082, April 2024. https://arxiv.org/abs/2404.14082

[59] "A Survey on Mechanistic Interpretability in AI." ACM, 2025. https://dl.acm.org/doi/10.1145/3787104

[60] Nanda, N. "An Extremely Opinionated Annotated List of My Favourite Mechanistic Interpretability Papers." Alignment Forum, July 2024. https://www.alignmentforum.org/posts/NfFST5Mio7BCAQHPA/an-extremely-opinionated-annotated-list-of-my-favourite-1

[61] "Characterizing Mechanistic Uniqueness and Equivalence." ICLR 2026. https://openreview.net/attachment?id=cHpZ5DE8wj&name=pdf

[62] "Interpretability Will Not Reliably Find Deceptive AI." Alignment Forum. https://www.alignmentforum.org/posts/PwnadG4BFjaER3MGf/interpretability-will-not-reliably-find-deceptive-ai

[63] "The Bechtel Approach to the Scope and Limits of Mechanistic Explanation." Biological Theory, 2025. https://link.springer.com/article/10.1007/s13752-025-00511-w

[64] "Causally Grounded Mechanistic Interpretability for LLMs." arXiv:2603.09988. https://arxiv.org/pdf/2603.09988.pdf

[65] "Mechanistic Interpretability Workshop at ICML 2026." https://mechinterpworkshop.com/

[66] Anthropic. "Tracing the thoughts of a large language model." March 2025. https://www.anthropic.com/research/tracing-thoughts-language-model