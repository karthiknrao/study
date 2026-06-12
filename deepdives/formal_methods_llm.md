# Deep Dive: Formal Methods with LLMs (2024–2026)

The intersection of LLMs and formal methods has gone from research curiosity to production-grade in 18 months. The field now spans **theorem proving** (Lean/Coq/Isabelle), **verified code generation** (Dafny/Verus), **model checking** (TLA+), and **hardware verification** (SVA/RTL) — with one frontier result in 2025 (Aristotle at IMO 2025) reaching gold-medal-level formal proofs.

---

## 1. Why This Space Is Exploding Now

Three forces converged in 2024–2026:

- **RLVR** (Reinforcement Learning with Verifiable Rewards) — a Lean 4 verifier gives a *binary, tamper-proof reward signal*, which is exactly what post-RLHF methods need.
- **Long-CoT reasoning models** (DeepSeek-R1, o1, Gemini Deep Think) — produce tactic sequences that search-based provers can drive.
- **Agentic frameworks** — multi-LLM loops (sketcher → prover → critic) now match specialist models with fewer parameters.

A useful framing: LLMs are *not* verifiers — they suggest proof steps. The **proof assistant** is the verifier. The LLM's job is to pick the right tactic in the right state, often with RAG over a math library like Mathlib.

---

## 2. Frontier Results (2024–2025)

| System | Org | Date | Result | Repo |
|---|---|---|---|---|
| **AlphaProof** | Google DeepMind | Nov 2024 (Nature 2025) | IMO 2024 silver (28/42), solved P6 (hardest) | closed |
| **AlphaGeometry 2** | DeepMind | 2024 | IMO silver combined | closed |
| **Gemini Deep Think** | DeepMind | 2025 | IMO 2025 **gold** | closed |
| **Aristotle** | Harmonic | Oct 2025 | IMO 2025 **gold-equivalent** in Lean 4 (P1–P5) | harmonic.fun |
| **Seed-Prover / 1.5** | ByteDance | 2025 | Strong on FATE-H/X (graduate-level) | – |
| **Goedel-Prover-V2 (32B)** | Princeton (Yang et al.) | 2025 | Beats 671B DeepSeek-Prover-V2 at 1/20 size | Goedel-LM |
| **DeepSeek-Prover-V2-671B** | DeepSeek | Apr 2025 | SOTA on miniF2F / PutnamBench | DeepSeek |
| **BFS-Prover-V2** | ByteDance | 2025 | Step-level best-first search | BFS-Prover-V2 |
| **Hilbert** | Apple | NeurIPS 2025 | Informal-sketch → formal-proof agent | arXiv 2509.22819 |
| **Leanabell-Prover-V2** | – | 2025 | Verifier-integrated reasoning | – |
| **APOLLO** | – | NeurIPS 2025 | Agentic LLM×Lean collaboration | – |
| **LEAP** | – | 2026 | Agentic + Gemini-3.1-pro | arXiv 2606.03303 |

DeepMind's 9 Erdos problems solve (2026) is the most consequential single result so far: an agent autonomously resolved 9 of 353 open Erdos problems using Lean.

---

## 3. The Major Sub-Areas

### 3.1 Theorem Proving in Lean 4

- **Datasets**: LeanDojo, ProofNet, PutnamBench, miniF2F, FormalMATH, ArXivLean (MathArena, ~1000 real arxiv theorems).
- **Methods**:
  - Best-first tree search (BFS-Prover, DeepSeek-Prover)
  - RAG over Mathlib (LeanDojo/ReProver)
  - Long-CoT + verifier (Seed-Prover 1.5)
  - Monte Carlo graph search (Aristotle)
  - Subgoal decomposition (DeepSeek-Prover-V2)
  - RLVR with Lean binary reward
  - Activation steering (Steering Language Models for Theorem Proving, 2025)
- **Benchmarks**: miniF2F-Test (~244 olympiad problems), PutnamBench (522 Putnam problems), FATE-H/X (grad-level), FormalMATH (NeurIPS 2025), SorryDB (real-world sorry placeholders in Mathlib).

### 3.2 Verified Code Generation (Dafny / Verus / Lean)

This is the most practical sub-area — generating code **with machine-checked proofs of correctness**.

Key papers and tools:

- **DafnyPro** — LLM-assisted Dafny verification
- **AlphaVerus** — bootstrapping verified code in Verus (Rust)
- **AutoVerus** — automated proof generation for Rust
- **CLEVER** — NeurIPS 2025 benchmark (avoids test-case supervision)
- **AlgoVeri** — aligned benchmark across Dafny, Verus, Lean
- **Vericoding benchmark** — POPL 2026 (Lean/Rust-Verus/Dafny)
- **Velvet** (MIT Ilya Sergey) — multi-modal verifier, integrates SMT-style Dafny/Verus with interactive Lean
- **"A Grand Challenge for Reliable Coding in the Age of AI Agents"** (Shuvendu K. Lahiri, Microsoft, arXiv 2603.17150) — formal "intent formalization" framing

The vision: rather than write tests, the LLM writes the *specification* in a verification-aware language, then the implementation, then the proof is checked by the SMT solver embedded in Dafny/Verus.

### 3.3 TLA+ and Model Checking

- **"Can LLMs Write Correct TLA+ Specifications?"** (arXiv 2606.05792, 2026) — evaluates TLA+ generation by GPT-4-class models.
- **"From Informal to Formal — Incorporating and Evaluating LLMs on TLA+"** (ACL 2025).
- **"Can LLMs model real-world systems in TLA+?"** (SIGOPS 2026).
- **"SysMoBench"** (2025) — formally modeling complex real-world systems.
- **Practical case**: AWS, Microsoft, and others have long used TLA+; LLMs are now being used to bootstrap specs (Sesh Nalla, "AI agents build functional equivalents").
- The Murat Demirbas blog ("TLA+ as a Design Accelerator") shows industry adoption lessons.

### 3.4 Hardware Verification (RTL / SVA)

This is the area with the most near-term industry value:

- **AssertionForge** (Bai et al., 2025) — structured-spec → SVA
- **STELLAR** (arXiv 2601.19903) — structure-guided retrieval
- **From Language to Logic: Bridging LLMs & Formal Representations for RTL Assertion Generation** (arXiv 2604.23100)
- **Agentic AI-based Coverage Closure for Formal Verification** (arXiv 2603.03147) — agent generates SVA from RTL + spec
- **AssertLLM** (ASPDAC 2025), **FVDebug**, **RTLFixer** — debugging formal failures
- **FormalRTL** (arXiv 2603.08738) — verified RTL synthesis at scale
- **"LLM-Assisted Circuit Verification: A Comprehensive Survey"** (ASPDAC 2026) — the field's first comprehensive survey
- **VeriGraphi** (arXiv 2604.14550) — multi-agent RTL framework
- **Comprehensive Verilog Design Problems** benchmark (arXiv 2506.14074)
- **OpenLLM-RTL** (ICCAD 2025)

### 3.5 Autoformalization

Translating natural-language math into formal statements. Key direction: "mathematician writes LaTeX, system autoformalizes to Lean, then prover attempts proof."

- **Towards a Common Framework for Autoformalization** (AAAI 2026)
- **Draft, Sketch, and Prove** (DSP) — guiding formal provers with informal sketches
- **Lean Finder** (SFU, 2025) — RAG to help discover the right Mathlib theorem
- **Moogle**, **LeanSearch** — semantic/premise search over Mathlib

### 3.6 Security and Protocol Verification

- **"LLM-Aided Automatic Modeling for Security Protocol Verification"** (ICSE 2025) — TLS 1.3, 5G AKA
- **SecGoal** (arXiv 2604.27601) — benchmark for extracting formalizable security goals
- **"Formal Security and Functional Verification of Cryptographic Protocol Implementations in Rust"** (CCS 2025)
- **Zeek-LLM** — generating Zeek IDS scripts with LLM + formal semantics

### 3.7 RLVR (Reinforcement Learning with Verifiable Rewards)

The unifying training paradigm. Lean 4's binary "proof accepted/rejected" feedback is the cleanest RL signal in all of LLM training. Repos:

- **Awesome RLVR** — opendilab curated list
- **JURY-RL** — label-free RLVR (OpenReview 2025)
- **Process-Verified RL for Theorem Proving** (NeurIPS 2025)
- **Leanabell-Prover-V2** — verifier-integrated reasoning
- **RPT: Reinforcement Learning during Pretraining** (Vizuara)

---

## 4. Open-Source Repos and Tools You Can Run Today

### Provers (training / inference)

- **lean-dojo/leandojo** — Python library, data extraction from Lean repos, ReProver (RAG-augmented prover)
- **ByteDance-Seed/BFS-Prover-V2** — step-level best-first search, SOTA open
- **Goedel-LM/Goedel-Prover-V2** — Princeton (Kaiyu Yang), 32B SOTA
- **deepseek-ai/DeepSeek-Prover-V2** — 671B open weights
- **HuggingFaceH4/Numina-Lean-Agent** — agentic general Lean agent (arXiv 2601.14027)

### Lean Tooling

- **leanprover-community/mathlib4** — the main math library
- **leanprover-community/lean4** — the prover itself
- **oOo0oOo/lean-lsp-mcp** — Lean LSP exposed as MCP for Claude/agent integration
- **lean-dojo/Pantograph** — Lean 4 interface for ML systems (state/tactic I/O)
- **BrettKoonce/lean4-mlir** — Lean specs for neural architectures (generates StableHLO MLIR)
- **ImperialCollegeLondon/FLT** — formal proof of Fermat's Last Theorem in Lean
- **UlamAI** (ulam.ai) — open-source Lean 4 theorem prover + formalizer

### Code Verification

- **dafny-lang/dafny** — Microsoft's verification language
- **verus-lang/verus** — Verus (verified Rust)
- **microsoft/verus**-related: AutoVerus
- **Ezio-SP/AlphaVerus** — bootstrapping Verus code with LLMs

### Hardware Verification

- **NVlabs/AutoDMP** — DMP macros
- **AssertionForge** (Bai et al.) — https://github.com/benklaber/AssertionForge
- **xforcevesa/STELLAR** — structure-guided SVA RAG

### Other Resources

- **opendilab/awesome-RLVR** — RLVR resources
- **seewoo5/awesome-ai-for-math** — 170 curated papers on AI for math
- **LightChen233/Awesome-Long-Chain-of-Thought-Reasoning**
- **gasstationmanager/lean4ai** — blog with running Lean+AI experiments

### Benchmark / Eval

- **matharena** — ArXivLean, FormalMATH eval
- **HuggingFaceTB/lean-atlas** — integrated proof environment
- **edayers/sorrydb** — real-world Lean sorry benchmark (arXiv 2603.02668)

---

## 5. Practical Ways to Get Started

If you want to *use* this stuff (not just read):

1. **Lean 4 + your favorite LLM** — install Lean 4 (`elan`, `lake`), open a `.lean` file in VS Code with the Lean extension, then point Claude/Cursor at it via the `lean-lsp-mcp` server. You'll get goal-aware suggestions.

2. **LeanCopilot** — `import Mathlib.Tactic.Copilot` in a Lean file and get LLM-suggested tactics inside the prover.

3. **Dafny quickstart** — `dafny-lang/dafny` has a tutorial; give Claude a function and ask for a `requires`/`ensures` spec.

4. **Verus** — install via `cargo`, write a `verus!` block, ask LLM to fill in the proof.

5. **TLA+** — Leslie Lamport's homepage has the spec; the PlusCal-to-TLA+ translator makes it readable. Then `tlc` model-checks it.

6. **For research** — start with LeanDojo-v2 (paper + lib), replicate ReProver, then swap in a current SOTA model (Goedel-Prover-V2 is a good default).

---

## 6. Field-Wide Open Problems

- **Autoformalization is the bottleneck.** Provers have caught up; turning math-typed-by-humans into Lean is still lossy.
- **Spec inference for code** — the "intent formalization" grand challenge. The LLM must guess what the user *meant*, then write both code and proof.
- **Repository-scale verification** — current LLM verifiers work on isolated functions; cross-module proofs are still hard.
- **Hallucination in formal settings** — Lean accepts *any* valid proof of any *plausible-looking* statement. A model can prove the wrong theorem. (This is the "Formal or not formal?" critique from Kevin Buzzard / Xena project.)
- **Cost** — Aristotle-scale runs need heavy compute. Cheap open models (32B class) catch up, but the frontier is compute-bound.

---

## 7. Recommended Reading (in priority order)

1. **"Formal Reasoning Meets LLMs"** — Communications of the ACM, 2025. The best field-wide overview.
2. **DeepMind AlphaProof Nature paper** (2025) — the IMO silver result, technically detailed.
3. **Hilbert** (arXiv 2509.22819) — cleanest agentic framework description.
4. **DeepSeek-Prover-V2** (arXiv 2504.21801) — clean RL + subgoal decomposition.
5. **Goedel-Prover-V2** — 32B beating 671B; elegant training recipe.
6. **"A Grand Challenge for Reliable Coding in the Age of AI Agents"** — Shuvendu Lahiri (Microsoft).
7. **"LLM-Assisted Circuit Verification: A Comprehensive Survey"** — ASPDAC 2026.
8. **"From Informal to Formal — Incorporating and Evaluating LLMs on TLA+"** — ACL 2025.

---

### Bottom line

The story of 2024–2026 in formal methods + LLMs is the **industrialization of the Lean 4 verifier as an RL reward**, plus agentic frameworks (Hilbert, Aristotle, LEAP) that wrap frontier LLMs in proof-search loops. Open-source is strong (Goedel-Prover-V2-32B, BFS-Prover-V2, LeanDojo, LeanCopilot). The most under-explored area is **TLA+**, and the most production-relevant is **hardware verification with SVA generation**. The grand challenge — autoformalization from natural language to a useful spec — remains open.
