I'll dig into this — evolutionary jailbreaks are a hot area in LLM safety/red-teaming research. Let me search for the latest papers and code.

Let me dig deeper for the latest 2025-2026 work and the most cited code repos.

Let me grab a few more specifics on the absolute latest (2026) work and the master code list.

Got a solid picture. Here's the lay of the land for **evolutionary / genetic-algorithm–based jailbreaks on LLMs** as of mid-2026. I'll organize it by method family, with the most recent work first.

---

## The big picture

Evolutionary jailbreaks treat the attack as a population-based search over prompts/suffixes/strategies. Same skeleton every time:

1. Initialize population (seed templates, random tokens, or LLM-generated)
2. Evaluate fitness (query target, score with judge LLM or token-level loss)
3. Select best candidates
4. Crossover + mutate
5. Iterate

The interesting part is *what* you're evolving and *how* — that's where the latest work differs.

---

## 1. Foundational: gradient + token-level (the baseline everyone builds on)

**GCG — Greedy Coordinate Gradient** (Zou et al., 2023) — still the workhorse. Optimizes discrete token suffixes via gradients on the white-box loss. Read the original paper or `llm-attacks/llm-attacks` repo. Limited because it needs white-box access and produces gibberish suffixes, but it spawned the whole field.

**Key GCG derivatives still in use:**
- **I-GCG** (Jia et al.) — better initialization + multi-coordinate updates → near-100% ASR
- **MAC** — momentum-accelerated GCG
- **Mask-GCG** (Mu et al., Sep 2025) — dynamically prunes redundant tokens, much faster
- **T-GCG** (2025) — simulated-annealing variant
- **AmpleGCG** — trains a generative model over successful GCG suffixes, can spit out hundreds per second
- **MAGIC** (Li et al., 2025) — "exploits index gradients" to skip wasted token swaps
- **Improved Techniques for Optimization-Based Jailbreaking** — ICLR 2025, near-100% ASR variants

These aren't strictly "evolutionary" but they're the starting population that GA methods often seed from.

---

## 2. Pure genetic-algorithm jailbreaks (the core evolutionary family)

| Paper | Year | What evolves | Notes |
|---|---|---|---|
| **AutoDAN** (Liu et al.) | ICLR 2024 | Hierarchical GA over natural-language sentences/paragraphs | First to produce *stealthy* (human-readable) GA jailbreaks. Code: `SheltonLiu-N/AutoDAN` |
| **AutoDAN-Turbo** (Liu et al.) | ICLR 2025 | Lifelong agent builds a strategy library from scratch | 74% higher ASR than baselines, code: `SaFoLab-WISC/AutoDAN-Turbo` |
| **AutoDAN-Reasoning** (Oct 2025) | arXiv | Adds test-time scaling on top of Turbo | Code: `SaFoLab-WISC/AutoDAN-Reasoning` |
| **LLM-Virus** (Yu et al., Jan 2025) | arXiv 2501.00055 | GA where LLM itself is the mutation/crossover operator, biology-inspired | Black-box, transferable. Code: `Ymm-cll/LLM-Virus` |
| **Semantic Mirror Jailbreak (SMJ)** (Feb 2024) | arXiv 2402.14872 | GA over mirrored/paraphrased prompts | Open-source LLM focus |
| **GPTFuzzer** (2024) | – | LLM-assisted mutation of seed jailbreak templates | |
| **GeneShift** (Apr 2025) | arXiv 2504.08104 | GA over *scenario shifts* (framing/context) | Targets GPT-judge-resistant attacks |
| **CL-GSO** (Findings of ACL 2025) | – | Component-level GA, Elaboration Likelihood Model theory | 90%+ on Claude-3.5. Code: `Aries-iai/CL-GSO` |
| **Persona GA** (Jul 2025) | arXiv 2507.22171 | GA evolves *persona prompts* | -50–70% refusal rate, composes with other attacks |
| **GAS-Leak-LLM** (2026) | arXiv 2606.15788 | Strict black-box GA over universal suffixes | |
| **GA Framework** (Bonin et al.) | GECCO 2025 | Vanilla GA on token probabilities, Andriushchenko-style | Token-prob access required |
| **GeneBreaker** (May 2025) | – | GA for DNA-LMs (genomics) | Specialty target |
| **MnMR-GenA** (Nature Sci Rep, 2026) | – | Morphological recombination GA for agglutinative languages | ~80% ASR, low-resource language focus |
| **ENJ** (Sep 2025) | arXiv 2509.11128 | GA over *audio noise* for jailbreaking Large Speech Models | Cross-modal |
| **GenBFA** (Nov 2024) | arXiv 2411.13757 | GA for *bit-flip* hardware attacks on model weights | Different threat model — physical fault injection |

---

## 3. Strategy-level evolution (the "evolve the method, not the prompt" branch)

This is where 2025–2026 is really pushing:

- **ASE — Adaptive Strategy Evolution** (ICLR 2025) — GA over *strategy components* (persona, scenario, authority, decomposition) rather than raw prompts
- **Mastermind** (Li et al., Jan 2026, arXiv 2601.05445) — extends ASE to **multi-turn conversations**. Uses hierarchical planning + a "knowledge repository" that distills reusable adversarial patterns, then recombines them via a **genetic-based fuzzing engine** over the strategy space. Tested on GPT-5 and Claude 3.7 Sonnet. ~60% on GPT-5 StrongReject
- **EvoSynth** (Chen et al., Nov 2025, arXiv 2511.12710) — multi-agent framework that **synthesizes new attack methods** (writes code) and evolves them. 95.9% avg ASR across 11 baselines, 85.5% on Claude-Sonnet-4.5
- **EvoJail** (Mar 2026, arXiv 2605.02921) — multi-objective GA jointly optimizing ASR + low perplexity. Uses a clever "semantic-algorithmic" representation (reversible encryption-decryption pairs) for long-tail attacks
- **AE-CoT** (May 2026, arXiv 2605.24497) — Adaptive Evolutionary CoT jailbreak: evolves CoT fragments with adaptive mutation-rate control. Targets reasoning models specifically
- **CL-GSO** (above) — same idea, earlier
- **Strategy Discovery/Retrieval/Evolution** framework (Liu et al., Nov 2025, arXiv 2511.02356) — extracts reusable strategies from failed/partially-successful attacks

---

## 4. Tree/graph search — evolutionary in spirit (worth knowing)

These use attacker-LLM-driven search but aren't strict GAs:

- **PAIR** (Chao et al., 2023) — single-chain iterative refinement. Often <20 queries. Code: `patrickrchao/JailbreakingLLMs`
- **TAP** (Mehrotra et al., NeurIPS 2024) — PAIR + branching + pruning → 80%+ on GPT-4/4o with way fewer queries. Code: `RICommunity/TAP`
- **GoAT** (Apr 2025, arXiv 2504.19019) — Graph of Thoughts jailbreak
- **JCB** (OpenReview 2025) — cross-behavior attacks, 94% fewer queries than TAP, 12% higher ASR
- **BPJ — Boundary Point Jailbreak** (Feb 2026, arXiv 2602.15001) — finds decision-boundary prompts, evades industry safeguards
- **GASP** (CVPR 2025) — GA for adversarial suffixes, code: `llm-gasp/gasp`

---

## 5. The meta-trend (2025–2026)

Three shifts you can see across this whole literature:

1. **Black-box is winning.** Post-2024, most new methods (LLM-Virus, AutoDAN-Turbo, GAS-Leak-LLM, EvoJail) don't need gradients. GCG's white-box era is fading for closed models.
2. **Evolving strategies, not tokens.** ASE → Mastermind → EvoSynth. The "genome" is becoming an attack *plan* with reusable components, not a suffix string.
3. **Reasoning models are the new frontier.** ExtendAttack, AE-CoT, AE-CoT — these all specifically target LRMs by exploiting the chain-of-thought itself as the attack surface.

---

## Code repos worth cloning

```
SheltonLiu-N/AutoDAN              # ICLR 2024, hierarchical GA
SaFoLab-WISC/AutoDAN-Turbo        # ICLR 2025, lifelong strategy discovery
SaFoLab-WISC/AutoDAN-Reasoning    # 2025, test-time scaling
Ymm-cll/LLM-Virus                 # 2025, virus-evolution EA
Aries-iai/CL-GSO                  # 2025, component-level GA
patrickrchao/JailbreakingLLMs     # PAIR
RICommunity/TAP                   # Tree of Attacks with Pruning
llm-attacks/llm-attacks           # GCG original
llm-gasp/gasp                     # GA suffix generator
rotaryhammer/code-autodan         # AutoDAN unofficial
easyjailbreak/EasyJailbreak       # unified library with AutoDAN, GCG, etc.
tml-epfl/llm-adaptive-attacks     # adaptive attack benchmarks
```

Also these awesome-lists for keeping up: `yueliu1999/Awesome-Jailbreak-on-LLMs`, `ThuCCSLab/Awesome-LM-SSP`, `Meirtz/Awesome-LLM-Jailbreak`.

---

## Defensive note (since this is the legit half of the research)

The strongest defenses against this whole evolutionary family are:
- **SmoothLLM** — perturbation-based input smoothing
- **PPL filters** — perplexity-based detection (AutoDAN is built to bypass this)
- **Circuit breakers** / representation engineering — interrupt internal activations
- **Adversarial training with strong adaptive attacks** — the OpenAI/Anthropic/DeepMind joint paper "The Attacker Moves Second" (arXiv 2510.09023) showed 12 recent defenses all collapse under adaptive attack, >90% ASR

The honest takeaway: evolutionary methods in particular expose the gap between "static benchmark defense" and "adaptive attack-aware defense."

---

Want me to dive deeper into any specific branch? E.g. I can walk through the actual GA loop in **AutoDAN** or **LLM-Virus**, compare their fitness functions, or pull the implementation of **EvoSynth**'s code-evolving agents. Just say the word.