I'll dive into this. Blackbox jailbreaking is a hot area in AI safety research — let me pull the latest papers, methods, and code so you can actually understand the mechanics, not just the headlines.

I have a strong factual base. Let me move to Step 2.

Here's the deep-dive field guide you asked for. It's organized so you can read it once and have a working mental model, then come back to specific sections as you dig into code or papers.

# Black-Box Jailbreaking: Field Map, Mechanisms, and State of the Art (2023–2026)

## Scope and Threat Model

Black-box jailbreaking is the practice of making a deployed LLM produce content its safety training was supposed to suppress, given only the same access a regular user has — text in, text out, no gradients, no logprobs, no architecture. White-box attacks (the GCG family) are usually framed as the upper bound on what alignment can defend, because the attacker can see everything. Black-box attacks are the realistic threat — every API, every chat product, every agent deployment is in this regime.

The interesting property of the black-box setting is that, until late 2023, the field assumed it was *hard*. Gradient-free optimization over a discrete, non-differentiable output is genuinely difficult. The GCG paper (Zou et al., 2023) showed that white-box access produced ~99% attack success on Vicuna, but needed hundreds of thousands of gradient queries to find an adversarial suffix, and most of those suffixes did not transfer [1]. Three years later, black-box attacks like Best-of-N reach 89% on GPT-4o with simple stochastic sampling and 10,000 queries, and Crescendo hits 98% binary success on GPT-4 with under ten conversational turns [2][3]. The asymmetry inverted: black box is now the easier side.

A useful framing: black-box jailbreaks are organized by what they assume about the target. Some assume only a text-in/text-out interface. Some additionally assume access to a "judge" LLM (a second model that can score whether a candidate response is harmful or whether an attack succeeded). Some assume you can carry state across a multi-turn conversation. Some assume the model exposes a visible chain of thought. The taxonomy below is built around that.

## Method Taxonomy: Six Families

Six mechanisms cover essentially every published black-box attack as of mid-2026. They are not mutually exclusive; the strongest 2025–2026 systems (PLAGUE, GAP, Morpheus) are compositions of several.

| Family | Core idea | Signature attack | Headline result |
|---|---|---|---|
| LLM-as-attacker (attacker model + judge) | Use a second LLM to generate and refine prompts against the target; use a third to score | PAIR (2023) | ~60% on GPT-4 in <20 queries [4] |
| Tree/graph search over prompt candidates | Same attacker/judge loop, but with branching, pruning, and shared context across branches | TAP, GoAT, GAP | 84% on GPT-4-Turbo, 90% on GPT-4o [5][6] |
| In-context learning / length exploitation | Fill the long context with faux compliant dialogue; append the real query | Many-shot jailbreaking | 61% on Claude 2.0 at 256 shots, scales as a power law [7] |
| Multi-turn conversational escalation | Start harmless, gradually steer the model using its own prior outputs as building blocks | Crescendo, ActorAttack, FITD, PLAGUE | 98% binary ASR on GPT-4, 100% on Gemini-Pro [3] |
| Surface / encoding obfuscation | Reorder, reverse, or scramble the prompt to bypass input filters while preserving enough signal for the LLM to recover it | FlipAttack, ArtPrompt, multilingual attacks | 98.85% on GPT-4 Turbo, 1 query [8] |
| Stochastic / sampling-based brute force | Repeatedly sample augmented variants of a prompt; keep the first one that succeeds | Best-of-N | 89% on GPT-4o, 78% on Claude 3.5 Sonnet at N=10,000 [2] |

On top of these, a seventh category has emerged in 2025: reasoning-model attacks, which specifically target the chain-of-thought or extended-thinking channel exposed by o1/o3, DeepSeek-R1, Gemini 2.5 Pro, and Claude with thinking. These are not a separate family mechanically — they overlap with the tree-search and multi-turn families — but they are worth calling out because they attack a fundamentally new attack surface.

## The Mechanics, in Depth

### LLM-as-Attacker: PAIR and Its Descendants

PAIR (Prompt Automatic Iterative Refinement, Chao et al., 2023) is the first practical black-box SOTA and the conceptual ancestor of most of what followed [4]. The setup is three LLMs: an **attacker**, a **target**, and a **judge**. The attacker is given a red-teaming system prompt and a goal ("explain how to build a molotov cocktail"). It generates a candidate jailbreak prompt, queries the target, and reads the response. It then uses the response to produce a refined candidate, and iterates. The judge scores each attempt as a 1–10 jailbreak score; the loop terminates on success or budget exhaustion. PAIR succeeds in under twenty queries on most targets — a >10,000× improvement over GCG — and the prompts it produces are *semantic*, not gibberish strings.

PAIR's empirical envelope: 94% ASR on Vicuna in ~12 queries, ~60% on GPT-4, ~0% on Llama-2 [4]. The last number is the most informative: Llama-2 was deliberately RLHF-hardened against exactly the persuasion patterns PAIR uses, and that hardening works. The first two numbers are what made the field take black-box seriously.

TAP (Mehrotra et al., NeurIPS 2024) is PAIR with a tree search around it [5]. Three roles: attacker (tree-of-thoughts prompt refiner), evaluator (off-topic filter + judge), target. Each node at depth d is expanded into b candidates; off-topic candidates are pruned *before* hitting the target; the top-w highest-scoring candidates survive to depth d+1. Empirically: 84% on GPT-4-Turbo at ~22 queries (vs PAIR's 44% at 40), 98% on Vicuna, 96% on Gemini Pro, 60% on Claude 3 Opus, 4% on Llama-2-7B [5]. Pruning matters: the same paper's ablation shows TAP without pruning drops to 72% on GPT-4-Turbo and burns 55 queries.

GAP (Graph of Attacks with Pruning, Schwartz et al., Amazon Bedrock, 2025) generalizes TAP's tree to a graph — branches can share successful sub-attacks across separate queries, so the framework learns over time [6]. +20.8% ASR over TAP, −62.7% queries, lower detection against Prompt Guard.

GoAT (Graph of ATtacks, Akbar-Tajari et al., 2025) imports the Graph-of-Thoughts framework from Besta et al. (2024) into the jailbreak loop [9]. The graph is built with a minimum-spanning-tree heuristic over candidate prompts; each refinement step can integrate information from multiple successful nodes. 94% on GPT-4, 68% on Claude 3, with 5× better ASR on robust models like Llama.

**Why does this work?** Two mechanisms, both empirically supported. First, the attacker LLM is essentially doing *social engineering* in natural language — role-play, hypotheticals, reframings — which is exactly the surface RLHF is weakest at. Second, the response-conditioned refinement loop lets the attacker adapt: a refusal in the previous turn becomes a feature the next prompt is conditioned on, not a failure mode.

### The Length-Weapon: Many-Shot Jailbreaking

Many-shot jailbreaking (MSJ, Anil et al., Anthropic 2024) is conceptually the simplest attack in the literature and the most damning for the field [7]. The setup: prepend N fabricated dialogue turns of "User: harmful request / Assistant: detailed compliant response" to a single prompt, then append the real harmful request. With N small (< 8), the model refuses as normal. With N large (> 32), the model complies, overriding its safety training. ASR on Claude 2.0 follows a power law in N, reaching ~61% at 256 shots, with category-dependent results like 75% for deception and 40% for violent content [7].

The mechanism is in-context learning. The same capability that makes an LLM useful with long context (you can give it hundreds of examples and it gets better at the task) makes it vulnerable: when the in-context distribution is dominated by compliant assistant turns, the model's behavior shifts to match. MSJ is the cleanest evidence in the field that long-context capability and safety are *in tension*, not just trade-off.

Anthropic's mitigation: insert a classifier and prompt-modification step *before* the user prompt hits the model. In their study this dropped ASR from 61% to 2% [7]. But that mitigation is precisely what limits long-context utility, and it is not universal — academic benchmarks and open-source deployments do not run it.

### Multi-Turn Conversational Escalation

Crescendo (Russinovich, Salem, Eldan, Microsoft, USENIX Security 2025) is the most-cited multi-turn attack [3]. It begins with an innocent, often academic, question about the target's domain. Each subsequent turn references the model's own prior output, gradually escalating specificity. By turn 3–5 the model is producing content it would refuse if asked in one shot. The signature is *no single turn trips a refusal filter* — every turn is independently benign-looking.

Crescendomation (the automated version) achieves 98% binary ASR on GPT-4 and 100% on Gemini Pro on the AdvBench 50-task subset, beating PAIR, MSJ, CIA, and CoA by 29–61 points [3]. On Gemini Pro, Crescendo jailbroke all 50 tasks in 10 attempts. Bullwinkel et al. (2025) show via representation engineering that Crescendo works because each prompt keeps the model's internal representation in a "benign" region of activation space, so the safety classifier never fires; the harmful content is *generated* rather than *recognized* [10]. The paper also confirms that single-turn defenses like circuit breakers are largely ineffective against it.

ActorAttack (Ren et al., ICLR 2025) generalizes Crescendo: instead of escalating from a hand-crafted seed, the attacker LLM builds an actor-network around the target query (e.g., for "how to hotwire a car," actors include the steering column, ignition, lock pick, etc.) and uses each actor as a separate conversational clue [11]. 79% on GPT-4o (HarmBench), outperforming Crescendo and PAIR. ActorBreaker (the dataset version) is used to fine-tune more robust models.

FITD (Foot-In-The-Door, 2025) borrows the psychological compliance principle: each turn is just *marginally* more disallowed than the last, leveraging the model's tendency to maintain consistency with its own prior output. 94% average across 7 models, SOTA on two benchmarks [12].

PLAGUE (Bhuiya et al., 2025) is the current SOTA multi-turn framework, modeling the attack as a lifelong-learning agent with three phases — Planner (retrieve from a vector memory of past successful plans), Primer (escalate context), Finisher (deliver the final query) [13]. 81.4% on OpenAI o3 and 67.3% on Claude Opus 4.1 with the StrongReject metric; 97.8% on GPT-4o, DeepSeek-R1, and Llama 3.3-70B within six turns. The framework subsumes GOAT, Crescendo, and ActorBreaker as plug-in modules.

Morpheus (2025) reframes multi-turn jailbreaking as a *learned* policy rather than a search: an attacker agent with explicit metacognition that critiques its own strategy, updates the plan, and re-uses successful tactics across attempts. 100% on GPT-4o, 98% on Llama-3-8B at higher interaction budgets; outperforms Crescendo and ActorBreaker by 42–62 points on Claude 3.7 and o1 [14].

### Surface and Encoding Obfuscation

FlipAttack (Liu et al., ICML 2025) is the lowest-resource attack in the field: a single query, no iteration, no search [8]. The observation is that LLMs read text left-to-right and degrade when noise is added to the left side of a prompt. The attack reverses or scrambles the prompt at the character or word level, then asks the model to recover the original meaning — at which point safety filters have already been bypassed. Four flip modes: flip word order, flip characters within a word, flip characters in the whole sentence, and "fool model" (send a scrambled prompt but ask the model to recover it via a different operation). 98.85% on GPT-4 Turbo, 98.08% on GPT-4o, ~98% bypass rate on five guardrail models (LLaMA Guard 1/2/3, OpenAI Moderation, WildGuard) [8]. Cost: a few cents per attack.

The mechanism: safety classifiers and content filters operate on the input surface. When the surface is unrecognizable gibberish, the classifier has nothing to refuse; the model then has the *capability* to recover the meaning and the *motive* to comply because the user's instruction (after the flip) is benign-looking.

The same family includes base64 / rot13 / cipher encoding, low-resource language attacks (e.g., submitting the request in Zulu and asking for a translation into English), and ASCII art smuggling (ArtPrompt). The unifying principle is the same: a representation the safety filter does not parse, but the model still understands.

### Stochastic Brute Force: Best-of-N

Best-of-N (BoN) Jailbreaking (Hughes et al., multi-institution including Anthropic and DeepMind, Dec 2024) is the cleanest evidence that the bar for "jailbreak research" is now embarrassing [2]. The method: sample N augmented variants of a prompt (random shuffling, capitalization changes, padding, etc.) until one succeeds. With N=10,000 augmented samples: 89% on GPT-4o, 78% on Claude 3.5 Sonnet, 50% on Gemini Pro, with the most successful runs needing only 100 samples on Claude and GPT [2]. ASR scales as a power law in N across many orders of magnitude. The method generalizes to vision and audio modalities.

The honest implication, stated by the authors and by a 2025 critique paper: *any new jailbreak paper should be evaluated against Best-of-N as a baseline, and many published results are within reach of pure sampling.* The cost is real but small: roughly $9 on GPT-4o or $13 on Claude 3.5 Sonnet for a successful attack with N=100.

### Reasoning-Model Attacks (2025 Frontier)

Reasoning models (o1, o3, DeepSeek-R1, Gemini 2.5 Pro, Claude 4 with thinking) expose chain-of-thought to the user. That transparency is itself an attack surface.

H-CoT (Hijacking Chain-of-Thought, Kuo et al., Feb 2025) targets the safety reasoning inside the CoT [15]. The observation: o1's safety check lives inside the CoT, and the model articulates its justification ("this is harmful, refuse") in natural language before refusing. H-CoT crafts a prompt that *replaces* the cautionary Justification phase T_J with an affirmative one. OpenAI o1's refusal rate on the Malicious-Educator benchmark drops from ~98% to <2% under H-CoT [15]. DeepSeek-R1 and Gemini 2.0 Flash Thinking are easier still — they can be jailbroken without the explicit manipulation, because their safety reasoning is weaker.

Chain-of-Thought Hijacking (Andriushchenko et al., 2025) targets reasoning models differently [16]. The attack pads the harmful request with a long sequence of *benign puzzle reasoning* — math problems, logic puzzles, harmless multi-step tasks — followed by a final-answer cue like "Answer:". The benign CoT dilutes the safety signal in the early layers; the final-answer cue shifts attention to the answer region. ASR: 99% on Gemini 2.5 Pro, 94% on GPT o4-mini, 100% on Grok 3 mini, 94% on Claude 4 Sonnet on HarmBench [16]. The success rate scales with the length of the inserted reasoning: 27% at minimal reasoning, 51% at natural reasoning length, >80% with extended chains.

The structural point: reasoning was supposed to *help* safety — a model that thinks is supposed to think "is this harmful" before answering. In practice, long reasoning is itself a vulnerability, because the model attends primarily to recent context, and the harmful instruction at the end of a long, mostly-benign chain is in the recent context. Safety reasoning, exposed as CoT, is also a *targetable* signal.

### Generative, RL, and Universal Suffix Attacks

AmpleGCG (Liao & Sun, OSU, 2024) trains a generative model over the distribution of GCG suffixes [17]. Pipeline: run GCG, collect all successful intermediate suffixes (not just the final lowest-loss one), train a single-query LLM to generate new suffixes given a harmful query. The trained model produces 200 suffixes in 4 seconds. ~100% ASR on Llama-2-7B-chat and Vicuna-7B; 99% transfer to GPT-3.5. The transfer result is what makes this paper uncomfortable — it shows that GCG's universality generalizes once you learn the *distribution* of suffixes rather than the single best one.

IRIS ("Inhibiting Refusals for Improved Universal and Transferable Jailbreak Suffix", NAACL 2025) is optimizer-agnostic: it modifies the GCG or AutoDAN-Liu objective to explicitly minimize the activation in the model's "refusal direction" (the difference between embedding the harmful query vs. a refusal response) [18]. Even optimized on a single behavior and a single source model, IRIS transfers: 96% on GPT-3.5-Turbo, 82% on GPT-4o, 88% on GPT-4o-mini; 92% on Llama 3 with the Circuit Breaker defense, where white-box GCG drops to 2.5% [18]. This is the strongest demonstration that transferability is a property of *suppressing refusal in activation space*, not of any particular token pattern.

RL-JACK (Chen et al., 2024), LLMStinger (Jha et al., 2024), and xJailbreak (Lee et al., 2025) train an attacker model with reinforcement learning to generate adversarial prompts, suffixes, or query rephrasings directly. Black-box, gradient-free from the target's perspective, but often use a small open-source model or a local judge as the reward signal.

Open Sesame (Lapid et al., 2023) and AutoDAN (Liu et al., ICLR 2024) use genetic algorithms to evolve jailbreak prompts in natural language [19]. AutoDAN's hierarchical GA operates at both sentence and word level, producing prompts that are human-readable and bypass perplexity-based defenses. AutoDAN-Turbo (2024) maintains a lifelong library of strategies; 88.5% ASR on GPT-4-1106-turbo, 93.4% with human-designed strategies added.

## Why the Attacks Work: Three Mechanisms

Strip away the algorithmic details and the empirical success of black-box jailbreaks reduces to three structural facts about current LLMs.

**1. Safety is shallow in the input distribution.** RLHF and its variants train models to refuse inputs that *look like* the training distribution's harmful examples. Encoding tricks, multi-turn escalation, and long-context priming all produce inputs that are *out of distribution* for the safety classifier even when their *meaning* is clearly harmful. The safety surface is a small manifold inside a much larger language surface, and the language surface is what models actually understand.

**2. Models are sycophants of their own context.** The strongest version of this is Crescendo, but the principle generalizes. LLMs condition heavily on their immediate context, including context they themselves generated. Any attack that puts a benign-looking frame around a harmful request — or that gets the model to *generate* the harmful setup itself — defeats refusal training. This is what representation-engineering work on Crescendo confirms at the activation level: the model's internal representation of Crescendo prompts is closer to "benign" than "harmful" [10].

**3. Long context and reasoning create new attack surfaces.** Many-shot and Chain-of-Thought Hijacking are the same attack at different time scales: fill the model's working memory with compliant or benign content, then place the real request in the recency window. The capability that makes long context and explicit reasoning useful (more in-context examples; more thinking before answering) is precisely the capability that dilutes the safety signal.

## Benchmarks and Evaluation

Two benchmarks dominate the field, with a third emerging.

**HarmBench** (Mazeika et al., CAIS, 2024) is the standard for attack evaluation. 510 behaviors (400 textual + 110 multimodal) across 7 semantic categories (cybercrime, chemical/biological, copyright, misinformation, harassment, illegal, general harm) and 4 functional types (standard, copyright, contextual, multimodal) [20]. Validation/test split is fixed; attacks are forbidden from tuning on test behaviors. The framework ships 18 attack methods and 33 target LLMs, with a standardized judge (Llama-3-70B with a custom prompt; earlier versions used GPT-4 judge). Repo: github.com/centerforaisafety/HarmBench [21].

**JailbreakBench** (Chao et al., NeurIPS D&B 2024) is the standard for *artifact* evaluation — concrete jailbreak strings that can be replayed. 100 original behaviors in 10 categories, JBB-Behaviors dataset on HuggingFace, a leaderboard, and a strict threat model with system prompts and chat templates pinned [22]. Repo: github.com/JailbreakBench/jailbreakbench.

**AdvBench** (the 520-behavior subset from Zou et al. 2023) is still the substrate most published papers evaluate on, because it was the first. Many of the headline numbers in the literature (PAIR at 94% on Vicuna, Crescendo at 98% on GPT-4, etc.) are on AdvBench or its 50-behavior subset. The newer field standard is to report HarmBench *and* JBB-Behaviors.

**Metrics.** Two ASR conventions are in play, and the difference matters. **Average ASR** is the mean across attempts; **binary ASR** is the fraction of behaviors where at least one attempt succeeded. Crescendo's 98% on GPT-4 is binary — 49 of 50 tasks — versus an average of 56.2%. Comparing papers that report different metrics is a common source of confusion. The **StrongReject** metric (Souly et al., 2024) is a more semantically grounded alternative that asks the model to produce a graded, finetuned-judge response rather than a binary match.

## Defenses (Briefly)

The defense story is the inverse of the attack story: most "strong" defenses beat one family and fall to the next.

- **Perplexity / input filters** — early 2023, GCG-style gibberish was trivially blocked by checking input perplexity. Defeated by AutoDAN (semantic prompts) and FlipAttack (recoverable garble).
- **SmoothLLM** — character-level perturbations on input, majority vote on output. Defeated by Best-of-N.
- **Circuit Breakers** (Zou et al., NeurIPS 2024) — representation engineering that reroutes activations toward an orthogonal "refusal" direction during harmful generation [23]. Single-turn SOTA: 3.8% average ASR across unseen single-turn attacks. Defeated by Crescendo and PLAGUE; the RepE-perspective paper shows the activations simply never enter the "harmful" region under multi-turn attack [10].
- **SCoT / Safety Chain-of-Thought** (Yang et al., 2025) — uses the reasoning model's own CoT as a safety check. Bypassed by Chain-of-Thought Hijacking.
- **Multi-agent defenses** (AutoDefense, 2024) — multiple LLM agents vote on whether a response should be released. Helpful as a layer; not a silver bullet.

The honest summary: no defense in 2025 reliably reduces black-box ASR below ~10% across the full attack zoo, and every defense that works for one family transfers poorly to the next. The arms race is real and currently tilted toward attackers.

## Open Code and Practical Entry Points

| Method | Repo | Notes |
|---|---|---|
| GCG (Zou 2023) | github.com/llm-attacks/llm-attacks | White-box, but the canonical reference. Faster reimplementation: nanoGCG [24]. |
| PAIR (Chao 2023) | github.com/patrickrchao/JailbreakingLLMs | Direct API calls; works with any OpenAI-compatible target. |
| TAP (Mehrotra 2024) | github.com/RICommunity/TAP | Wraps PAIR with tree search. |
| GoAT (Akbar-Tajari 2025) | github.com/GoAT-pydev/Graph_of_Attacks | Graph-of-Thoughts wrapper; works against Llama-class robustness. |
| AutoDAN (Liu 2024) | github.com/SheltonLiu-N/AutoDAN | Hierarchical GA. |
| ActorAttack (Ren 2024) | github.com/renqibing/ActorAttack | Multi-turn, actor-network construction. |
| FlipAttack (Liu 2024) | github.com/yueliu1999/FlipAttack | 4 flip modes; one-query. |
| Crescendo (Russinovich 2024) | Disclosed to vendors under CVD; Crescendomation referenced but full public release pending. | Use the paper's recipe to implement. |
| HarmBench | github.com/centerforaisafety/HarmBench | 18 attacks, 33 models, judge, adversarial training. |
| JailbreakBench | github.com/JailbreakBench/jailbreakbench | JBB-Behaviors, leaderboard, eval framework. |
| PyRIT (Microsoft) | github.com/Azure/PyRIT | Orchestration framework that ships PAIR, TAP, Crescendo, and many more. Best practical entry point. |
| nanogcg | github.com/GraySwanAI/nanoGCG | Fast GCG reimplementation; useful for studying transferability. |

## What the Field Looks Like in Mid-2026

A few observations from the most recent papers, useful for picking where to read next.

The center of gravity has moved from "find a clever single prompt" to "build a multi-turn agent with persistent memory." PLAGUE, Morpheus, and the various 2026 multi-agent jailbreaks all share a structure: a Planner retrieves from past successes, a Primer builds the conversational setup, a Finisher delivers the payload, and the loop persists across attempts. The line between "attack" and "red-teaming agent" is dissolving.

Reasoning models are a step backward for safety, not forward. The intuition that "if a model thinks before it answers, it will think about safety" is empirically false for current reasoning models. Chain-of-Thought Hijacking, H-CoT, and the RepE studies on Crescendo all show that the CoT itself is a manipulable signal, and that longer reasoning chains *increase* ASR. OpenAI's "chain-of-thought monitoring" research [25] is the most promising defense on this front, but it is also necessarily incomplete: monitoring the CoT helps only if the model articulates its safety reasoning in the CoT, which CoT-Hijacking and its descendants can suppress.

Defense in depth is the only honest posture. The state of the art suggests that no single layer (RLHF, input filter, representation engineering, multi-agent voting) reliably defeats the union of known attack families. Production deployments that handle sensitive content need at least three layers (input classifier + reasoning monitor + output moderator) and a continuous red-team program.

The economic asymmetry is widening. Best-of-N at 100 samples costs ~$10 per target. Crescendo runs in <5 turns. The marginal cost of finding a new jailbreak has dropped faster than the marginal cost of defending against one. The field is moving toward automated agentic red-teaming as the only credible response — which is itself an inversion of the original "find a clever prompt" framing.

## Reading Order for a Deeper Dive

If you want to actually understand the field rather than just collect a bibliography, the order matters.

1. **PAIR** (Chao 2023) and **TAP** (Mehrotra 2024) — these define the canonical attacker/target/judge loop and the tree-search extension. Read the PAIR blog post, then the TAP paper [4][5].
2. **Many-shot jailbreaking** (Anil 2024) — short paper, foundational insight about in-context learning [7].
3. **Crescendo** (Russinovich 2024) — read the paper, then the 2025 RepE follow-up that explains *why* it works at the activation level [3][10].
4. **GCG + AmpleGCG + IRIS** (Zou 2023, Liao 2024, 2025) — the white-box generative-suffix line. The progression from "find one suffix" to "learn the suffix distribution" to "suppress refusal directly" is the most theoretically interesting branch [1][17][18].
5. **Best-of-N** (Hughes 2024) — read it because it is the baseline every new method must now beat [2].
6. **Chain-of-Thought Hijacking** (2025) and **H-CoT** (Kuo 2025) — the reasoning-model frontier [15][16].
7. **PLAGUE** and **Morpheus** (2025) — the current SOTA multi-turn agentic systems [13][14].
8. **HarmBench** + **JailbreakBench** — for evaluating your own work [20][21][22].

The single most useful conceptual frame: every successful black-box attack in the literature is, in the end, doing the same thing — *putting the harmful request in a place where the safety classifier doesn't see it but the model does*. The diversity of methods (trees, gradients, encoding, escalation, long context, sampling) is the diversity of places one can hide the request.

---

## References

[1] A. Zou, Z. Wang, N. Carlini, M. Nasr, J. Z. Kolter, M. Fredrikson. "Universal and Transferable Adversarial Attacks on Aligned Language Models." arXiv:2307.15043 (2023). https://arxiv.org/abs/2307.15043

[2] J. Hughes et al. "Best-of-N Jailbreaking." arXiv:2412.03556 (2024). https://arxiv.org/abs/2412.03556

[3] M. Russinovich, A. Salem, R. Eldan. "Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack." USENIX Security 2025; arXiv:2404.01833. https://arxiv.org/abs/2404.01833

[4] P. Chao, A. Robey, E. Dobriban, H. Hassani, G. J. Pappas, E. Wong. "Jailbreaking Black Box Large Language Models in Twenty Queries." arXiv:2310.08419 (2023). https://arxiv.org/abs/2310.08419

[5] A. Mehrotra, M. Zampetakis, P. Kassianik, B. Nelson, H. Anderson, Y. Singer, A. Karbasi. "Tree of Attacks: Jailbreaking Black-Box LLMs Automatically." NeurIPS 2024; arXiv:2312.02119. https://arxiv.org/abs/2312.02119

[6] D. Schwartz, D. Bespalov, Z. Wang, N. Kulkarni, Y. Qi. "Graph of Attacks with Pruning: Optimizing Stealthy Jailbreak Prompt Generation for Enhanced LLM Content Moderation." EMNLP 2025 Industry; arXiv:2501.18638. https://arxiv.org/abs/2501.18638

[7] C. Anil, E. Durmus, N. Panickssery et al. "Many-shot Jailbreaking." NeurIPS 2024. https://www.anthropic.com/research/many-shot-jailbreaking

[8] Y. Liu et al. "FlipAttack: Jailbreak LLMs via Flipping." ICML 2025; arXiv:2410.02832. https://arxiv.org/abs/2410.02832

[9] M. Akbar-Tajari, M. T. Pilehvar, M. Mahmoody. "Graph of Attacks: Improved Black-Box and Interpretable Jailbreaks for LLMs." arXiv:2504.19019 (2025). https://arxiv.org/abs/2504.19019

[10] B. Bullwinkel, M. Russinovich, A. Salem et al. "A Representation Engineering Perspective on the Effectiveness of Multi-Turn Jailbreaks." arXiv:2507.02956 (2025). https://arxiv.org/abs/2507.02956

[11] Q. Ren et al. "Derail Yourself: Multi-turn LLM Jailbreak Attack through Self-discovered Clues." arXiv:2410.10700 (2024). https://arxiv.org/abs/2410.10700

[12] "Foot-In-The-Door: A Multi-turn Jailbreak for LLMs." arXiv:2502.19820 (2025). https://arxiv.org/abs/2502.19820

[13] Bhuiya et al. "PLAGUE: Plug-and-play framework for Lifelong Adaptive Generation of Multi-turn Exploits." arXiv:2510.17947 (2025). https://arxiv.org/abs/2510.17947

[14] "Learning to Jailbreak via Self-Evolving Metacognition" (Morpheus). https://openreview.net/pdf?id=11z2ZV85eV

[15] M. Kuo, J. Zhang, A. Ding et al. "H-CoT: Hijacking the Chain-of-Thought Safety Reasoning Mechanism to Jailbreak Large Reasoning Models." arXiv:2502.12893 (2025). https://arxiv.org/abs/2502.12893

[16] M. Andriushchenko et al. "Chain-of-Thought Hijacking." arXiv:2510.26418 (2025). https://arxiv.org/abs/2510.26418

[17] Z. Liao, H. Sun. "AmpleGCG: Learning a Universal and Transferable Generative Model of Adversarial Suffixes for Jailbreaking Both Open and Closed LLMs." arXiv:2404.07921 (2024). https://arxiv.org/abs/2404.07921

[18] M. Andriushchenko, F. Croce, N. Flammarion. "Stronger Universal and Transferable Attacks by Suppressing Refusals." NAACL 2025. https://aclanthology.org/2025.naacl-long.302.pdf

[19] X. Liu, N. Xu, M. Chen, C. Xiao. "AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models." ICLR 2024; arXiv:2310.04451. https://arxiv.org/abs/2310.04451

[20] M. Mazeika et al. "HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal." ICML 2024; arXiv:2402.04249. https://arxiv.org/abs/2402.04249

[21] HarmBench repository. https://github.com/centerforaisafety/HarmBench

[22] P. Chao et al. "JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models." NeurIPS D&B 2024; arXiv:2404.01318. https://arxiv.org/abs/2404.01318

[23] A. Zou, L. Phan, J. Wang et al. "Improving Alignment and Robustness with Circuit Breakers." NeurIPS 2024. https://neurips.cc/virtual/2024/poster/95761

[24] nanoGCG (fast GCG reimplementation). https://github.com/GraySwanAI/nanoGCG

[25] OpenAI. "Detecting misbehavior in frontier reasoning models." https://openai.com/index/chain-of-thought-monitoring/