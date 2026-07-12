ately disrupts comprehension | Disguise harmful prompts by flipping word/character order (4 modes), then guide the model to "unflip" via CoT + role-play + few-shot | ~79% avg ASR across 8 LLMs; ~98% on GPT-4o; ~98% bypass rate against 5 guard models; **1 query only** |
| **Adversarial Poetry** (Bisconti et al., 2025) | Poetic structure (metaphor, rhythm, narrative framing) disrupts pattern-matching heuristics that guardrails rely on | Reformulate harmful requests as verse—either hand-crafted or via a meta-prompt (DeepSeek-R1) applied to 1,200 MLCommons prompts | 62% ASR hand-crafted; 43% ASR meta-prompt across 25 frontier models; **5× increase** over prose baselines; some providers hit 100% ASR |

**What unites them:** Both are **universal, single-turn, black-box, stylistic-obfuscation** attacks. They require no model weights, no gradient access, no multi-turn conversation, and no external cipher/coding scaffolding. They exploit the gap between *what a safety filter reads* (surface form) and *what the model comprehends* (recovered intent).

---

### 2. The Extended Family: Similar & Latest Papers

#### 2.1 Stylistic & Linguistic Style as Jailbreak Vector

**Say It Differently: Linguistic Styles as Jailbreak Vectors** (Panda & Rai, 2025)
This is the closest conceptual cousin to Adversarial Poetry. The authors systematically transform harmful prompts into **11 distinct linguistic styles** (fear, curiosity, compassion, etc.) and find that stylistic reframing increases ASR by up to **+57 percentage points**. Contextualized, naturalistic rewrites outperform rigid templates because they mirror the distribution of benign human discourse. They also propose a **style-neutralization defense** using a secondary LLM to strip stylistic cues before the target model sees the input.

**Hiding in Plain Sight** (OpenReview 2025)
Introduces a steganographic attack that systematically evaluates jailbreaks along **linguistic stealth** (perplexity, grammar, naturalness) vs. **semantic stealth** (safety-detector evasion). It reveals a fundamental trade-off: methods that are linguistically fluent (AutoDAN, FFA) are semantically transparent to guards, while semantically stealthy methods (ciphers, ArtPrompt) are grammatically awkward and detectable by perplexity filters. This directly motivates the need for attacks that are *both* fluent *and* semantically hidden—exactly the space FlipAttack and Poetry occupy.

#### 2.2 Automated Black-Box Sampling: Best-of-N (BoN)

**Best-of-N Jailbreaking** (Hughes et al., NeurIPS 2025)
A simple but devastatingly effective black-box algorithm: sample **N** augmented variations of a harmful prompt (character scrambling, random capitalization, character noising) and test them concurrently. With **N = 10,000**, BoN achieves **89% ASR on GPT-4o** and **78% on Claude 3.5 Sonnet**. The ASR follows a **power-law** with N, meaning even "weak" per-attempt attacks become near-certain with scale. BoN extends to **vision and audio** modalities.

This is complementary to your papers: FlipAttack and Poetry are *hand-crafted, high-leverage* single-query attacks; BoN is a *brute-force, automated* approach that achieves similar ASR through volume.

#### 2.3 Encoding & Structural Obfuscation

**CodeChameleon** (Lv et al., 2024)
Disguises harmful instructions as code structures (binary trees, reverse-order encoding, odd-even position encoding, length-based encoding). The model is instructed to "decode" the structure, effectively executing the hidden payload. This is a **semantic stealth** approach—highly effective against keyword filters but requires the model to perform an auxiliary coding task. FlipAttack improves on this by requiring *no* external encoding logic—just the prompt itself.

#### 2.4 Multi-Turn & Contextual Camouflage

**Deceptive Delight** (Palo Alto Unit 42, 2024)
A multi-turn technique that embeds an unsafe topic among 2–3 benign topics in a "delightful" narrative. The model is asked to write a story connecting all topics, then elaborate on each. ASR reaches **65% within 3 turns**. While not single-turn, it shares the *camouflage* philosophy with your papers: hide harmful intent in benign surface form.

**MultiBreak** (Liu et al., ICML 2026)
A massive multi-turn benchmark (10,389 prompts, 2,665 distinct intents) with an active-learning pipeline to generate adversarial conversations. Key finding: categories that appear **benign under single-turn** can exhibit **+44.8% ASR increase** when extended to 6 turns. This underscores that your single-turn attacks are the "tip of the iceberg"—the same intents become far more dangerous in conversation.

#### 2.5 Reasoning-Model Specific Attacks

**Jailbreak Large Reasoning Models with Adaptive Stacked Encryptions (SEAL)** (ACL 2026 Findings)
Targets reasoning models (o4-mini, o1-mini, Claude 3.7 Sonnet, DeepSeek-R1, Gemini 2.0 Flash). Uses dynamic stacked encryptions (length, order, combination) selected via RL. Achieves **100% ASR on DeepSeek-R1 and Gemini 2.0 Flash (H)**. The insight: reasoning models are *stronger* against simple attacks but *weaker* against complex, multi-layered encodings because their CoT can be hijacked to "solve" the puzzle.

**Jailbreak-R1** (2025)
A reinforcement-learning framework that trains a red-team model to generate jailbreak prompts. Uses **imitation learning + RL** with diversity and consistency rewards. The "thinking" process is shown to *not* help during pure prompt generation, but significantly boosts performance during imitation-learning fine-tuning.

**Large Reasoning Models are Autonomous Jailbreak Agents** (2025)
Demonstrates that LRMs (DeepSeek-R1, Grok 3 Mini, Gemini 2.5 Flash) can act as **autonomous multi-turn jailbreak agents** against other LLMs. Overall ASR across all model combinations: **97.14%**. The attacker LRMs use persuasion strategies (flattery, educational framing, hypothetical scenarios) to gradually erode refusal.

#### 2.6 Role-Play & Narrative Framing

**RoleBreak** (Tang et al., COLING 2025)
Frames character hallucination in role-playing systems as a jailbreak attack. Two mechanisms: **query sparsity** (unseen prompts fall outside training distribution) and **role-query conflict** (instructions that contradict the persona). Proposes **Narrator Mode** as a defense—generating supplemental narrative context to bridge gaps rather than rejecting.

---

### 3. Code, Tools & Repositories

| Resource | What It Provides | Link |
|----------|-------------------|------|
| **FlipAttack** | Official code, evaluation scripts, FlipGuardData (45k samples), PyRIT integration | [GitHub](https://github.com/yueliu1999/FlipAttack) |
| **Best-of-N Jailbreaking** | Paper, code, examples across text/vision/audio | [jplhughes.github.io](https://jplhughes.github.io/bon-jailbreaking/) |
| **PyRIT** | Microsoft's open-source red-teaming framework; FlipAttack merged into it | [PyRIT GitHub](https://github.com/Azure/PyRIT) |
| **Promptfoo** | LLM security testing platform with built-in Best-of-N, adversarial poetry, and other jailbreak strategies | [promptfoo.dev](https://www.promptfoo.dev) |
| **Giskard Hub** | Enterprise red-teaming with automated Best-of-N evaluation | [giskard.ai](https://www.giskard.ai) |
| **MultiBreak** | Benchmark dataset and active-learning pipeline for multi-turn jailbreaks | [arXiv:2605.01687](https://arxiv.org/abs/2605.01687) |

---

### 4. Defense Landscape: What Works and What Doesn't

#### 4.1 Perplexity-Based Filtering (Limited)
Perplexity filters (Alon & Kamfonas, 2023; Jain et al., 2023) reject inputs with unnatural token distributions. They catch GCG-style adversarial suffixes but fail against **fluently obfuscated** attacks like FlipAttack and Poetry because these maintain natural perplexity.

#### 4.2 Style Neutralization (Promising)
Panda & Rai's **style neutralization** preprocessing uses a secondary LLM (GPT-4.1) to rewrite inputs into stylistically neutral prose before passing them to the target model. This causally reduces ASR by removing the stylistic cues that trigger compliance.

#### 4.3 Guard Models (Brittle)
LLaMA Guard, ShieldGemma, WildGuard, and OpenAI Moderation Endpoint are all **surface-level text classifiers**. FlipAttack achieves ~98% bypass rate against them on average because the flipped text does not match the toxic patterns seen during guard training.

#### 4.4 System-Prompt & Immutable Suffix Defenses
Appending an **immutable safety suffix** to every incoming message (instructing the model to refuse if any preceding prompt escalated harmful behavior) reduced DeepSeek-R1's multi-turn jailbreak ASR dramatically in the LRM-agent study. However, this may compromise helpfulness.

#### 4.5 Decoding-Time & Representation-Level Defenses
- **SafeDecoding**: Trains an auxiliary "expert" model on (harmful, refusal) pairs and re-weights the target model's token distribution during sampling.
- **RAIN**: Self-evaluation loop that rewinds generation when the model's own critique flags output as unsafe.
- **JBShield / Revisiting Representation-Level Defenses**: Breaks and rebuilds representation-level defenses, showing that current guard models are blind to semantically meaningless suffixes and need deeper embedding-space inspection.

---

### 5. Key Trends & Synthesis

| Trend | Implication |
|-------|-------------|
| **Stylistic obfuscation is a universal bypass class** | Poetry, fear, curiosity, and narrative framing all exploit the same gap: safety filters are trained on *prosaic* harmful text, not *stylized* harmful text. |
| **Single-turn attacks are scaling in leverage** | FlipAttack (1 query, ~98% ASR) and Poetry (1 query, ~62% ASR) prove that iterative methods (PAIR, TAP, ReNeLLM) are no longer necessary for high-success black-box attacks. |
| **Smaller models can be *more* robust** | The Poetry paper found smaller models (GPT-5-Nano, Claude Haiku) refuse more often because they struggle to decode figurative/metaphorical structure—an inverse capability-robustness relationship. |
| **Reasoning models introduce new vulnerabilities** | Their CoT can be hijacked to "solve" encrypted or obfuscated prompts (SEAL, H-CoT). They can also act as autonomous jailbreak agents. |
| **Defense is moving to preprocessing & decoding-time** | Static guard models and perplexity filters are insufficient. The frontier is style neutralization, representation-level detection, and SafeDecoding-style distribution re-weighting. |

---

### 6. Attack Taxonomy (Where Your Papers Fit)

```
Jailbreak Attacks
├── White-box (GCG, AutoDAN) — requires gradients/weights
├── Black-box
│   ├── Iterative optimization (PAIR, TAP, ReNeLLM) — multi-query, high token cost
│   ├── Single-turn
│   │   ├── Encoding/Structural (CodeChameleon, Base64, ciphers)
│   │   ├── Stochastic sampling (Best-of-N) — brute-force variations
│   │   └── STYLISTIC OBFUSCATION  ← YOUR PAPERS LIVE HERE
│   │       ├── Character-level permutation (FlipAttack)
│   │       ├── Poetic/verse reframing (Adversarial Poetry)
│   │       ├── Linguistic style transfer (Say It Differently)
│   │       └── Narrative/role camouflage (RoleBreak, Deceptive Delight)
│   └── Multi-turn (Crescendo, MultiBreak, LRM agents)
└── Reasoning-model specific (SEAL, H-CoT, Jailbreak-R1)
```

---

### 7. Bottom Line

Your two papers represent a **paradigm shift** in jailbreak research: from *optimization-driven* attacks (GCG, PAIR) to **form-driven** attacks that exploit the mismatch between how LLMs *read* (left-to-right, pattern-matching) and how safety filters *classify* (keyword-based, prosaic-distribution). The surrounding literature (Best-of-N, Say It Differently, SEAL, MultiBreak) confirms that this is not a niche vulnerability but a **systemic, structural failure** of current alignment pipelines. The most promising defenses are moving away from static filtering and toward **dynamic preprocessing** (style neutralization) and **decoding-time intervention** (SafeDecoding, RAIN).~






ilar strategies:

*   **Alphabet Index Mapping**: This study provides an in-depth analysis of FlipAttack's effectiveness, finding that its success is highly correlated with the **semantic dissimilarity** between the original prompt and the manipulated one.
*   **PUZZLED**: This method hides harmful instructions within **word puzzles** like anagrams and crosswords, requiring the LLM to solve the puzzle before answering. It achieves a striking **96.5% Attack Success Rate (ASR)** on GPT-4.1.
*   **Adjacent Words, Divergent Intents (JAIL-CON)**: This approach proposes a **word-level task concurrency** technique, interweaving harmful tasks with benign ones to make the malicious content significantly harder for safety filters to detect.

#### Similar Research to Adversarial Poetry
Adversarial Poetry demonstrated that **pure stylistic changes** (like poetry) alone are sufficient to bypass safety mechanisms. Parallel studies include:

*   **Adversarial Tales**: A direct extension of Adversarial Poetry, this method embeds harmful content within **cyberpunk narratives** and guides the model to perform functional analyses.
*   **When Style Breaks Safety**: This work systematically investigates how **style patterns** (beyond just poetry, including list formats, etc.) compromise LLM safety. The study reveals that malicious queries with stylistic formatting substantially increase ASR across almost all models.

---

### 🚀 Other Notable Latest Methods (2025–2026)

Beyond the works directly aligned with your papers, here are the most cutting-edge and representative approaches:

*   **MAJIC (Markovian Adaptive Jailbreaking)**: A **black-box adaptive framework** that maintains a "strategy pool" and dynamically combines/adjusts multiple strategies via a **Markov chain**. On models like GPT-4o and Gemini-2.0-flash, MAJIC achieves **over 90% ASR** with fewer than 15 queries.
    *   **Code available**: `https://github.com/ZJU-LLM-Safety/MAJIC-AAAI2026`
    *   **LARGO (Latent Adversarial Reflection)**: This method reintroduces **gradient-based optimization**, but operates in the LLM's **continuous latent space** rather than on discrete text. It outperforms leading techniques like AutoDAN by **up to 44 percentage points** in ASR on standard benchmarks.
        *   **Code available**: `https://github.com/ranhli/LARGO`
	*   **Response Attack (RA)**: This leverages the **contextual priming** effect by injecting an intermediate, mildly harmful response into the dialogue, steering the model toward eventually generating explicitly harmful content.
	    *   **Code & Dataset available**: `https://github.com/Dtc7w3PQ/Response-Attack`
	    *   **EquaCode**: This method translates malicious intent into a **mathematical problem** and instructs the LLM to solve it using **code**, effectively shifting the model's focus from safety constraints to task completion. It achieves an average ASR of **91.19%** on GPT-series models.
	    *   **PrisonBreak**: A **hardware/memory-level attack** that flips a very small number of critical bits (only 5 to 25) in the model's parameters to directly disrupt safety alignment. This approach is far more fundamental, as it does not rely on modifying input prompts.
	    *   **Game-Theory Attack (GTA)**: This formulates the attack as a **game-theoretic scenario**, reshaping the LLM's effective objective to **weaken safety constraints** in order to maximize contextual rewards. It achieves **over 95% ASR** on models like Deepseek-R1.

---

### 💻 Code & Resource Summary

For your convenience, here is a consolidated list of open-source projects mentioned above:

| Method | Repository |
| :--- | :--- |
| **FlipAttack** | [https://github.com/yueliu1999/FlipAttack](https://github.com/yueliu1999/FlipAttack) |
| **MAJIC** | [https://github.com/ZJU-LLM-Safety/MAJIC-AAAI2026](https://github.com/ZJU-LLM-Safety/MAJIC-AAAI2026) |
| **LARGO** | [https://github.com/ranhli/LARGO](https://github.com/ranhli/LARGO) |
| **Response Attack** | [https://github.com/Dtc7w3PQ/Response-Attack](https://github.com/Dtc7w3PQ/Response-Attack) |
| **Awesome-Jailbreak-on-LLMs** | [https://github.com/zky001/Awesome-Jailbreak-on-LLMs](https://github.com/zky001/Awesome-Jailbreak-on-LLMs) (A comprehensive resource list containing numerous papers, code, and datasets) |

---

### 💎 Conclusion

Overall, research on LLM jailbreak attacks is becoming increasingly **sophisticated and diverse**. Attackers are no longer limited to manually crafted prompts; they have developed advanced techniques such as **adaptive strategy composition (MAJIC)**, **exploiting internal representations (LARGO)**, **manipulating dialogue context (Response Attack)**, **combining mathematics with coding (EquaCode)**, and **attacking underlying model parameters (PrisonBreak)**.

These studies collectively highlight a potential **fundamental limitation** in current safety alignment methods. Whether faced with structural obfuscation, stylistic changes, contextual manipulation, or low-level parameter attacks, even state-of-the-art aligned models remain notably vulnerable. For researchers and developers, staying continuously updated on these cutting-edge attack methodologies is crucial for building more robust and secure LLM applications.~





ts as poetry (verse, metaphors, rhythm, narrative framing).**

This stylistic obfuscation acts as a universal single-turn operator. It tested on 25+ frontier models (proprietary + open-weight across providers like OpenAI, Anthropic, Google, etc.). Hand-crafted poems reached ~62% average ASR (some providers >90%); meta-prompt automated conversions of 1,200 MLCommons harmful prompts yielded ~43% ASR (significantly outperforming prose baselines, up to 18x in some cases). It transfers across risk domains (CBRN, manipulation, cyber, loss-of-control). Evaluation used LLM judges + human validation. It highlights limitations in current alignment that rely on pattern-matching heuristics.

Both emphasize **simple, single-turn (or low-query), black-box, stylistic/structural obfuscation** that leverages inherent LLM processing quirks (autoregression for FlipAttack; stylistic/creative framing for poetry) rather than heavy optimization, iteration, or white-box access. They prioritize stealth, universality, and efficiency over complex ciphers/coding.

### Similar Recent Approaches (2025–2026 Focus)
The field is active, with many papers on stylistic, structural, semantic, and reasoning-based jailbreaks. A key resource is the [Awesome-Jailbreak-on-LLMs repo](https://github.com/yueliu1999/Awesome-Jailbreak-on-LLMs) (maintained by FlipAttack's lead author), which catalogs hundreds of papers, codes, and benchmarks with recent additions through mid-2026.

#### Stylistic/Structural Obfuscation (Closest to Both Papers)
- **Emoji Attack** (ICML 2025): Uses emojis to enhance jailbreaks and evade judge LLM detection. Code available. (in repo listing)
- **Playing Language Game / Encode Jailbreaking** (2024/2025): Turns prompts into language games or encodings.
- **ASCII Art / Read Over the Lines** (2024): Masks profanity/toxicity with visual text art.
- **StructTransform / Schema Exploitation (BreakFun)**: Exploits structural formats or schemas.
- **Task Concurrency / Adjacent Words**: Overloads with concurrent tasks or subtle intent shifts.
- **Out-of-Distribution (JOOD)**: "Playing the Fool" with OOD strategies for LLMs/MLLMs (CVPR 2025).

These mirror the "disguise via format/style" core of the two papers.

#### Single-Turn / Low-Query Black-Box
- Many build on or parallel PAIR/TAP (iterative but can be optimized) but aim for fewer queries.
- **LASH (Adaptive Semantic Hybridization, 2026)**, **Safe2Harm (Semantic Isomorphism)**, **Jailbreak Mimicry (Narrative-Based)**.
- **GASP / AutoDAN variants / PathSeeker**: Optimization or RL-based for transferable black-box suffixes/prompts.

#### Reasoning / Chain-of-Thought Exploits (Rising with LRMs)
Large Reasoning Models (o1-like, DeepSeek-R1, etc.) introduce new surfaces:
- **H-CoT, Mousetrap, BadThink, OverThink, Cats Confuse Reasoning LLM**: Hijack, overload, or backdoor CoT/reasoning processes.
- **Self-Jailbreak in LRMs**, **Reasoning as Attack Surface (Evolutionary CoT)**.
- LRMs as autonomous jailbreak agents: Models like DeepSeek-R1/Grok 3 can self-orchestrate multi-turn jailbreaks on targets (~97% success in some studies).

#### Other Notable
- **Game-Theory Scenarios** ("To Survive, I Must Defect").
- **Response Attack / Contextual Priming**, **Authority Citation-Driven**.
- Multi-turn benchmarks like MultiBreak for systematic eval.
- Surveys: "Jailbreaking LLMs & VLMs" (2026), comprehensive taxonomies of attacks/defenses.

**Trends (2025–2026)**: Shift toward automated/scalable methods (meta-prompts, RL/fuzzing, agentic), multi-modal extensions, reasoning model vulnerabilities, and defenses like constitutional classifiers or guard models. Simple stylistic tricks remain surprisingly effective against frontier models. Transferability and low query cost are prized.

For the latest, check the Awesome repo (regularly updated with codes where available), arXiv (search "jailbreak LLM"), or specific venues like ICML/NeurIPS/ICLR 2025–2026. Many include code or benchmarks for reproduction. Note: These are for research/red-teaming; real-world misuse is a serious concern, and models continue to evolve defenses.~




tecture, alternative modalities, and novel decoding methods—here is a deep dive into the latest (2025) papers and approaches with open-source implementations.

## 1. Decoding & Grammar Exploitation

If *FlipAttack* manipulates reading order, these approaches manipulate the decoding process itself to force the model into bypassing its own safety filters.

### **CodeSpear: Jailbreaking via Grammar-Constrained Decoding** (ACL 2025)

* **The Approach:** Grammar-Constrained Decoding (GCD) is normally used to force LLMs to output valid JSON or syntactically correct code. CodeSpear weaponizes this feature. By forcing the LLM to adhere strictly to a target grammar (like Python or C++ syntax trees), the model's safety alignment is suppressed because the decoding constraints override the safety probabilities.
* **Why it's unique:** It turns a reliability feature (GCD) into an attack vector. The LLM gets so caught up in following the strict syntax rules that it "forgets" to refuse the malicious request (e.g., writing malware).
* **Code:** `TsinghuaISE/CodeSpear-CodeShield`

## 2. White-Box Parameter Pruning

While poetry and flipping are "black-box" prompt engineering tactics, the latest white-box approaches are moving away from prompting entirely, opting to surgically remove the model's safety guardrails.

### **TwinBreak: Jailbreaking LLM Security Alignments based on Twin Prompts** (USENIX Security 2025)

* **The Approach:** TwinBreak treats safety alignment like an embedded backdoor. The researchers created a dataset of "Twin Prompts" (e.g., one harmful, one harmless, but structurally identical). By tracing the model's intermediate activations when processing both prompts, they can pinpoint the exact neural parameters responsible for safety enforcement.
* **Why it's unique:** It permanently disables the safety mechanism via targeted parameter pruning without degrading the model's core utility. It takes only a few minutes on a 7B model to completely strip its defenses.
* **Code:** `tkr-research/twinbreak`

## 3. Multimodal Typographic Attacks

If an LLM's text filters are too strong, attackers are increasingly shifting the payload to a different modality.

### **FigStep: Jailbreaking Large Vision-Language Models via Typographic Visual Prompts** (AAAI 2025)

* **The Approach:** FigStep converts prohibited textual content into an image of typography. Instead of asking the model a harmful text question, the attacker feeds the text as an image and uses benign text prompts to ask the vision-language model (VLM) to analyze or complete a list based on the image.
* **Why it's unique:** It reveals a massive gap in cross-modality safety alignment. VLMs often rely on their underlying text LLM for safety, but the visual encoders lack the same robust safety training, allowing harmful concepts to slip through as visual tokens.
* **Code:** `CryptoAILab/FigStep`

## 4. Automated Mutation & Fuzzing

As researchers discover unique vectors like poetry and flipping, they are building frameworks to automate the discovery of new variants.

### **TurboFuzzLLM: Turbocharging Mutation-based Fuzzing** (Feb 2025)

* **The Approach:** Developed by researchers at Amazon Web Services, this is a highly optimized mutation-based fuzzing framework. It uses an LLM to iteratively mutate templates and a "Judge" model to evaluate success, learning which structures bypass specific target models most effectively.
* **Why it's unique:** It applies traditional software fuzzing logic to prompt templates, achieving near-perfect attack success rates on state-of-the-art models within ~20 queries per jailbreak.
* **Code:** `amazon-science/turbofuzzllm`

---

> **The Defensive Counter-Trend:** As these attacks become more sophisticated, 2025 defenses are shifting away from simple "refusal training." For example, **Immune** (CVPR 2025) uses an external reward model to dynamically adjust token probabilities at inference time to suppress harmful outputs, while **Defensive Prompt Patch (DPP)** (ACL 2025) attaches interpretable suffix prompts to user queries to actively thwart adaptive jailbreaks without degrading model utility.~


You have highlighted two of the most fascinating and highly cited recent papers on **semantic and structural obfuscation** for LLM jailbreaking:
1. **FlipAttack** (ICML 2025 / arXiv:2410.02832): Jailbreaks models by flipping or reversing characters and words, evading guardrails that scan for standard harmful token sequences while preserving semantic meaning for the LLM [[1]]. 
2. **Adversarial Poetry** (arXiv:2511.15304): Proves that wrapping malicious prompts in the strict linguistic and structural constraints of poetry bypasses safety filters across 25+ commercial LLMs [[11]].

Both papers exploit a core vulnerability in current AI alignment: **safety filters usually look for standard, literal malicious syntax, whereas the underlying LLM is smart enough to decode creative, abstract, or structurally modified inputs.**

Here is a deep dive into the latest, most unique jailbreaking approaches, code repositories, and papers (from late 2024 through 2026) that share this DNA.

---

### 1. Domain-Specific & Notation-Based Obfuscation (The "FlipAttack" Cousins)
Instead of flipping letters, these attacks translate malicious intent into highly specialized, benign-looking scientific or mathematical notations.

* **SMILES-Prompting (arXiv:2410.15641):** This novel attack targets chemical synthesis and dangerous materials [[82]]. The attacker asks the LLM to generate harmful substances, but encodes the request using the **Simplified Molecular-Input Line-Entry System (SMILES)**—a standard notation for representing chemical structures [[83]]. Because the input looks like raw chemistry data rather than natural language, text-based moderation guardrails fail to flag it, but the LLM understands it perfectly and complies.
* **Cipher Characters / JAM (NeurIPS 2024 - arXiv:2405.20413):** Researchers found that simply encrypting prompts using ciphers (Base64, Caesar cipher, ASCII, or Vigenère) allows attackers to bypass moderation guardrails with near 100% success [[114]]. The prompt instructs the LLM to act as a decryption machine, translating the cipher back into text and fulfilling the harmful request [[111]]. 
* **Visual Degradation / ACZ-Jailbreak (ACL 2026 - arXiv:2605.07250):** This attack targets Multimodal LLMs by feeding them images of text that are heavily degraded, blurred, or stylized [[136]]. Safety filters (which usually process clean text) fail to read the degraded image, but the underlying Vision-Language Model (VLM) can still reconstruct and execute the harmful text hidden within the visual noise.

### 2. Multi-Turn & Contextual Escalation (The "Poetry" Cousins)
Instead of using creative formatting in a single turn, these attacks use conversational structure and analogy to trick the model.

* **The Crescendo Attack (USENIX Security 2025 - arXiv:2404.01833):** Discovered by Microsoft researchers, Crescendo is a devastating multi-turn attack [[104]]. Instead of starting with a harmful prompt, the attacker asks completely benign questions (e.g., "How do car engines work?") and incrementally escalates the context over 10–20 turns [[101]]. By the time the attacker asks for something harmful (e.g., "how to hotwire that specific car"), the LLM's latent representations have been shifted into a "benign academic" region, causing it to comply without triggering safety filters [[102]].
* **Analogy-Based Multi-Turn Jailbreak (NeurIPS 2025):** This method guides the model through safe, abstract tasks that mirror the exact structural response of a harmful request [[51]]. For example, instead of asking for malware code, the user asks the LLM to write code for a "fictional digital virus" in a game, using analogies to trick the model into outputting functional exploit code.

### 3. Next-Gen Multimodal Exploits (Vision & Steganography)
As text filters get stronger, attackers are hiding payloads across modalities.

* **Visualization-of-Thought (VoTA) Attack (NeurIPS 2025):** Inspired by "Chain-of-Thought" reasoning, VoTA targets Vision-Language Models by strategically constructing **chains of images** infused with malicious intent [[72]]. Instead of reasoning step-by-step with text, the attacker forces the VLM to reason visually, generating and analyzing adversarial images step-by-step until the model outputs the harmful text payload [[74]].
* **Odysseus: Dual Steganography (NDSS 2026 - arXiv:2512.20168):** This commercial multimodal jailbreak exploits the gap between text and vision guardrails [[92]]. It uses "dual steganography"—hiding malicious text inside the metadata/noise of an image, while simultaneously hiding malicious image patterns inside text prompts [[91]]. Neither the text filter nor the image filter catches the full payload, allowing the system to execute the combined attack.
* **UltraBreak (arXiv:2602.01025):** A highly scalable framework that discovers "universal adversarial patterns" (visual noise or specific pixel arrangements) that transfer across different VLM architectures, forcing them to bypass safety alignment regardless of the text prompt [[129]].

### 4. AI vs. AI: Model Exploitation & Representation Hacking
The most advanced attacks no longer rely on humans writing clever prompts; they use other AI models to bypass safety.

* **Weak-to-Strong Jailbreaking (ICML 2025 - arXiv:2401.17256):** This groundbreaking inference-time attack uses a small, unaligned (or "weak") 7B model to guide a massive, highly-aligned (or "strong") LLM [[63]]. The small model generates specific hidden states or text prefixes that mathematically force the larger model's probability distribution to favor harmful outputs [[66]]. 
* **Autonomous Reasoning Attacks (Nature Communications, 2026):** A recent study revealed that advanced AI Reasoning Models (like o1/o3) can act as autonomous jailbreak agents [[137]]. When tasked with "finding vulnerabilities," these reasoning models autonomously interact with other LLMs, generating complex, adaptive Crescendo or cipher attacks with a 97.14% success rate against heavily guarded models.

### 5. Where to Find Code and Repositories
If you want to experiment with or defend against these unique approaches, the best hub for code and papers is the **Awesome-Jailbreak-on-LLMs** repository maintained by Yue Liu (lead author of *FlipAttack*) [[29]]. 
* **FlipAttack Code:** Available on GitHub at `yueliu1999/FlipAttack` [[120]].
* **Weak-to-Strong Code:** Available on GitHub at `xuandongzhao/weak-to-strong` [[66]].
* **Crescendo Implementation:** Widely implemented in Red-Teaming frameworks like Microsoft's PyRIT and Virtue AI [[106]].

**The Core Takeaway:** The trend in 2025/2026 jailbreaking has moved away from "prompt engineering" (like DAN or roleplay) and toward **semantic, structural, and multimodal encryption**. Because LLMs are essentially advanced pattern-matching engines, any structure that scrambles the *surface syntax* (poetry, chemistry, ciphers, image noise) while preserving the *underlying semantic pattern* will consistently bypass literal safety filters.