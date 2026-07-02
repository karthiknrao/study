I'll check out that paper first, then hunt for similar ones and matching code repos.

Got the gist — TwinBreak is a white-box jailbreak method that prunes "safety backdoor" parameters in LLMs using structurally similar twin prompts. Let me dig for related work and code repos now.

I've got a solid picture now. Let me do a couple more targeted searches for the very latest 2025/2026 follow-up work and any specific TwinBreak-related repos.

Let me drop this all into a clean reference file you can keep, then give you the TL;DR up top.

Done. Here's the quick map of what's adjacent to TwinBreak — I've saved the full version (with all arXiv IDs + repo links + dates) to `twinbreak_related_works.md` in your workspace.

## TwinBreak in one line
**Krauß, Dashtbani & Dmitrienko (USENIX Security '25)** — treat the LLM safety mechanism as an embedded backdoor, then **prune the refusal-related parameters** by comparing activations between highly similar "twin" (harmful + harmless) prompts. 89–98% ASR across 16 LLMs in <5 min on a 7B.
**Official code →** https://github.com/tkr-research/twinbreak (⚠️ needs ~48 GB GPU, 120 GB disk)

## What to read first (closest cousins)

1. **Wei et al., arXiv:2402.05162** — *"Assessing the Brittleness of Safety Alignment via Pruning…"* — the technical predecessor. Showed safety lives in ~3% of weights; removed them and ASR went 0 → 90%+. TwinBreak is the smarter follow-up.
2. **DBDI, arXiv:2511.06852 (AAAI '26)** — bi-direction intervention (Harm Detect + Refusal Execution). **Beats TwinBreak** (95.96% vs 94.62% ASR on AdvBench); explicitly uses `TwinPrompt` as a baseline.
3. **Activation Surgery, arXiv:2603.14278** — nearest sibling in spirit: builds a benign "organ-donor" twin prompt and replaces layer-wise activations; no weight edit.
4. **"Safety Alignment Should Be Made More Than Just a Few Tokens Deep" (Qi et al., ICLR 2025 Outstanding Paper)** — why alignment is so shallow and easy to bypass.
5. **Safety Layers in Aligned LLMs (Li et al., ICLR 2025)** — middle layers are the safety locus; SPPFT freezes them.
6. **Towards Understanding Safety Alignment: Safety Neurons (Chen et al., NeurIPS 2025)** — ~5% safety neurons; activation patching recovers 90% safety without retraining.

## Defenses the same way (other side of the coin)
- **Antidote (ICML 2025)** — one-shot pruning of harmful weights after a fine-tuning attack.
- **SPLoRA / Safe Pruning LoRA (TACL 2025)** — https://github.com/AoShuang92/SPLoRA
- **Pruning for Protection (arXiv:2401.10862)** — WANDA pruning *increases* jailbreak resistance.
- **NeWTral (arXiv:2605.04992)** — MoE weight-space translator for safety recovery.
- **Multilingual Safety Alignment via Sparse Weight Editing (arXiv:2602.22554)** — same toolkit, multilingual setting.

## GitHub repos worth ⭐-ingx
- https://github.com/tkr-research/twinbreak — the paper itself
- https://github.com/yueliu1999/Awesome-Jailbreak-on-LLMs — best curated list
- https://github.com/chen37058/Red-Team-Arxiv-Paper-Update — near-daily updated paper feed (flagged TwinBreak on day-1)
- https://github.com/tml-epfl/llm-adaptive-attacks — logprob-based white-box attack
- https://github.com/patrickrchao/JailbreakingLLMs — PAIR
- https://github.com/llm-attacks/llm-attacks — canonical GCG
- https://huggingface.co/CSMaya/er_ablations_qwen_2.5-3B_twinbreak — community "abliteration" port to Qwen-2.5-3B
- https://www.promptfoo.dev/lm-security-db/tag/whitebox — vulnerability database covering WO, Amnesia, SAHA, etc.

## Surveys to read first if you want the landscape
- *Jailbreaking LLMs: A Survey of Attacks, Defenses and Evaluation* (TechRxiv 2026) — best recent unified framing
- *A Survey of Modern LLM Jailbreaks* (OpenReview) — 43 methods classified
- *Jailbreak Attacks and Defenses Against LLMs: A Survey* (arXiv:2407.04295)
- *TeleAI-Safety* benchmark paper (arXiv:2512.05485)

<deliver-assets>
<media type="md" src="commit-id-415178360541483" caption="Full annotated bibliography of TwinBreak-related papers + code repos" name="twinbreak_related_works.md" />
</deliver-assets>

Want me to deep-dive on any one angle — e.g., dig out the actual GitHub stars/issues for the latest activation-level attacks, or sketch a reading plan of the top 5?