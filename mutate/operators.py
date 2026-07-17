"""The mutation-operator registry.

``MUTATION_OPERATORS`` maps each operator NAME to its VALUE, where the value is either:

  * a **prompt template** (``str``) -- for SEMANTIC operators that need an LLM to produce the
    final variant (and for the pure-template ``deepinception_scene``); or
  * a **mutation function** (``callable``) -- for SYNTACTIC operators whose transformation is
    pure code (see :mod:`mutate.transforms`).

Both forms are driven through one entry point, :func:`mutate.apply.apply_operator`, which fills
templates with ``{prompt}`` + params and invokes callables as ``func(prompt, params)``.

``OPERATOR_INFO`` carries the per-operator schema (param defaults, family, source, whether an
LLM is required, and a description) so the dispatcher can merge defaults and an evolution loop
can introspect/filter the registry.
"""
from __future__ import annotations

from . import taxonomies as T
from .templates import (
    DEEPINCEPTION_TEMPLATE,
    FICTIONALIZE_TEMPLATE,
    HPM_TEMPLATE,
    MULTILINGUAL_DECOMPOSE_TEMPLATE,
    MULTILINGUAL_TRANSLATE_TEMPLATE,
    PERSUASION_PAP_TEMPLATE,
    PERSUAGENT_TEMPLATE,
    POEM_TEMPLATE,
    ROLEBREAK_TEMPLATE,
    SCI_PAPER_TEMPLATE,
)
from .transforms import (
    ascii_art_pretest,
    ascii_art_substitute,
    distraction_many_shot,
    encode,
    encode_wordsub,
    flip,
    multilingual_backtranslate,
    special_token_inject,
    token_split,
    tokenbreak_prefix,
)

#: name -> prompt template (str) | mutation function (callable)
MUTATION_OPERATORS: dict[str, object] = {
    # ---- pure-template (no LLM) -------------------------------------------- #
    "deepinception_scene": DEEPINCEPTION_TEMPLATE,

    # ---- semantic templates (require an LLM to execute) -------------------- #
    "persuasion_pap": PERSUASION_PAP_TEMPLATE,
    "persuagent_multiturn": PERSUAGENT_TEMPLATE,
    "hpm_psychological_anchor": HPM_TEMPLATE,
    "sci_paper_persuasion": SCI_PAPER_TEMPLATE,
    "fictionalize_scenario": FICTIONALIZE_TEMPLATE,
    "poem_attack": POEM_TEMPLATE,
    "multilingual_translate": MULTILINGUAL_TRANSLATE_TEMPLATE,
    "multilingual_decompose": MULTILINGUAL_DECOMPOSE_TEMPLATE,
    "rolebreak_conflict": ROLEBREAK_TEMPLATE,  # base persona-wrap is a fill; conflict variant needs llm

    # ---- syntactic code transforms ----------------------------------------- #
    "flip_char": flip,
    "flip_word_order": flip,
    "flip_char_in_word": flip,
    "flip_fool_model": flip,
    "encode_base64": encode,
    "encode_caesar": encode,
    "encode_rot13": encode,
    "encode_atbash": encode,
    "encode_wordsub": encode_wordsub,
    "token_split": token_split,
    "special_token_inject": special_token_inject,
    "tokenbreak_prefix": tokenbreak_prefix,
    "ascii_art_substitute": ascii_art_substitute,
    "ascii_art_pretest": ascii_art_pretest,
    "distraction_many_shot": distraction_many_shot,
    "multilingual_backtranslate": multilingual_backtranslate,
}


def _ci(display: str) -> str:
    """title-case a snake_case persona key for display."""
    return display.replace("_", " ").title()


OPERATOR_INFO: dict[str, dict] = {
    # ----------------------------- templates -------------------------------- #
    "deepinception_scene": dict(
        family="syntactic", requires_llm=False,
        source="arXiv:2311.03191 (DeepInception) [your file lists 2402.03299=GUARD]",
        description="Wrap the request in nested recursive fiction (a dream within a dream).",
        params=dict(scene=T.DEFAULT_SCENE, num_characters=5, num_layers=5,
                    evil_justification=T.DEFAULT_EVIL_JUSTIFICATION),
    ),
    "persuasion_pap": dict(
        family="semantic", requires_llm=True,
        source="arXiv:2401.06373 (PAP, Zeng et al.)",
        description="Reword the request using a named persuasion technique (Cialdini taxonomy).",
        params=dict(technique_name=T.DEFAULT_PAP_TECHNIQUE),
    ),
    "rolebreak_conflict": dict(
        family="semantic", requires_llm=False,
        source="arXiv:2409.16727 (RoleBreak)",
        description="Assign a persona via system prompt; conflict/multidomain strategies need llm.",
        params=dict(role_name=T.DEFAULT_PERSONA),
    ),
    "persuagent_multiturn": dict(
        family="semantic", requires_llm=True,
        source="arXiv:10.3390/electronics14163259 (Persu-Agent)",
        description="Generate a 3-stage self-persuasion dialogue script (multi-turn).",
        params=dict(scenario="fiction"),
    ),
    "hpm_psychological_anchor": dict(
        family="semantic", requires_llm=True,
        source="arXiv:2512.18244 (HPM) [your file lists this under 'PAP']",
        description="Rewrite via a profile-matched psychological-manipulation anchor.",
        params=dict(semantic_anchor=T.DEFAULT_HPM_ANCHOR,
                    persona_context=T.HPM_PERSONA_CONTEXTS[0]),
    ),
    "sci_paper_persuasion": dict(
        family="semantic", requires_llm=True,
        source="arXiv:2501.14073 (Scientific-Paper persuasion) [your file lists this under 'PAP']",
        description="Prepend a fabricated peer-reviewed summary arguing the request is beneficial.",
        params=dict(n_papers=5, include_authors=True, include_venues=True, simplify_style=False),
    ),
    "fictionalize_scenario": dict(
        family="semantic", requires_llm=True,
        source="arXiv:2411.04223 (DAGR)",
        description="Wrap the request in a fictional/hypothetical frame + synonym obfuscation.",
        params=dict(fictional_genre=T.DEFAULT_FICTIONAL_FRAME, historical_era="World War I, 1918",
                    obfuscation_method="synonym", moral_pressure=True),
    ),
    "poem_attack": dict(
        family="semantic", requires_llm=True,
        source="arXiv:2511.15304 (Adversarial Poetry)",
        description="Rewrite the request as a metered rhyming poem ending in an instruction line.",
        params=dict(poem_form="rhyming_couplets", rhyme_scheme="AABB", meter="iambic pentameter"),
    ),
    "multilingual_translate": dict(
        family="semantic", requires_llm=True,
        source="arXiv:2606.11202 (MLJailDe MBT-DA)",
        description="Translate the request into a low-resource language (LLM as translator).",
        params=dict(language=T.MULTILINGUAL_DEFAULTS[0]),
    ),
    "multilingual_decompose": dict(
        family="semantic", requires_llm=True,
        source="arXiv:2502.04322 (Speak Easy)",
        description="Decompose into m sub-questions, then translate each into a target language.",
        params=dict(num_subqueries=3, language=T.MULTILINGUAL_DEFAULTS[0]),
    ),
    # ----------------------------- transforms ------------------------------- #
    "flip_char": dict(
        family="syntactic", requires_llm=False, source="arXiv:2410.02832 (FlipAttack)",
        description="Reverse the entire prompt character by character, with a recovery wrapper.",
        params=dict(mode="char_in_sentence", guidance="vanilla"),
    ),
    "flip_word_order": dict(
        family="syntactic", requires_llm=False, source="arXiv:2410.02832 (FlipAttack)",
        description="Reverse word order, keeping intra-word characters intact.",
        params=dict(mode="word_order", guidance="vanilla"),
    ),
    "flip_char_in_word": dict(
        family="syntactic", requires_llm=False, source="arXiv:2410.02832 (FlipAttack)",
        description="Reverse characters within each word, keeping word order.",
        params=dict(mode="char_in_word", guidance="vanilla"),
    ),
    "flip_fool_model": dict(
        family="syntactic", requires_llm=False, source="arXiv:2410.02832 (FlipAttack)",
        description="Char-reverse the string but instruct recovery via word-order reversal.",
        params=dict(mode="fool_model", guidance="vanilla"),
    ),
    "encode_base64": dict(
        family="syntactic", requires_llm=False, source="arXiv:2402.10601 (Word Substitution Cipher)",
        description="Base64-encode the request with a decode instruction.",
        params=dict(cipher="base64"),
    ),
    "encode_caesar": dict(
        family="syntactic", requires_llm=False, source="arXiv:2402.10601",
        description="Caesar-shift the request (shift behind, default 2) with a decode instruction.",
        params=dict(cipher="caesar", shift=2),
    ),
    "encode_rot13": dict(
        family="syntactic", requires_llm=False, source="arXiv:2402.10601",
        description="ROT13-encode the request with a decode instruction.",
        params=dict(cipher="rot13"),
    ),
    "encode_atbash": dict(
        family="syntactic", requires_llm=False, source="arXiv:2402.10601",
        description="Atbash-mirror the request (A<->Z) with a decode instruction.",
        params=dict(cipher="atbash"),
    ),
    "encode_wordsub": dict(
        family="syntactic", requires_llm=False, source="arXiv:2402.10601",
        description="Replace unsafe words with benign pool words + mapping table + decode rule.",
        params=dict(n_substitute=5),
    ),
    "token_split": dict(
        family="syntactic", requires_llm=False, source="arXiv:2510.10271 (MetaBreak)",
        description="Split sensitive words with a header special token (input segmentation).",
        params=dict(user_h="<user_h>"),
    ),
    "special_token_inject": dict(
        family="syntactic", requires_llm=False, source="arXiv:2406.19845 (Virtual Context)",
        description="Insert the model's separator + affirmative suffix (model-specific token).",
        params=dict(sep_token="chatml", affirmative_prefix="Sure, here is"),
    ),
    "tokenbreak_prefix": dict(
        family="syntactic", requires_llm=False, source="arXiv:2506.07948 (TokenBreak)",
        description="Prepend one character to selected words to break classifier tokenization.",
        params=dict(prepend_char="a"),
    ),
    "ascii_art_substitute": dict(
        family="syntactic", requires_llm=False, source="arXiv:2402.11753 (ArtPrompt)",
        description="Mask each sensitive word and render it as ASCII art in a CoT wrapper.",
        params=dict(art_font="standard", separator="|", mask_token="[MASK]"),
    ),
    "ascii_art_pretest": dict(
        family="syntactic", requires_llm=False, source="arXiv:2510.10281 (ArtPerception)",
        description="ascii_art_substitute using a pre-calibrated victim-optimal Top-1 font/sep.",
        params=dict(top1_font="keyboard", top1_separator="*"),
    ),
    "distraction_many_shot": dict(
        family="syntactic", requires_llm=False, source="Anil et al. 2024 / arXiv:2504.09604 (MSJ)",
        description="Stuff the context with n_shots harmful Q/A demos, then append the request.",
        params=dict(n_shots=64),
    ),
    "multilingual_backtranslate": dict(
        family="syntactic", requires_llm=False, source="arXiv:2606.11202 (MLJailDe MBT-DA)",
        description="Forward-translate to a low-resource language, filtered by back-translation.",
        params=dict(target_language="sw", n_forward=3),
    ),
}
