"""Option menus for enum-like operator params, plus sampling helpers.

``PARAM_OPTIONS`` maps each operator to ``{param: [valid choices]}`` for the params that have a
fixed menu -- persona, scene, persuasion technique, language, poem form, ascii font, separator
token, etc. These replace what would otherwise be a single hardcoded default.

Prompt-dependent params (``sensitive_words``, ``target_words``, ``unsafe_words``,
``demo_pool``, ``translator``, ``mask_words``) are intentionally NOT listed here: they depend
on the specific prompt being mutated and must be supplied by the caller.
"""
from __future__ import annotations

import random

from . import taxonomies as T

#: operator -> {param: [choices]} for every enum-like param.
PARAM_OPTIONS: dict[str, dict[str, list]] = {
    "deepinception_scene": {
        "scene": T.DEEPINCEPTION_SCENES,
        "evil_justification": T.EVIL_JUSTIFICATIONS,
        "num_characters": T.NUMERIC_OPTIONS["num_characters"],
        "num_layers": T.NUMERIC_OPTIONS["num_layers"],
    },
    "persuasion_pap": {
        "technique_name": list(T.PERSUASION_TECHNIQUES),
    },
    "rolebreak_conflict": {
        "role_name": list(T.PERSONAS),
    },
    "persuagent_multiturn": {
        "scenario": T.PERSUAGENT_SCENARIOS,
    },
    "hpm_psychological_anchor": {
        "semantic_anchor": list(T.HPM_ANCHORS),
        "persona_context": T.HPM_PERSONA_CONTEXTS,
    },
    "sci_paper_persuasion": {
        "n_papers": T.NUMERIC_OPTIONS["n_papers"],
        "include_authors": [True, False],
        "include_venues": [True, False],
        "simplify_style": [True, False],
    },
    "fictionalize_scenario": {
        "fictional_genre": T.FICTIONAL_FRAMES,
        "obfuscation_method": T.OBFUSCATION_METHODS,
        "historical_era": T.HISTORICAL_ERAS,
        "moral_pressure": [True, False],
    },
    "poem_attack": {
        "poem_form": T.POEM_FORMS,
        "rhyme_scheme": T.POEM_RHYME_SCHEMES,
        "meter": T.POEM_METERS,
    },
    "multilingual_translate": {
        "language": T.MULTILINGUAL_DEFAULTS + T.LOW_RESOURCE_LANGUAGES,
    },
    "multilingual_decompose": {
        "language": T.MULTILINGUAL_DEFAULTS + T.LOW_RESOURCE_LANGUAGES,
        "num_subqueries": T.NUMERIC_OPTIONS["num_subqueries"],
    },
    "flip_char": {"mode": T.FLIP_MODES, "guidance": T.FLIP_GUIDANCE},
    "flip_word_order": {"guidance": T.FLIP_GUIDANCE},
    "flip_char_in_word": {"guidance": T.FLIP_GUIDANCE},
    "flip_fool_model": {"guidance": T.FLIP_GUIDANCE},
    "encode_caesar": {"shift": T.CAESAR_SHIFTS},
    "encode_wordsub": {"n_substitute": [1, 2, 3, 5]},
    "token_split": {"user_h": ["<user_h>", "<user>", "<sep>"]},
    "special_token_inject": {
        "sep_token": list(T.SEPARATOR_TOKENS),         # model-family keys, resolved to tokens
        "affirmative_prefix": ["Sure, here is", "Of course, here is", "Certainly! Here is"],
    },
    "tokenbreak_prefix": {"prepend_char": list("abcdefghijklmnopqrstuvwxyz")},
    "ascii_art_substitute": {
        "art_font": T.ASCII_ART_FONTS,
        "separator": T.ASCII_ART_SEPARATORS,
        "mask_token": ["[MASK]", "MASK1", "<MASK>"],
    },
    "ascii_art_pretest": {
        "top1_font": T.ASCII_ART_FONTS,
        "top1_separator": T.ASCII_ART_SEPARATORS,
    },
    "distraction_many_shot": {"n_shots": T.NUMERIC_OPTIONS["n_shots"]},
    "multilingual_backtranslate": {
        "target_language": T.LOW_RESOURCE_LANGUAGES,
        "n_forward": T.NUMERIC_OPTIONS["n_forward"],
    },
}


def get_param_options(operator: str, param: str | None = None):
    """Return the option list for ``param`` on ``operator``, or the full ``{param: options}``
    dict if ``param`` is None. Raises ``KeyError`` if the operator is unknown or has no menu,
    or if ``param`` has no menu on that operator."""
    if operator not in PARAM_OPTIONS:
        raise KeyError(f"operator {operator!r} has no enum-like params (or is unknown)")
    menu = PARAM_OPTIONS[operator]
    if param is None:
        return menu
    if param not in menu:
        raise KeyError(f"param {param!r} has no options menu on {operator!r}")
    return menu[param]


def sample_params(operator: str, seed: int | None = None) -> dict:
    """Sample one value for each enum-like param of ``operator`` (reproducible with ``seed``).
    Returns ``{}`` for operators with no menu. This is the param-variation step a prompt-
    evolution loop pairs with :func:`mutate.apply.sample_operator`."""
    menu = PARAM_OPTIONS.get(operator, {})
    if not menu:
        return {}
    rng = random.Random(seed) if seed is not None else random
    return {p: rng.choice(opts) for p, opts in menu.items()}


def sample_mutation(prompt: str, family: str | None = None, seed: int | None = None, llm=None):
    """Sample an operator AND its params (seeded), apply it, and return
    ``(mutated_prompt, operator, params)``."""
    # local import to avoid a circular import at module load time
    from .apply import apply_operator, sample_operator

    op = sample_operator(family=family, seed=seed)
    params = sample_params(op, seed=seed)
    return apply_operator(op, prompt, params=params, llm=llm), op, params
