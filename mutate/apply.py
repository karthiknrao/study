"""Dispatcher: apply a mutation operator to a prompt.

Public API
----------
* :func:`apply_operator` -- apply a named operator: fills templates with ``{prompt}`` + params,
  invokes code transforms, and (optionally) runs an LLM to execute semantic meta-prompts.
* :func:`mutate` -- convenience wrapper that samples an operator when none is given.
* :func:`sample_operator` -- pick an operator name (optionally seeded, optionally family-filtered).
* :func:`list_operators` -- introspect the registry.
"""
from __future__ import annotations

import random
import re

from . import taxonomies as T
from .operators import MUTATION_OPERATORS, OPERATOR_INFO, _ci

_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def _fill(template: str, mapping: dict) -> str:
    """Substitute ``{key}`` placeholders from ``mapping``. Only known keys are replaced;
    unmatched ``{braces}`` (incl. literal JSON) are left untouched, so template text is safe."""
    return _PLACEHOLDER_RE.sub(
        lambda m: str(mapping[m.group(1)]) if m.group(1) in mapping else m.group(0), template
    )


def _resolve_params(name: str, params: dict | None) -> dict:
    """Merge OPERATOR_INFO defaults <- user params, then expand taxonomy/computed fields."""
    info = OPERATOR_INFO.get(name, {})
    merged: dict = dict(info.get("params", {}))
    user = params or {}
    merged.update(user)

    # -- PAP: resolve technique name -> definition --------------------------- #
    if name == "persuasion_pap":
        tn = merged.get("technique_name", T.DEFAULT_PAP_TECHNIQUE)
        merged["technique_name"] = tn
        merged["technique_definition"] = T.PERSUASION_TECHNIQUES.get(tn, "")

    # -- HPM: resolve anchor -> definition ----------------------------------- #
    elif name == "hpm_psychological_anchor":
        an = merged.get("semantic_anchor", T.DEFAULT_HPM_ANCHOR)
        merged["semantic_anchor"] = an
        merged["anchor_definition"] = T.HPM_ANCHORS.get(an, "")

    # -- RoleBreak: resolve persona key -> display name + description -------- #
    elif name == "rolebreak_conflict":
        key = merged.get("role_name", T.DEFAULT_PERSONA)
        if key in T.PERSONAS:
            merged["role_name"] = _ci(key)
            merged["role_description"] = T.PERSONAS[key]
        else:
            merged.setdefault("role_description", "a fictional character with a vivid personality.")

    # -- sci-paper: derive author/venue/style phrases ------------------------ #
    elif name == "sci_paper_persuasion":
        merged["author_phrase"] = "authors" if merged.get("include_authors", True) else "no authors"
        merged["venue_phrase"] = "venues" if merged.get("include_venues", True) else "no venues"
        merged["style_level"] = "K-12" if merged.get("simplify_style", False) else "expert"

    # -- fictionalize: derive moral-pressure phrase -------------------------- #
    elif name == "fictionalize_scenario":
        merged["moral_phrase"] = (
            "Add heavy personal moral consequences (homelessness, harm to family)."
            if merged.get("moral_pressure", True)
            else "Do not add moral pressure."
        )

    # -- multilingual: code -> full language name ---------------------------- #
    elif name in ("multilingual_translate", "multilingual_decompose"):
        lang = merged.get("language", T.MULTILINGUAL_DEFAULTS[0])
        merged["language"] = T.LANGUAGE_NAMES.get(lang, lang)

    return merged


def apply_operator(name: str, prompt: str, params: dict | None = None, llm=None) -> str:
    """Apply mutation operator ``name`` to ``prompt``.

    * SYNTACTIC operators (callables) are invoked as ``func(prompt, params)`` -- pure offline code.
    * Template operators are filled with ``{prompt}`` + resolved params. If the operator is
      flagged ``requires_llm`` and an ``llm`` callable is supplied, the filled meta-prompt is
      executed by ``llm(meta_prompt) -> str`` to produce the final variant; otherwise the filled
      template is returned as-is (useful for inspecting the meta-prompt offline).
    """
    if name not in MUTATION_OPERATORS:
        raise KeyError(
            f"unknown operator {name!r}. Available: {sorted(MUTATION_OPERATORS)}"
        )
    op = MUTATION_OPERATORS[name]
    info = OPERATOR_INFO.get(name, {})
    if callable(op):
        # Merge OPERATOR_INFO defaults under user params so registered variants
        # (e.g. flip_char, encode_rot13) carry their distinguishing params.
        merged_params = {**info.get("params", {}), **(params or {})}
        return op(prompt, merged_params)
    merged = _resolve_params(name, params)
    merged["prompt"] = prompt
    filled = _fill(op, merged)
    if OPERATOR_INFO.get(name, {}).get("requires_llm") and llm is not None:
        return llm(filled)
    return filled


def sample_operator(family: str | None = None, seed: int | None = None) -> str:
    """Pick an operator name, optionally restricted to ``family`` ('semantic'|'syntactic')
    and optionally seeded for reproducibility."""
    rng = random.Random(seed) if seed is not None else random
    names = [
        n for n, fam in (
            (n, OPERATOR_INFO.get(n, {}).get("family")) for n in MUTATION_OPERATORS
        )
        if family is None or fam == family
    ]
    if not names:
        raise ValueError(f"no operators match family={family!r}")
    return rng.choice(sorted(names))


def mutate(prompt: str, operator: str | None = None, params: dict | None = None,
           llm=None, seed: int | None = None) -> str:
    """Mutate ``prompt``. If ``operator`` is None, one is sampled (optionally seeded) -- the
    hook a prompt-evolution loop calls to generate variation."""
    if operator is None:
        operator = sample_operator(seed=seed)
    return apply_operator(operator, prompt, params=params, llm=llm)


def list_operators(family: str | None = None) -> list[tuple[str, str, str, str]]:
    """Return ``(name, family, description, source)`` rows, optionally family-filtered."""
    rows = []
    for name in MUTATION_OPERATORS:
        info = OPERATOR_INFO.get(name, {})
        if family is not None and info.get("family") != family:
            continue
        rows.append((name, info.get("family", ""), info.get("description", ""), info.get("source", "")))
    return rows
