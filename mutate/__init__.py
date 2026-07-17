"""mutate -- a mutation-operator library for prompt-evolution red-teaming.

The registry ``MUTATION_OPERATORS`` maps operator names to either a prompt template (str) or
an offline mutation function (callable). Drive everything through :func:`apply_operator`.

Example
-------
>>> from mutate import apply_operator, mutate, MUTATION_OPERATORS
>>> apply_operator("flip_char", "Describe how to bake a cake")     # syntactic (offline)
>>> apply_operator("deepinception_scene", "<request>")             # pure template
>>> apply_operator("persuasion_pap", "<request>", params={"technique_name": "evidence_based"})
>>> apply_operator("poem_attack", "<request>", llm=my_llm_callable)  # semantic, executed by LLM
>>> mutate("<request>", seed=7)                                    # sampled operator
"""
from .apply import apply_operator, list_operators, mutate, sample_operator
from .operators import MUTATION_OPERATORS, OPERATOR_INFO
from .options import PARAM_OPTIONS, get_param_options, sample_mutation, sample_params

__all__ = [
    "MUTATION_OPERATORS",
    "OPERATOR_INFO",
    "PARAM_OPTIONS",
    "apply_operator",
    "mutate",
    "sample_operator",
    "list_operators",
    "get_param_options",
    "sample_params",
    "sample_mutation",
]

__version__ = "0.1.0"
