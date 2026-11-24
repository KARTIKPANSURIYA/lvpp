"""LVPP: Latent Verification with Prefix Pruning for efficient self-consistency.

Public API (CPU-only parts import without ``transformers``)::

    from lvpp.verifier import fit_verifier
    from lvpp.selection import CandidateSet, selection_hits
    from lvpp.pruning import prune_keep_and_cost
    from lvpp.stats import mcnemar_exact, paired
    from lvpp.caches import load_pool_A, load_dict_B, load_prefix_C
"""
__version__ = "1.0.0"
