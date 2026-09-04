"""Stubs for `topology.physics_activation_density_ph.data` (imported by his `data.py` and
`coordinates.py`). Every function raises if called; `PreparedActivations` is an empty
placeholder class used only in his type annotations."""

from . import _placeholder


class PreparedActivations:
    """Placeholder for the colleague's prepared-activations record; never instantiated here."""


effective_rank_from_cov = _placeholder("topology.physics_activation_density_ph.data.effective_rank_from_cov")
l2_normalize = _placeholder("topology.physics_activation_density_ph.data.l2_normalize")
prepare_activations = _placeholder("topology.physics_activation_density_ph.data.prepare_activations")
summarize_population = _placeholder("topology.physics_activation_density_ph.data.summarize_population")

__all__ = [
    "PreparedActivations", "effective_rank_from_cov", "l2_normalize", "prepare_activations",
    "summarize_population",
]
