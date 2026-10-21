"""Post-generation structure correctors."""

from .hydrogenation import (
    AmbiguousHydrogenationError,
    HydrogenatedStructure,
    HydrogenationError,
    HydrogenationNoSolutionError,
    HydrogenationPlan,
    HydrogenPlacement,
    ResolvedValenceState,
    UnsupportedHydrogenationError,
    hydrogenate_structure,
    infer_hydrogenation_plan,
    place_hydrogens,
)
from .symmetry_rigid_corrector import (
    SymmetryRigidCorrectionResult,
    count_symmetry_periodic_clashes,
    normalize_symmetry_corrector_config,
    symmetry_rigid_postprocess_batch,
)

__all__ = [
    "AmbiguousHydrogenationError",
    "HydrogenPlacement",
    "HydrogenatedStructure",
    "HydrogenationError",
    "HydrogenationNoSolutionError",
    "HydrogenationPlan",
    "ResolvedValenceState",
    "SymmetryRigidCorrectionResult",
    "UnsupportedHydrogenationError",
    "count_symmetry_periodic_clashes",
    "hydrogenate_structure",
    "infer_hydrogenation_plan",
    "normalize_symmetry_corrector_config",
    "place_hydrogens",
    "symmetry_rigid_postprocess_batch",
]
