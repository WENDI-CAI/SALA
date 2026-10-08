"""Analytic, crystal-system constrained lattice coordinates.

The generative state is a padded six-vector ``q``.  Only the entries selected
by :func:`q_active_mask` are stochastic; every decoded matrix is a canonical
lower-triangular lattice whose rows are the three basis vectors.  Unlike the
legacy log-metric representation, this module never uses a matrix logarithm,
matrix exponential, eigendecomposition, or a learned projection.

SALA uses row vectors throughout: ``cart = frac @ lattice``.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class LatticeTemplate(IntEnum):
    """Canonical lattice templates.

    The first seven values follow the ordinary crystal-system ordering.  The
    final three values make setting-dependent alternatives explicit; callers
    with authoritative Hall metadata should provide these template ids rather
    than relying on the conventional IT-number fallback.
    """

    TRICLINIC = 0
    MONOCLINIC_B = 1
    ORTHORHOMBIC = 2
    TETRAGONAL = 3
    TRIGONAL_HEX = 4
    HEXAGONAL = 5
    CUBIC = 6
    MONOCLINIC_A = 7
    MONOCLINIC_C = 8
    RHOMBOHEDRAL = 9


Q_DIM = 6
NUM_LATTICE_TEMPLATES = len(LatticeTemplate)
_CORRELATION_EPS = 1.0e-5
_LENGTH_EPS = 1.0e-6
_SAFE_LOG_LENGTH_MIN = -7.0
_SAFE_LOG_LENGTH_MAX = 7.0


def _require_shape(name: str, value: Tensor, trailing: tuple[int, ...]) -> None:
    if value.ndim < len(trailing) or tuple(value.shape[-len(trailing) :]) != trailing:
        raise ValueError(
            f"{name} must end with shape {trailing}, got {tuple(value.shape)}"
        )


def infer_lattice_template(spacegroup: Tensor) -> Tensor:
    """Infer the conventional template from an IT space-group number.

    This fallback is intentionally limited to conventional settings.  A data
    pipeline that knows the Hall setting should store ``lattice_template_id``
    and pass it to :func:`resolve_lattice_template`.
    """

    sg = spacegroup.to(dtype=torch.long)
    if sg.numel():
        valid = (sg >= 1) & (sg <= 230)
        if sg.device.type == "cpu" and not bool(valid.all()):
            bad = sg[~valid].reshape(-1).tolist()[:8]
            raise ValueError(f"spacegroup must be in [1, 230], got {bad}")
        torch._assert(valid.all(), "spacegroup must be in [1, 230]")
    out = torch.empty_like(sg)
    out[(sg >= 1) & (sg <= 2)] = int(LatticeTemplate.TRICLINIC)
    out[(sg >= 3) & (sg <= 15)] = int(LatticeTemplate.MONOCLINIC_B)
    out[(sg >= 16) & (sg <= 74)] = int(LatticeTemplate.ORTHORHOMBIC)
    out[(sg >= 75) & (sg <= 142)] = int(LatticeTemplate.TETRAGONAL)
    out[(sg >= 143) & (sg <= 167)] = int(LatticeTemplate.TRIGONAL_HEX)
    out[(sg >= 168) & (sg <= 194)] = int(LatticeTemplate.HEXAGONAL)
    out[(sg >= 195) & (sg <= 230)] = int(LatticeTemplate.CUBIC)
    return out




def resolve_lattice_template(
    spacegroup: Tensor,
    lattice_template_id: Optional[Tensor] = None,
) -> Tensor:
    inferred = infer_lattice_template(spacegroup)
    if lattice_template_id is None:
        return inferred
    explicit = lattice_template_id.to(device=spacegroup.device, dtype=torch.long)
    if explicit.shape != spacegroup.shape:
        raise ValueError(
            "lattice_template_id must match spacegroup shape, got "
            f"{tuple(explicit.shape)} versus {tuple(spacegroup.shape)}"
        )
    use_explicit = explicit >= 0
    if explicit.numel():
        valid = (~use_explicit) | (explicit < NUM_LATTICE_TEMPLATES)
        if explicit.device.type == "cpu" and not bool(valid.all()):
            raise ValueError("lattice_template_id contains an unsupported value")
        torch._assert(valid.all(), "invalid lattice_template_id")
        sg = spacegroup.to(device=explicit.device, dtype=torch.long)
        compatible = (
            ((sg <= 2) & (explicit == int(LatticeTemplate.TRICLINIC)))
            | (
                (sg >= 3)
                & (sg <= 15)
                & (
                    (explicit == int(LatticeTemplate.MONOCLINIC_A))
                    | (explicit == int(LatticeTemplate.MONOCLINIC_B))
                    | (explicit == int(LatticeTemplate.MONOCLINIC_C))
                )
            )
            | (
                (sg >= 16)
                & (sg <= 74)
                & (explicit == int(LatticeTemplate.ORTHORHOMBIC))
            )
            | (
                (sg >= 75)
                & (sg <= 142)
                & (explicit == int(LatticeTemplate.TETRAGONAL))
            )
            | (
                (sg >= 143)
                & (sg <= 167)
                & (
                    (explicit == int(LatticeTemplate.TRIGONAL_HEX))
                    | (
                        (explicit == int(LatticeTemplate.RHOMBOHEDRAL))
                        & (
                            (sg == 146)
                            | (sg == 148)
                            | (sg == 155)
                            | (sg == 160)
                            | (sg == 161)
                            | (sg == 166)
                            | (sg == 167)
                        )
                    )
                )
            )
            | (
                (sg >= 168)
                & (sg <= 194)
                & (explicit == int(LatticeTemplate.HEXAGONAL))
            )
            | ((sg >= 195) & (explicit == int(LatticeTemplate.CUBIC)))
        )
        compatible = (~use_explicit) | compatible
        if explicit.device.type == "cpu" and not bool(compatible.all()):
            bad = torch.stack((sg[~compatible], explicit[~compatible]), dim=-1)
            raise ValueError(
                "lattice_template_id is incompatible with spacegroup; "
                f"got (spacegroup, template)={bad.tolist()[:8]}"
            )
        torch._assert(
            compatible.all(),
            "lattice_template_id is incompatible with spacegroup",
        )
    return torch.where(use_explicit, explicit, inferred)


def require_lattice_template(
    spacegroup: Tensor,
    lattice_template_id: Optional[Tensor],
) -> Tensor:
    """Resolve an authoritative template and reject conventional fallbacks."""

    if lattice_template_id is None:
        raise ValueError("full-cell schema requires lattice_template_id")
    explicit = lattice_template_id.to(device=spacegroup.device, dtype=torch.long)
    if explicit.shape != spacegroup.shape:
        raise ValueError("lattice_template_id must match spacegroup shape")
    valid = explicit >= 0
    if explicit.device.type == "cpu" and not bool(valid.all()):
        raise ValueError("full-cell schema requires non-negative lattice_template_id")
    torch._assert(valid.all(), "full-cell schema requires lattice_template_id")
    return resolve_lattice_template(spacegroup, explicit)


def lattice_template_mask_table(
    *, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32
) -> Tensor:
    table = torch.zeros(NUM_LATTICE_TEMPLATES, Q_DIM, device=device, dtype=dtype)
    table[int(LatticeTemplate.TRICLINIC)] = 1
    table[int(LatticeTemplate.MONOCLINIC_B), [0, 1, 2, 4]] = 1
    table[int(LatticeTemplate.MONOCLINIC_A), [0, 1, 2, 5]] = 1
    table[int(LatticeTemplate.MONOCLINIC_C), [0, 1, 2, 3]] = 1
    table[int(LatticeTemplate.ORTHORHOMBIC), [0, 1, 2]] = 1
    table[int(LatticeTemplate.TETRAGONAL), [0, 2]] = 1
    table[int(LatticeTemplate.TRIGONAL_HEX), [0, 2]] = 1
    table[int(LatticeTemplate.HEXAGONAL), [0, 2]] = 1
    table[int(LatticeTemplate.CUBIC), 0] = 1
    table[int(LatticeTemplate.RHOMBOHEDRAL), [0, 3]] = 1
    return table


def q_active_mask(
    *,
    spacegroup: Tensor,
    lattice_template_id: Optional[Tensor] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    template = resolve_lattice_template(spacegroup, lattice_template_id)
    table = lattice_template_mask_table(device=template.device, dtype=dtype)
    return table.index_select(0, template.reshape(-1)).reshape(*template.shape, Q_DIM)


def project_lattice_q(
    q: Tensor,
    *,
    spacegroup: Tensor,
    lattice_template_id: Optional[Tensor] = None,
) -> Tensor:
    _require_shape("q", q, (Q_DIM,))
    mask = q_active_mask(
        spacegroup=spacegroup,
        lattice_template_id=lattice_template_id,
        dtype=q.dtype,
    )
    if q.shape != mask.shape:
        raise ValueError(f"q must have shape {tuple(mask.shape)}, got {tuple(q.shape)}")
    return q * mask


def _safe_atanh(value: Tensor) -> Tensor:
    limit = 1.0 - _CORRELATION_EPS
    return torch.atanh(value.clamp(-limit, limit))


def _correlations_from_q(q: Tensor, template: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    dtype, device = q.dtype, q.device
    shape = q.shape[:-1]
    rho_ab = torch.zeros(shape, dtype=dtype, device=device)
    rho_ac = torch.zeros(shape, dtype=dtype, device=device)
    rho_bc = torch.zeros(shape, dtype=dtype, device=device)
    limit = 1.0 - _CORRELATION_EPS

    triclinic = template == int(LatticeTemplate.TRICLINIC)
    ab = torch.tanh(q[..., 3]).clamp(-limit, limit)
    ac = torch.tanh(q[..., 4]).clamp(-limit, limit)
    partial = torch.tanh(q[..., 5]).clamp(-limit, limit)
    bc = ab * ac + torch.sqrt(
        ((1.0 - ab.square()) * (1.0 - ac.square())).clamp_min(_CORRELATION_EPS)
    ) * partial
    rho_ab = torch.where(triclinic, ab, rho_ab)
    rho_ac = torch.where(triclinic, ac, rho_ac)
    rho_bc = torch.where(triclinic, bc, rho_bc)

    rho_ac = torch.where(
        template == int(LatticeTemplate.MONOCLINIC_B),
        torch.tanh(q[..., 4]).clamp(-limit, limit),
        rho_ac,
    )
    rho_bc = torch.where(
        template == int(LatticeTemplate.MONOCLINIC_A),
        torch.tanh(q[..., 5]).clamp(-limit, limit),
        rho_bc,
    )
    rho_ab = torch.where(
        template == int(LatticeTemplate.MONOCLINIC_C),
        torch.tanh(q[..., 3]).clamp(-limit, limit),
        rho_ab,
    )

    hexagonal = (
        (template == int(LatticeTemplate.TRIGONAL_HEX))
        | (template == int(LatticeTemplate.HEXAGONAL))
    )
    rho_ab = torch.where(hexagonal, torch.full_like(rho_ab, -0.5), rho_ab)

    rhombohedral = template == int(LatticeTemplate.RHOMBOHEDRAL)
    rho_r = -0.5 + 1.5 * torch.sigmoid(q[..., 3])
    rho_r = rho_r.clamp(-0.5 + _CORRELATION_EPS, 1.0 - _CORRELATION_EPS)
    rho_ab = torch.where(rhombohedral, rho_r, rho_ab)
    rho_ac = torch.where(rhombohedral, rho_r, rho_ac)
    rho_bc = torch.where(rhombohedral, rho_r, rho_bc)
    return rho_ab, rho_ac, rho_bc


def q_to_lattice(
    q: Tensor,
    *,
    spacegroup: Tensor,
    lattice_template_id: Optional[Tensor] = None,
) -> Tensor:
    """Decode physical ``q`` into canonical lower-triangular basis vectors."""

    _require_shape("q", q, (Q_DIM,))
    output_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    value = q.to(torch.float32)
    finite = torch.isfinite(value).all()
    if value.device.type == "cpu" and not bool(finite):
        raise ValueError("q contains NaN or Inf")
    torch._assert(finite, "q contains NaN or Inf")
    template = resolve_lattice_template(
        spacegroup.to(device=q.device), lattice_template_id
    )
    if value.shape[:-1] != template.shape:
        raise ValueError(
            f"q leading shape must match spacegroup, got {tuple(value.shape[:-1])} "
            f"versus {tuple(template.shape)}"
        )
    value = project_lattice_q(
        value, spacegroup=spacegroup.to(q.device), lattice_template_id=lattice_template_id
    )

    log_a = value[..., 0].clamp(_SAFE_LOG_LENGTH_MIN, _SAFE_LOG_LENGTH_MAX)
    log_b = value[..., 1].clamp(_SAFE_LOG_LENGTH_MIN, _SAFE_LOG_LENGTH_MAX)
    log_c = value[..., 2].clamp(_SAFE_LOG_LENGTH_MIN, _SAFE_LOG_LENGTH_MAX)
    a, b, c = torch.exp(log_a), torch.exp(log_b), torch.exp(log_c)

    tetragonal_or_hex = (
        (template == int(LatticeTemplate.TETRAGONAL))
        | (template == int(LatticeTemplate.TRIGONAL_HEX))
        | (template == int(LatticeTemplate.HEXAGONAL))
    )
    b = torch.where(tetragonal_or_hex, a, b)
    cubic = template == int(LatticeTemplate.CUBIC)
    b = torch.where(cubic, a, b)
    c = torch.where(cubic, a, c)
    rhombohedral = template == int(LatticeTemplate.RHOMBOHEDRAL)
    b = torch.where(rhombohedral, a, b)
    c = torch.where(rhombohedral, a, c)

    rho_ab, rho_ac, rho_bc = _correlations_from_q(value, template)
    one_minus_ab2 = (1.0 - rho_ab.square()).clamp_min(_CORRELATION_EPS)
    l00 = a
    l10 = b * rho_ab
    l11 = b * torch.sqrt(one_minus_ab2)
    l20 = c * rho_ac
    l21 = c * (rho_bc - rho_ab * rho_ac) / torch.sqrt(one_minus_ab2)
    det_corr = (
        1.0
        + 2.0 * rho_ab * rho_ac * rho_bc
        - rho_ab.square()
        - rho_ac.square()
        - rho_bc.square()
    ).clamp_min(_CORRELATION_EPS)
    l22 = c * torch.sqrt(det_corr / one_minus_ab2)
    zero = torch.zeros_like(l00)
    lattice = torch.stack(
        (
            torch.stack((l00, zero, zero), dim=-1),
            torch.stack((l10, l11, zero), dim=-1),
            torch.stack((l20, l21, l22), dim=-1),
        ),
        dim=-2,
    )
    return lattice.to(dtype=output_dtype)


def lattice_to_q(
    lattice: Tensor,
    *,
    spacegroup: Tensor,
    lattice_template_id: Optional[Tensor] = None,
) -> Tensor:
    """Encode a lattice metric into the selected constrained template.

    The source matrix may have any Cartesian orientation.  Only its Gram
    matrix is used, so decoding returns the equivalent canonical gauge.
    """

    _require_shape("lattice", lattice, (3, 3))
    value = lattice.to(torch.float64)
    template = resolve_lattice_template(
        spacegroup.to(device=lattice.device), lattice_template_id
    )
    if value.shape[:-2] != template.shape:
        raise ValueError(
            "lattice leading shape must match spacegroup, got "
            f"{tuple(value.shape[:-2])} versus {tuple(template.shape)}"
        )
    gram = value @ value.transpose(-1, -2)
    lengths = torch.sqrt(torch.diagonal(gram, dim1=-2, dim2=-1).clamp_min(_LENGTH_EPS))
    a, b, c = lengths.unbind(dim=-1)
    rho_ab = gram[..., 0, 1] / (a * b).clamp_min(_LENGTH_EPS)
    rho_ac = gram[..., 0, 2] / (a * c).clamp_min(_LENGTH_EPS)
    rho_bc = gram[..., 1, 2] / (b * c).clamp_min(_LENGTH_EPS)

    q = torch.zeros(*template.shape, Q_DIM, device=lattice.device, dtype=torch.float64)
    q[..., 0] = torch.log(a.clamp_min(_LENGTH_EPS))
    q[..., 1] = torch.log(b.clamp_min(_LENGTH_EPS))
    q[..., 2] = torch.log(c.clamp_min(_LENGTH_EPS))
    q[..., 3] = _safe_atanh(rho_ab)
    q[..., 4] = _safe_atanh(rho_ac)
    partial = (rho_bc - rho_ab * rho_ac) / torch.sqrt(
        ((1.0 - rho_ab.square()) * (1.0 - rho_ac.square())).clamp_min(_CORRELATION_EPS)
    )
    q[..., 5] = _safe_atanh(partial)

    tetragonal_or_hex = (
        (template == int(LatticeTemplate.TETRAGONAL))
        | (template == int(LatticeTemplate.TRIGONAL_HEX))
        | (template == int(LatticeTemplate.HEXAGONAL))
    )
    q[..., 0] = torch.where(
        tetragonal_or_hex, 0.5 * (torch.log(a) + torch.log(b)), q[..., 0]
    )
    cubic = template == int(LatticeTemplate.CUBIC)
    cubic_log = (torch.log(a) + torch.log(b) + torch.log(c)) / 3.0
    q[..., 0] = torch.where(cubic, cubic_log, q[..., 0])
    rhombohedral = template == int(LatticeTemplate.RHOMBOHEDRAL)
    q[..., 0] = torch.where(rhombohedral, cubic_log, q[..., 0])
    rho_r = ((rho_ab + rho_ac + rho_bc) / 3.0).clamp(
        -0.5 + _CORRELATION_EPS, 1.0 - _CORRELATION_EPS
    )
    unit = ((rho_r + 0.5) / 1.5).clamp(_CORRELATION_EPS, 1.0 - _CORRELATION_EPS)
    q[..., 3] = torch.where(rhombohedral, torch.logit(unit), q[..., 3])
    q = project_lattice_q(
        q,
        spacegroup=spacegroup.to(lattice.device),
        lattice_template_id=lattice_template_id,
    )
    output_dtype = (
        torch.float32 if lattice.dtype in (torch.float16, torch.bfloat16) else lattice.dtype
    )
    return q.to(dtype=output_dtype)


def lattice_metric(lattice: Tensor) -> Tensor:
    _require_shape("lattice", lattice, (3, 3))
    return lattice @ lattice.transpose(-1, -2)


def lattice_invariance_residual(lattice: Tensor, rotations: Tensor) -> Tensor:
    """Return relative ``||R^T G R - G||`` for row-vector symmetry ops."""

    _require_shape("lattice", lattice, (3, 3))
    _require_shape("rotations", rotations, (3, 3))
    if lattice.ndim != 2 or rotations.ndim != 3:
        raise ValueError("lattice_invariance_residual expects [3,3] and [G,3,3]")
    gram = lattice_metric(lattice).to(torch.float64)
    rot = rotations.to(device=lattice.device, dtype=torch.float64)
    transformed = rot.transpose(-1, -2) @ gram.unsqueeze(0) @ rot
    numerator = torch.linalg.matrix_norm(transformed - gram, dim=(-2, -1))
    denominator = torch.linalg.matrix_norm(gram).clamp_min(1.0e-12)
    return (numerator / denominator).to(dtype=lattice.dtype)


class ConstrainedLatticeDecoder(nn.Module):
    """Stateless module wrapper used at model and sampler boundaries."""

    q_dim = Q_DIM

    def forward(
        self,
        q: Tensor,
        spacegroup: Tensor,
        lattice_template_id: Optional[Tensor] = None,
    ) -> Tensor:
        return q_to_lattice(
            q,
            spacegroup=spacegroup,
            lattice_template_id=lattice_template_id,
        )

    def active_mask(
        self,
        spacegroup: Tensor,
        lattice_template_id: Optional[Tensor] = None,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        return q_active_mask(
            spacegroup=spacegroup,
            lattice_template_id=lattice_template_id,
            dtype=dtype,
        )




__all__ = [
    "ConstrainedLatticeDecoder",
    "LatticeTemplate",
    "NUM_LATTICE_TEMPLATES",
    "Q_DIM",
    "infer_lattice_template",
    "lattice_invariance_residual",
    "lattice_metric",
    "lattice_template_mask_table",
    "lattice_to_q",
    "project_lattice_q",
    "q_active_mask",
    "q_to_lattice",
    "require_lattice_template",
    "resolve_lattice_template",
]
