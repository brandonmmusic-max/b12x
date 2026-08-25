"""Weight-plan declarations for projection-tiered trellis checkpoints.

The b12x_trellis checkpoint standard's rate granularity surfaces in the
weight plan as ``trellis_rate_structure="per_expert_projection"`` with a
``trellis_projection_bits`` summary; planning fails closed outside the
projection-mixed runtime's qualified support (MCG bitrate sets within
{3, 4, 5}, PR #223).
"""

from __future__ import annotations

import pytest
import torch

from b12x.moe import fused_moe

_EXPERTS, _HIDDEN, _INTERMEDIATE = 4, 64, 1024


def _plan(**overrides):
    settings = dict(
        quant_modes="w4a16",
        source_format="btx",
        activation="situ",
        params_dtype=torch.float16,
        num_experts=_EXPERTS,
        hidden_size=_HIDDEN,
        intermediate_size=_INTERMEDIATE,
        trellis_bits=3,
        trellis_codebook="mcg",
        trellis_rate_structure="per_expert_projection",
        trellis_projection_bits=(3, 4, 5),
    )
    settings.update(overrides)
    return fused_moe.plan_weights(**settings)


def test_projection_plan_accepts_the_qualified_contract() -> None:
    plan = _plan()
    assert plan.trellis_rate_structure == "per_expert_projection"
    assert plan.trellis_projection_bits == frozenset({3, 4, 5})
    assert plan.trellis_pair_kinds is None


def test_projection_plan_accepts_bit_subsets() -> None:
    plan = _plan(trellis_projection_bits=(3, 4))
    assert plan.trellis_projection_bits == frozenset({3, 4})


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        (
            {"trellis_projection_bits": None},
            "declare trellis_projection_bits",
        ),
        ({"trellis_pair_kinds": ("P33",)}, "no trellis_pair_kinds"),
        ({"trellis_projection_bits": (3, 4, 6)}, "within {3, 4, 5}"),
        (
            {
                "trellis_codebook": "sqg_e4m3",
                "trellis_projection_bits": (3, 4),
            },
            "qualified only",
        ),
        ({"trellis_bits": 4}, "trellis_bits=3 base"),
        (
            {
                "trellis_rate_structure": "uniform",
                "trellis_projection_bits": (3,),
            },
            "no trellis_projection_bits",
        ),
        (
            {
                "trellis_rate_structure": "per_expert_pair",
                "trellis_pair_kinds": ("P33",),
            },
            "no trellis_projection_bits",
        ),
    ),
)
def test_projection_plan_validation_fails_closed(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        _plan(**overrides)
