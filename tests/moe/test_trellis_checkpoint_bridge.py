"""b12x_trellis checkpoint -> projection-mixed runtime bridge.

Qualifies the seam issue #242 asked for: native tier storage assembled
from a b12x_trellis checkpoint binds and executes through the
projection-mixed MCG K3/K4/K5 runtime (PR #223). The suite self-skips on
trees without that runtime, so it ships with the checkpoint-standard PR
and activates when the runtime lands.

Two angles:

- Degenerate per-expert tiering (gate == up == down per expert), where a
  per-tier serial reference exists: checkpoint -> reader -> assembly ->
  three-tier kernel must match the serial answer and replay identically
  under CUDA graph capture.
- Divergent projection tiering with unit scale vectors: tier storage
  built directly from the source payload versus through the checkpoint
  round-trip must produce bitwise-equal kernel output. This arm
  qualifies payload transparency under unit scale vectors; scale-row
  placement is exercised by the degenerate arm's serial comparison.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("cutlass")

mixed = pytest.importorskip("b12x.moe._shared.kernels.w4a16.mixed_trellis")
for symbol in (
    "compile_mixed_trellis3",
    "build_projection_tiered_maps",
    "combine_trellis_rotations",
    "bind_mixed_trellis3",
    "make_mixed_trellis3_buffers",
    "run_bound_mixed_trellis3",
):
    if not hasattr(mixed, symbol):
        pytest.skip(
            "projection-mixed MCG K3/K4/K5 runtime (PR #223) is not on this tree",
            allow_module_level=True,
        )

from b12x.moe._shared.kernels.w4a16.host import (  # noqa: E402
    make_w4a16_packed_buffers,
)
from b12x.moe._shared.kernels.w4a16.kernel import (  # noqa: E402
    run_w4a16_moe,
)
from b12x.moe._shared.kernels.w4a16.prepare import (  # noqa: E402
    prepare_trellis256_moe_weights,
)
from b12x.moe._shared.kernels.w4a16.trellis_checkpoint_reader import (  # noqa: E402
    assemble_trellis_projection_weights,
    read_trellis_checkpoint,
    read_trellis_checkpoint_layer,
)

from .test_trellis_checkpoint import _write_checkpoint  # noqa: E402

_HIDDEN = 128
_INTERMEDIATE = 128
_TILE = (128, 128, 128, 128)


def _sm12x_available() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return major == 12 and minor in (0, 1)


def _write_bridge_checkpoint(root, triples, *, unit_scales: bool):
    return _write_checkpoint(
        root,
        hidden=_HIDDEN,
        intermediate=_INTERMEDIATE,
        moe_layers=(1,),
        triples=triples,
        block_size=_INTERMEDIATE,
        unit_scales=unit_scales,
    )


def _tier_prepared(assembled, *, device: torch.device) -> tuple[list, list[int]]:
    """Per-tier PreparedW4A16MoeWeights + slot counts from assembly output.

    Rotation and hidden-scale rows are placed by each projection's own
    tier membership (gate rows by gate members, up by up, down by down),
    matching the per-projection descriptor namespace.
    """

    experts = len(assembled.gate_tiers)
    local = assembled.channel_count
    tiers = []
    counts = []
    for tier in range(len(assembled.tier_bits)):
        gate_ids = [e for e in range(experts) if assembled.gate_tiers[e] == tier]
        up_ids = [e for e in range(experts) if assembled.up_tiers[e] == tier]
        down_ids = [e for e in range(experts) if assembled.down_tiers[e] == tier]
        slots = max(len(gate_ids), len(up_ids), len(down_ids))
        counts.append(slots)
        bits = assembled.tier_bits[tier]

        def _padded(
            stack_map, fc1: bool, *, bits=bits, tier=tier, slots=slots
        ) -> torch.Tensor:
            shape = (
                (_HIDDEN // 16, local // 16, 16 * bits)
                if fc1
                else (local // 16, _HIDDEN // 16, 16 * bits)
            )
            rows = list(stack_map.get(tier, torch.empty((0, *shape))))
            while len(rows) < slots:
                rows.append(torch.zeros(shape, dtype=torch.int16, device=device))
            return torch.stack([row.to(device=device) for row in rows], dim=0)

        w13 = torch.stack(
            [
                _padded(assembled.gate_weights, True),
                _padded(assembled.up_weights, True),
            ],
            dim=0,
        )
        w2 = _padded(assembled.down_weights, False)

        def _segment_rows(member_ids, segment: int, *, slots=slots) -> torch.Tensor:
            # One projection's [slots, local] slice of the rotation table,
            # rows placed by that projection's tier-local membership.
            table = torch.ones((slots, local), dtype=torch.float16, device=device)
            begin = segment * local
            for slot, expert in enumerate(member_ids):
                table[slot] = assembled.intermediate_rotations[
                    expert, begin : begin + local
                ]
            return table

        rotations = torch.cat(
            [
                _segment_rows(gate_ids, 0),
                _segment_rows(up_ids, 1),
                _segment_rows(down_ids, 2),
            ],
            dim=1,
        )

        def _hidden_rows(
            table: torch.Tensor, member_ids, *, slots=slots
        ) -> torch.Tensor:
            moved = table.to(device=device)
            if moved.shape[0] == 1:
                return moved.expand(slots, -1).contiguous()
            rows = torch.ones((slots, _HIDDEN), dtype=torch.float16, device=device)
            for slot, expert in enumerate(member_ids):
                rows[slot] = moved[expert]
            return rows

        tiers.append(
            prepare_trellis256_moe_weights(
                w13=w13,
                w2=w2,
                hidden_size=_HIDDEN,
                intermediate_size=local,
                num_experts=slots,
                activation="silu",
                fc1_tile_n=_TILE[1],
                fc2_tile_n=_TILE[3],
                device=device,
                params_dtype=torch.float16,
                w13_layout="trellis_t256_proj",
                trellis_bits=bits,
                codebook="mcg",
                gate_suh=_hidden_rows(assembled.gate_suh, gate_ids),
                up_suh=_hidden_rows(assembled.up_suh, up_ids),
                intermediate_rotations=rotations,
                down_svh=_hidden_rows(assembled.down_svh, down_ids),
                tile_config=_TILE,
            )
        )
    return tiers, counts


def _run_mixed3(assembled, *, device, x, topk_ids, topk_weights):
    tiers, counts = _tier_prepared(assembled, device=device)
    if len(tiers) != 3:
        raise AssertionError("bridge fixtures must populate three tiers")
    props = torch.cuda.get_device_properties(device)
    launch = mixed.compile_mixed_trellis3(
        size_m=int(x.shape[0]),
        hidden_size=_HIDDEN,
        intermediate_size=_INTERMEDIATE,
        tier0_num_experts=counts[0],
        tier1_num_experts=counts[1],
        tier2_num_experts=counts[2],
        top_k=int(topk_ids.shape[1]),
        max_m_blocks=8,
        sms=int(props.multi_processor_count),
        max_shared_mem=int(props.shared_memory_per_block_optin),
        force_tile_config=_TILE,
        trellis_codebook="mcg",
    )
    global_to_combined, descriptor = mixed.build_projection_tiered_maps(
        list(assembled.gate_tiers),
        list(assembled.up_tiers),
        list(assembled.down_tiers),
        tier_slots=tuple(counts),
        device=device,
    )
    rotations = mixed.combine_trellis_rotations(*tiers)
    buffers = mixed.make_mixed_trellis3_buffers(
        launch, device=device, sms=int(props.multi_processor_count)
    )
    binding = mixed.bind_mixed_trellis3(
        *tiers, global_to_combined, descriptor, rotations, launch
    )
    out = mixed.run_bound_mixed_trellis3(
        x, topk_weights, topk_ids, binding, buffers
    ).clone()
    torch.cuda.synchronize(device)
    return out, (binding, buffers)


def _serial_tier(x, prepared, topk_weights, topk_ids, expert_map):
    m, topk = int(topk_ids.shape[0]), int(topk_ids.shape[1])
    buffers = make_w4a16_packed_buffers(
        prepared,
        m=m,
        topk=topk,
        dtype=torch.float16,
        device=x.device,
        route_num_experts=int(expert_map.numel()),
        full_rotation=True,
        block_size_m=8,
    )
    return run_w4a16_moe(
        x,
        prepared,
        topk_weights,
        topk_ids,
        activation="silu",
        intermediate_cache13=buffers.intermediate_cache13,
        intermediate_cache2=buffers.intermediate_cache2,
        output=buffers.output,
        fc1_c_tmp=buffers.fc1_c_tmp,
        fc2_c_tmp=buffers.fc2_c_tmp,
        packed_route_indices=buffers.packed_route_indices,
        block_expert_ids=buffers.block_expert_ids,
        packed_route_count=buffers.packed_route_count,
        expert_offsets=buffers.expert_offsets,
        expert_counts=buffers.expert_counts,
        expert_map=expert_map,
        output_expert_map=expert_map,
        route_block_size_m=8,
        intermediate_rotation_scales=prepared.intermediate_rotations,
        full_rotation=True,
        suh_gate_table=prepared.gate_suh,
        suh_up_table=prepared.up_suh,
        svh_table=prepared.down_svh,
        rotation_a_gate=buffers.rotation_a_gate,
        rotation_a_up=buffers.rotation_a_up,
    )


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
def test_checkpoint_degenerate_tiering_matches_serial(tmp_path) -> None:
    triples = ((3, 3, 3), (3, 3, 3), (4, 4, 4), (4, 4, 4), (5, 5, 5), (5, 5, 5))
    _write_bridge_checkpoint(tmp_path, triples, unit_scales=False)
    device = torch.device("cuda", torch.cuda.current_device())
    checkpoint = read_trellis_checkpoint(tmp_path)
    layer = read_trellis_checkpoint_layer(
        checkpoint, 1, first_channel=0, channel_count=_INTERMEDIATE
    )
    assembled = assemble_trellis_projection_weights(layer, device=device)

    torch.manual_seed(20260824)
    x = (torch.randn((2, _HIDDEN), device=device) * 1.0e-3).to(torch.bfloat16)
    topk_ids = torch.tensor([[0, 2, 4], [5, 3, 1]], dtype=torch.int32, device=device)
    topk_weights = torch.tensor(
        [[0.5, 0.3, 0.2], [0.25, 0.25, 0.5]],
        dtype=torch.float32,
        device=device,
    )
    out, (binding, buffers) = _run_mixed3(
        assembled,
        device=device,
        x=x,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )
    assert not torch.isnan(out).any()
    assert torch.count_nonzero(out).item() > 0

    tiers, _counts = _tier_prepared(assembled, device=device)
    tier_maps = (
        torch.tensor([0, 1, -1, -1, -1, -1], dtype=torch.int32, device=device),
        torch.tensor([-1, -1, 0, 1, -1, -1], dtype=torch.int32, device=device),
        torch.tensor([-1, -1, -1, -1, 0, 1], dtype=torch.int32, device=device),
    )
    serial = sum(
        (
            _serial_tier(x, tier, topk_weights, topk_ids, expert_map)
            for tier, expert_map in zip(tiers, tier_maps, strict=True)
        ),
        torch.zeros((2, _HIDDEN), dtype=torch.float32, device=device),
    )
    relative = (out - serial).norm() / serial.norm().clamp_min(1.0e-12)
    assert float(relative) < 4.0e-3

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = mixed.run_bound_mixed_trellis3(
            x, topk_weights, topk_ids, binding, buffers
        )
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(captured, out)


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
def test_checkpoint_roundtrip_is_bitwise_transparent(tmp_path) -> None:
    """Direct-from-payload vs through-checkpoint tier storage: equal output."""

    triples = ((3, 4, 5), (4, 5, 3), (5, 3, 4), (3, 3, 4), (4, 4, 5), (5, 5, 3))
    source = _write_bridge_checkpoint(tmp_path, triples, unit_scales=True)
    device = torch.device("cuda", torch.cuda.current_device())
    checkpoint = read_trellis_checkpoint(tmp_path)
    layer = read_trellis_checkpoint_layer(
        checkpoint, 1, first_channel=0, channel_count=_INTERMEDIATE
    )
    assembled = assemble_trellis_projection_weights(layer, device=device)

    # Arm B replaces every payload slice with the SOURCE tensors, bypassing
    # the container read path entirely; everything else is identical.
    from dataclasses import replace

    direct_trellis = {
        key: source["payload"][(1, *key)].to(device=device) for key in layer.trellis
    }
    direct_layer = replace(layer, trellis=direct_trellis)
    direct = assemble_trellis_projection_weights(direct_layer, device=device)

    torch.manual_seed(20260824)
    x = (torch.randn((2, _HIDDEN), device=device) * 1.0e-3).to(torch.bfloat16)
    topk_ids = torch.tensor([[0, 2, 4], [5, 3, 1]], dtype=torch.int32, device=device)
    topk_weights = torch.tensor(
        [[0.5, 0.3, 0.2], [0.25, 0.25, 0.5]],
        dtype=torch.float32,
        device=device,
    )
    out_checkpoint, _ = _run_mixed3(
        assembled,
        device=device,
        x=x,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )
    out_direct, _ = _run_mixed3(
        direct,
        device=device,
        x=x,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )
    assert not torch.isnan(out_checkpoint).any()
    assert torch.count_nonzero(out_checkpoint).item() > 0
    assert torch.equal(out_checkpoint, out_direct)
