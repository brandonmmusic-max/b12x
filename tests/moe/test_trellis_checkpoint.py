"""b12x_trellis checkpoint standard: declarations, reader, tier assembly.

Host-side suite for the version-2 in-checkpoint standard
(`docs/b12x-trellis-checkpoint-format.md`): configuration validation,
metadata-tensor contracts, topology-neutral extent slicing, and native
tier assembly. Fused execution belongs to the projection-mixed MCG
K3/K4/K5 runtime (PR #223); this suite pins the checkpoint contract that
runtime consumes.
"""

from __future__ import annotations

import copy
import json
import pathlib

import pytest
import torch
from safetensors.torch import save_file

from b12x.moe._shared.trellis_checkpoint import (
    TrellisCheckpointConfig,
    rate_byte,
    rate_byte_bits,
)
from b12x.moe._shared.kernels.w4a16.trellis_checkpoint_reader import (
    assemble_trellis_projection_weights,
    prepare_trellis_checkpoint_moe_weights,
    read_trellis_checkpoint,
    read_trellis_checkpoint_layer,
)

_HIDDEN = 64
_INTERMEDIATE = 512  # four 128-channel transform blocks
_EXPERTS = 4
_MOE_LAYERS = (2, 3)
_TRIPLES = ((3, 4, 5), (3, 3, 3), (4, 4, 3), (5, 3, 4))

_QUANT_BLOCK = {
    "quant_method": "b12x_trellis",
    "b12x_trellis": {
        "version": 2,
        "codebook": "mcg",
        "rate": {"granularity": "per_expert_projection"},
        "scale": {
            "input_scales": {"vectors": "per_layer", "gains": "none"},
            "intermediate_scales": {
                "vectors": "per_expert",
                "gains": "none",
            },
            "output_scales": {"vectors": "per_layer", "gains": "none"},
        },
        "transform": {
            "projection": {"kind": "scaled_hadamard", "block_size": 128},
            "expert": {"kind": "none"},
        },
    },
}


def _write_checkpoint(root: pathlib.Path, *, with_index: bool = True) -> dict:
    """Write a synthetic v2 checkpoint; return its source tensors."""

    generator = torch.Generator().manual_seed(20260824)
    root.mkdir(parents=True, exist_ok=True)
    L, E = len(_MOE_LAYERS), _EXPERTS
    config = {
        "hidden_size": _HIDDEN,
        "moe_intermediate_size": _INTERMEDIATE,
        "n_routed_experts": E,
        "first_k_dense_replace": _MOE_LAYERS[0],
        "num_hidden_layers": _MOE_LAYERS[-1] + 1,
        "quantization_config": copy.deepcopy(_QUANT_BLOCK),
    }
    (root / "config.json").write_text(json.dumps(config, indent=1))

    rate = torch.zeros((L, E, 3), dtype=torch.uint8)
    input_scales = torch.zeros((L, _HIDDEN), dtype=torch.float16)
    output_scales = torch.zeros((L, _HIDDEN), dtype=torch.float16)
    inter = torch.zeros((L, E, 3, _INTERMEDIATE), dtype=torch.float16)
    source: dict = {"payload": {}, "weight_map": {}}

    def _values(shape):
        raw = torch.rand(shape, generator=generator, dtype=torch.float32)
        return (0.5 + raw).to(torch.float16)

    for row, layer in enumerate(_MOE_LAYERS):
        tensors = {}
        input_scales[row] = _values((_HIDDEN,))
        output_scales[row] = _values((_HIDDEN,))
        for expert, triple in enumerate(_TRIPLES):
            for pi, (proj, bits) in enumerate(
                zip(("gate_proj", "up_proj", "down_proj"), triple, strict=True)
            ):
                rate[row, expert, pi] = rate_byte(bits)
                inter[row, expert, pi] = _values((_INTERMEDIATE,))
                fc1 = pi < 2
                shape = (
                    (_HIDDEN // 16, _INTERMEDIATE // 16, 16 * bits)
                    if fc1
                    else (_INTERMEDIATE // 16, _HIDDEN // 16, 16 * bits)
                )
                name = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.trellis"
                payload = torch.randint(
                    -(1 << 15),
                    1 << 15,
                    shape,
                    dtype=torch.int16,
                    generator=generator,
                )
                tensors[name] = payload
                source["payload"][(layer, expert, pi)] = payload
        filename = f"model-layer-{layer:03d}.safetensors"
        save_file(tensors, str(root / filename))
        for name in tensors:
            source["weight_map"][name] = filename

    metadata = {
        "b12x_trellis.rate": rate,
        "b12x_trellis.input_scales": input_scales,
        "b12x_trellis.intermediate_scales": inter,
        "b12x_trellis.output_scales": output_scales,
    }
    save_file(metadata, str(root / "b12x-trellis.safetensors"))
    source.update(
        rate=rate,
        input_scales=input_scales,
        intermediate_scales=inter,
        output_scales=output_scales,
    )
    if with_index:
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": source["weight_map"]})
        )
    return source


def test_rate_byte_helpers_reject_split_nibbles() -> None:
    assert rate_byte(3) == 0x33 and rate_byte(5) == 0x55
    assert rate_byte_bits(0x44) == 4
    with pytest.raises(ValueError, match="differing nibbles"):
        rate_byte_bits(0x34)


def test_config_block_parses() -> None:
    config = TrellisCheckpointConfig.from_quantization_config(_QUANT_BLOCK)
    assert config.codebook == "mcg"
    assert config.rate_granularity == "per_expert_projection"
    assert config.input_scales.vectors == "per_layer"
    assert config.intermediate_scales.vectors == "per_expert"
    assert config.transform.projection_block_size == 128


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda b: b.update(quant_method="exl3"), "quant_method"),
        (lambda b: b["b12x_trellis"].update(version=1), "version"),
        (lambda b: b["b12x_trellis"].update(codebook="mcg2"), "codebook"),
        (
            lambda b: b["b12x_trellis"]["rate"].update(granularity="per_expert"),
            "granularity",
        ),
        (
            lambda b: b["b12x_trellis"]["scale"]["input_scales"].update(
                gains="per_expert"
            ),
            "gains",
        ),
        (
            lambda b: b["b12x_trellis"]["scale"]["output_scales"].update(
                vectors="per_pair"
            ),
            "vectors",
        ),
        (
            lambda b: b["b12x_trellis"]["transform"]["projection"].update(
                kind="hadamard"
            ),
            "scaled_hadamard",
        ),
        (
            lambda b: b["b12x_trellis"]["transform"]["projection"].update(
                block_size=100
            ),
            "multiple of 32",
        ),
        (
            lambda b: b["b12x_trellis"].update(extra=1),
            "unknown keys",
        ),
    ),
)
def test_config_block_fails_closed(mutate, message) -> None:
    block = copy.deepcopy(_QUANT_BLOCK)
    mutate(block)
    with pytest.raises(ValueError, match=message):
        TrellisCheckpointConfig.from_quantization_config(block)


@pytest.mark.parametrize("with_index", (True, False))
def test_checkpoint_discovery_resolves_all_tensors(tmp_path, with_index) -> None:
    _write_checkpoint(tmp_path, with_index=with_index)
    checkpoint = read_trellis_checkpoint(tmp_path)
    assert checkpoint.moe_layer_indices == _MOE_LAYERS
    assert checkpoint.num_experts == _EXPERTS
    trellis_names = [name for name in checkpoint.locations if name.endswith(".trellis")]
    assert len(trellis_names) == len(_MOE_LAYERS) * _EXPERTS * 3
    assert "b12x_trellis.rate" in checkpoint.locations


def test_layer_extents_slice_losslessly(tmp_path) -> None:
    source = _write_checkpoint(tmp_path)
    checkpoint = read_trellis_checkpoint(tmp_path)
    for first, count in ((0, _INTERMEDIATE), (0, 128), (256, 128), (128, 256)):
        layer = read_trellis_checkpoint_layer(
            checkpoint, 3, first_channel=first, channel_count=count
        )
        assert layer.local_intermediate_size == count
        row = _MOE_LAYERS.index(3)
        for expert, triple in enumerate(_TRIPLES):
            for pi, bits in enumerate(triple):
                assert int(layer.rates[expert, pi]) == bits
                full = source["payload"][(3, expert, pi)]
                expected = (
                    full[:, first // 16 : (first + count) // 16]
                    if pi < 2
                    else full[first // 16 : (first + count) // 16]
                )
                assert torch.equal(layer.trellis[(expert, pi)], expected)
                assert torch.equal(
                    layer.intermediate_scales[expert, pi],
                    source["intermediate_scales"][
                        row, expert, pi, first : first + count
                    ],
                )
        assert torch.equal(layer.gate_suh[0], source["input_scales"][row])
        assert torch.equal(layer.up_suh[0], source["input_scales"][row])
        assert torch.equal(layer.down_svh[0], source["output_scales"][row])


def test_extent_and_layer_validation_fail_closed(tmp_path) -> None:
    _write_checkpoint(tmp_path)
    checkpoint = read_trellis_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="align"):
        read_trellis_checkpoint_layer(
            checkpoint, 3, first_channel=64, channel_count=128
        )
    with pytest.raises(ValueError, match="not one of the checkpoint's"):
        read_trellis_checkpoint_layer(checkpoint, 0, first_channel=0, channel_count=128)


def test_split_nibble_rate_bytes_fail_closed(tmp_path) -> None:
    _write_checkpoint(tmp_path)
    from safetensors import safe_open

    path = tmp_path / "b12x-trellis.safetensors"
    with safe_open(str(path), framework="pt") as handle:
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}
    tensors["b12x_trellis.rate"][0, 0, 0] = 0x34
    save_file(tensors, str(path))
    checkpoint = read_trellis_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="differing nibbles"):
        read_trellis_checkpoint_layer(
            checkpoint, 2, first_channel=0, channel_count=_INTERMEDIATE
        )


def test_payload_width_must_match_rate(tmp_path) -> None:
    _write_checkpoint(tmp_path)
    from safetensors import safe_open

    shard = tmp_path / "model-layer-002.safetensors"
    with safe_open(str(shard), framework="pt") as handle:
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}
    name = "model.layers.2.mlp.experts.0.gate_proj.trellis"
    tensors[name] = torch.zeros(
        (_HIDDEN // 16, _INTERMEDIATE // 16, 16 * 4), dtype=torch.int16
    )  # rate says K3
    save_file(tensors, str(shard))
    checkpoint = read_trellis_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="disagrees with the rate tensor"):
        read_trellis_checkpoint_layer(
            checkpoint, 2, first_channel=0, channel_count=_INTERMEDIATE
        )


def test_assembly_groups_tiers_byte_identically(tmp_path) -> None:
    source = _write_checkpoint(tmp_path)
    checkpoint = read_trellis_checkpoint(tmp_path)
    layer = read_trellis_checkpoint_layer(
        checkpoint, 3, first_channel=128, channel_count=256
    )
    assembled = assemble_trellis_projection_weights(layer, device="cpu")

    assert assembled.tier_bits == (3, 4, 5)
    tier_of = {bits: tier for tier, bits in enumerate(assembled.tier_bits)}
    for pi, tiers in enumerate(
        (assembled.gate_tiers, assembled.up_tiers, assembled.down_tiers)
    ):
        assert tiers == tuple(tier_of[triple[pi]] for triple in _TRIPLES)

    weights = {
        0: assembled.gate_weights,
        1: assembled.up_weights,
        2: assembled.down_weights,
    }
    for pi in range(3):
        locals_seen: dict[int, int] = {}
        for expert, triple in enumerate(_TRIPLES):
            tier = tier_of[triple[pi]]
            local = locals_seen.get(tier, 0)
            locals_seen[tier] = local + 1
            native = weights[pi][tier][local]
            full = source["payload"][(3, expert, pi)]
            expected = (
                full[:, 8:24] if pi < 2 else full[8:24]
            )  # channels 128..384 = tiles 8..24
            assert torch.equal(native, expected)

    local = layer.local_intermediate_size
    assert assembled.intermediate_rotations.shape == (_EXPERTS, 3 * local)
    row = _MOE_LAYERS.index(3)
    expert = 2
    manual = torch.cat(
        [source["intermediate_scales"][row, expert, pi, 128:384] for pi in range(3)]
    )
    assert torch.equal(assembled.intermediate_rotations[expert], manual)


def test_prepare_fails_closed_toward_pr223(tmp_path) -> None:
    _write_checkpoint(tmp_path)
    checkpoint = read_trellis_checkpoint(tmp_path)
    layer = read_trellis_checkpoint_layer(
        checkpoint, 2, first_channel=0, channel_count=128
    )
    with pytest.raises(ValueError, match="PR #223"):
        prepare_trellis_checkpoint_moe_weights(layer)
