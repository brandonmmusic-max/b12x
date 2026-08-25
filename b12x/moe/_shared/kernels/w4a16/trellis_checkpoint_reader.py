"""b12x_trellis checkpoint reading, TP slicing, and native tier assembly.

One metadata-driven load path serves every declared configuration of the
`b12x_trellis` version-2 checkpoint standard
(`docs/b12x-trellis-checkpoint-format.md`): the configuration block and
the four global metadata tensors locate and validate every value, rank
extents slice the topology-neutral whole-matrix payload on the expert
intermediate axis, and `assemble_trellis_projection_weights` restores
native per-tier storage in exactly the form the projection-mixed MCG
K3/K4/K5 runtime's tier maps consume (PR #223). Nothing in this module
depends on a specific model geometry, producer, or shard naming
convention.
"""

from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass

import torch

from b12x.moe._shared.trellis_checkpoint import (
    INPUT_SCALES_TENSOR,
    INTERMEDIATE_SCALES_TENSOR,
    METADATA_TENSORS,
    OUTPUT_SCALES_TENSOR,
    PROJECTIONS,
    RATE_TENSOR,
    SCALE_VECTORS_PER_LAYER,
    TrellisCheckpointConfig,
    rate_byte_bits,
)

_EXPERT_TRELLIS = re.compile(
    r"^(?P<prefix>.+\.experts)\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.trellis$"
)
_LAYER = re.compile(r"\.layers\.(\d+)\.")
_PROJ_INDEX = {"gate_proj": 0, "up_proj": 1, "down_proj": 2}


@dataclass(frozen=True)
class TrellisCheckpoint:
    """A validated checkpoint directory with resolved tensor locations."""

    root: pathlib.Path
    config: TrellisCheckpointConfig
    num_experts: int
    hidden_size: int
    intermediate_size: int
    moe_layer_indices: tuple[int, ...]
    # tensor name -> shard filename, for the metadata tensors and every
    # per-expert trellis tensor.
    locations: dict[str, str]


@dataclass(frozen=True)
class TrellisCheckpointLayer:
    """One layer's rank extent: decoded rates, scales, and payload slices."""

    checkpoint: TrellisCheckpoint
    layer_index: int
    first_channel: int
    channel_count: int
    # [E, 3] int bitrates decoded from the rate tensor.
    rates: torch.Tensor
    # fp16 [E, 3, local] intermediate-boundary vectors for this extent.
    intermediate_scales: torch.Tensor
    # fp16 [1, H] or [E, H] hidden-axis vectors.
    gate_suh: torch.Tensor
    up_suh: torch.Tensor
    down_svh: torch.Tensor
    # per (expert, projection) int16 payload slices, native-shaped:
    # gate/up [H/16, local/16, 16*K]; down [local/16, H/16, 16*K].
    trellis: dict[tuple[int, int], torch.Tensor]

    @property
    def local_intermediate_size(self) -> int:
        return self.channel_count


@dataclass(frozen=True)
class TrellisProjectionTierWeights:
    """Native tier storage for the projection-mixed runtime's tier maps.

    ``tier_bits`` is the ascending bitrate list; tier IDs index it.
    ``gate_tiers``/``up_tiers``/``down_tiers`` hold one tier ID per global
    expert in global order — the inputs of ``build_projection_tiered_maps``.
    ``gate_weights``/``up_weights`` map tier IDs to int16
    ``[count, hidden/16, 2*slots, 16*bits]`` native FC1 tensors and
    ``down_weights`` to int16 ``[count, 2*slots, hidden/16, 16*bits]``
    native FC2 tensors, tier-local experts ordered by ascending global ID.
    """

    layer_index: int
    first_channel: int
    channel_count: int
    tier_bits: tuple[int, ...]
    gate_tiers: tuple[int, ...]
    up_tiers: tuple[int, ...]
    down_tiers: tuple[int, ...]
    gate_weights: dict[int, torch.Tensor]
    up_weights: dict[int, torch.Tensor]
    down_weights: dict[int, torch.Tensor]
    gate_suh: torch.Tensor
    up_suh: torch.Tensor
    down_svh: torch.Tensor
    intermediate_rotations: torch.Tensor


def read_trellis_checkpoint(root: str | pathlib.Path) -> TrellisCheckpoint:
    """Parse and validate a checkpoint directory's declarations.

    Tensor locations resolve through the safetensors index when present
    and fall back to scanning shard headers, so metadata shards outside
    the model index are found without convention.
    """

    root = pathlib.Path(root)
    model_config = json.loads((root / "config.json").read_text())
    config = TrellisCheckpointConfig.from_quantization_config(
        model_config.get("quantization_config") or {}
    )

    def _geometry(key: str) -> int:
        value = model_config.get(key)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"b12x_trellis checkpoints require config.json {key!r}")
        return value

    hidden = _geometry("hidden_size")
    intermediate = _geometry("moe_intermediate_size")
    experts = _geometry("n_routed_experts")
    first_dense = model_config.get("first_k_dense_replace", 0)
    num_layers = _geometry("num_hidden_layers")
    moe_layers = tuple(range(int(first_dense), num_layers))
    if not moe_layers:
        raise ValueError("b12x_trellis checkpoints declare no MoE layers")

    locations: dict[str, str] = {}
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text()).get("weight_map", {})
        for name, filename in weight_map.items():
            if name in METADATA_TENSORS or _EXPERT_TRELLIS.match(name):
                locations[name] = filename

    missing = [name for name in METADATA_TENSORS if name not in locations]
    needs_payload_scan = not any(_EXPERT_TRELLIS.match(name) for name in locations)
    if missing or needs_payload_scan:
        from safetensors import safe_open

        for shard in sorted(root.glob("*.safetensors")):
            with safe_open(str(shard), framework="pt") as handle:
                for name in handle.keys():
                    if name in locations:
                        continue
                    if name in METADATA_TENSORS or _EXPERT_TRELLIS.match(name):
                        locations[name] = shard.name
        missing = [name for name in METADATA_TENSORS if name not in locations]
    if missing:
        raise ValueError(f"b12x_trellis metadata tensors not found: {missing}")

    return TrellisCheckpoint(
        root=root,
        config=config,
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        moe_layer_indices=moe_layers,
        locations=locations,
    )


def _load(checkpoint: TrellisCheckpoint, name: str) -> torch.Tensor:
    from safetensors import safe_open

    filename = checkpoint.locations.get(name)
    if filename is None:
        raise ValueError(f"b12x_trellis tensor {name!r} has no location")
    with safe_open(str(checkpoint.root / filename), framework="pt") as handle:
        return handle.get_tensor(name)


def _load_slice(
    checkpoint: TrellisCheckpoint,
    name: str,
    *,
    dim: int,
    begin: int,
    end: int,
) -> torch.Tensor:
    from safetensors import safe_open

    filename = checkpoint.locations.get(name)
    if filename is None:
        raise ValueError(f"b12x_trellis tensor {name!r} has no location")
    with safe_open(str(checkpoint.root / filename), framework="pt") as handle:
        view = handle.get_slice(name)
        if dim == 0:
            return view[begin:end]
        if dim == 1:
            return view[:, begin:end]
        raise ValueError(f"unsupported slice dim {dim}")


def read_trellis_checkpoint_layer(
    checkpoint: TrellisCheckpoint,
    layer_index: int,
    *,
    first_channel: int,
    channel_count: int,
) -> TrellisCheckpointLayer:
    """Load one rank extent of one layer as CPU tensors, fail-closed.

    Every rate byte is decoded and validated against the codebook, every
    payload tensor's bit width is validated against its rate entry, and
    every metadata tensor's shape is validated against the declared
    granularity before any value is returned.
    """

    config = checkpoint.config
    if layer_index not in checkpoint.moe_layer_indices:
        raise ValueError(
            f"layer {layer_index} is not one of the checkpoint's MoE "
            f"layers {checkpoint.moe_layer_indices[0]}.."
            f"{checkpoint.moe_layer_indices[-1]}"
        )
    config.validate_extent(first_channel, channel_count, checkpoint.intermediate_size)
    row = checkpoint.moe_layer_indices.index(layer_index)
    L = len(checkpoint.moe_layer_indices)
    E = checkpoint.num_experts
    H = checkpoint.hidden_size
    inter_size = checkpoint.intermediate_size

    rate = _load(checkpoint, RATE_TENSOR)
    if tuple(rate.shape) != config.rate_shape(L, E) or rate.dtype != torch.uint8:
        raise ValueError(
            f"{RATE_TENSOR} must be uint8 {config.rate_shape(L, E)}, got "
            f"{rate.dtype} {tuple(rate.shape)}"
        )
    rates = torch.empty((E, len(PROJECTIONS)), dtype=torch.int64)
    for expert in range(E):
        for pi in range(len(PROJECTIONS)):
            bits = rate_byte_bits(int(rate[row, expert, pi]))
            config.validate_bits(bits)
            rates[expert, pi] = bits

    def _hidden(name: str, family) -> torch.Tensor:
        tensor = _load(checkpoint, name)
        expected = config.hidden_scales_shape(family, L, E, H)
        if tuple(tensor.shape) != expected or tensor.dtype != torch.float16:
            raise ValueError(
                f"{name} must be fp16 {expected}, got "
                f"{tensor.dtype} {tuple(tensor.shape)}"
            )
        selected = tensor[row]
        return (
            selected.reshape(1, H)
            if family.vectors == SCALE_VECTORS_PER_LAYER
            else selected.reshape(E, H)
        ).contiguous()

    gate_up = _hidden(INPUT_SCALES_TENSOR, config.input_scales)
    down_svh = _hidden(OUTPUT_SCALES_TENSOR, config.output_scales)

    inter = _load(checkpoint, INTERMEDIATE_SCALES_TENSOR)
    expected = config.intermediate_scales_shape(L, E, inter_size)
    if tuple(inter.shape) != expected or inter.dtype != torch.float16:
        raise ValueError(
            f"{INTERMEDIATE_SCALES_TENSOR} must be fp16 {expected}, got "
            f"{inter.dtype} {tuple(inter.shape)}"
        )
    selected = inter[row]
    if config.intermediate_scales.vectors == SCALE_VECTORS_PER_LAYER:
        selected = selected.reshape(1, len(PROJECTIONS), inter_size).expand(
            E, len(PROJECTIONS), inter_size
        )
    intermediate = selected[
        :, :, first_channel : first_channel + channel_count
    ].contiguous()

    tile_begin = first_channel // 16
    tile_end = (first_channel + channel_count) // 16
    hidden_tiles = H // 16
    trellis: dict[tuple[int, int], torch.Tensor] = {}
    names = {
        (int(m.group("expert")), _PROJ_INDEX[m.group("proj")]): name
        for name in checkpoint.locations
        for m in (_EXPERT_TRELLIS.match(name),)
        if m and _layer_of(name) == layer_index
    }
    for expert in range(E):
        for pi, _proj in enumerate(PROJECTIONS):
            name = names.get((expert, pi))
            if name is None:
                raise ValueError(
                    f"layer {layer_index} expert {expert} "
                    f"{PROJECTIONS[pi]} has no trellis tensor"
                )
            fc1 = pi < 2
            piece = _load_slice(
                checkpoint,
                name,
                dim=1 if fc1 else 0,
                begin=tile_begin,
                end=tile_end,
            )
            bits = int(rates[expert, pi])
            expected_shape = (
                (hidden_tiles, tile_end - tile_begin, 16 * bits)
                if fc1
                else (tile_end - tile_begin, hidden_tiles, 16 * bits)
            )
            if tuple(piece.shape) != expected_shape or piece.dtype != torch.int16:
                raise ValueError(
                    f"{name} slice must be int16 {expected_shape}, got "
                    f"{piece.dtype} {tuple(piece.shape)}; the payload "
                    "disagrees with the rate tensor or geometry"
                )
            trellis[(expert, pi)] = piece

    return TrellisCheckpointLayer(
        checkpoint=checkpoint,
        layer_index=layer_index,
        first_channel=first_channel,
        channel_count=channel_count,
        rates=rates,
        intermediate_scales=intermediate,
        gate_suh=gate_up,
        up_suh=gate_up.clone(),
        down_svh=down_svh,
        trellis=trellis,
    )


def _layer_of(name: str) -> int:
    match = _LAYER.search(name)
    return int(match.group(1)) if match else -1


def assemble_trellis_projection_weights(
    layer: TrellisCheckpointLayer, *, device: torch.device | str
) -> TrellisProjectionTierWeights:
    """Group the extent's payload slices into native per-tier storage.

    Pure plumbing: whole-matrix slices are already native-shaped, so
    assembly is tier grouping plus the rotation-table concatenation. No
    symbol is decoded or re-encoded, so equality against the producing
    quantizer's tensors is checkable byte-for-byte.
    """

    device = torch.device(device)
    E = layer.checkpoint.num_experts
    local = layer.channel_count
    bits_by_proj = [[int(layer.rates[e, pi]) for e in range(E)] for pi in range(3)]
    tier_bits = tuple(sorted({b for row in bits_by_proj for b in row}))
    tier_of = {bits: tier for tier, bits in enumerate(tier_bits)}

    def _stack(pi: int) -> dict[int, torch.Tensor]:
        grouped: dict[int, list[torch.Tensor]] = {}
        for expert in range(E):
            bits = bits_by_proj[pi][expert]
            grouped.setdefault(tier_of[bits], []).append(layer.trellis[(expert, pi)])
        return {
            tier: torch.stack(members, dim=0).to(device=device)
            for tier, members in grouped.items()
        }

    # [E, 3*local] boundary values in gate ‖ up ‖ down order, matching the
    # trellis preparation API's intermediate_rotations layout.
    rotations = layer.intermediate_scales.permute(1, 0, 2).reshape(3, E, local)
    intermediate = (
        torch.cat([rotations[0], rotations[1], rotations[2]], dim=1)
        .to(device=device)
        .contiguous()
    )

    return TrellisProjectionTierWeights(
        layer_index=layer.layer_index,
        first_channel=layer.first_channel,
        channel_count=layer.channel_count,
        tier_bits=tier_bits,
        gate_tiers=tuple(tier_of[b] for b in bits_by_proj[0]),
        up_tiers=tuple(tier_of[b] for b in bits_by_proj[1]),
        down_tiers=tuple(tier_of[b] for b in bits_by_proj[2]),
        gate_weights=_stack(0),
        up_weights=_stack(1),
        down_weights=_stack(2),
        gate_suh=layer.gate_suh.to(device=device),
        up_suh=layer.up_suh.to(device=device),
        down_svh=layer.down_svh.to(device=device),
        intermediate_rotations=intermediate,
    )


def prepare_trellis_checkpoint_moe_weights(
    layer: TrellisCheckpointLayer, **_kwargs
) -> None:
    """Fused preparation entry: fails closed toward the three-tier runtime."""

    raise ValueError(
        "b12x_trellis projection-tiered extents prepare through the "
        "projection-mixed MCG K3/K4/K5 runtime (b12x PR #223); this "
        "module provides the checkpoint contract only. Checkpoint "
        "adapters assemble native tier storage with "
        "assemble_trellis_projection_weights and bind it through that "
        "runtime's tier maps."
    )
