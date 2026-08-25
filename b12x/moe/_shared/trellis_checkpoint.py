"""b12x_trellis checkpoint-standard declarations (version 2).

`b12x_trellis` is the in-checkpoint declaration standard for trellis-coded
MoE expert weights (`docs/b12x-trellis-checkpoint-format.md`): a
`quantization_config` block in the model configuration plus four global
metadata tensors, with the routed-expert payload stored as
topology-neutral whole-matrix exllamav3-family trellis tensors in the
model's own shards.

This module is torch-free: configuration parsing, fail-closed validation,
rate-byte arithmetic, and metadata-tensor shape contracts. Tensor I/O and
native tier assembly live with the W4A16 kernel host code
(`b12x/moe/_shared/kernels/w4a16/trellis_checkpoint_reader.py`).
"""

from __future__ import annotations

from dataclasses import dataclass

from .trellis_codebooks import CODEBOOKS, validate_codebook_bits

QUANT_METHOD = "b12x_trellis"
TRELLIS_CHECKPOINT_VERSION = 2

RATE_GRANULARITY_PER_EXPERT_PROJECTION = "per_expert_projection"
SCALE_VECTORS_PER_LAYER = "per_layer"
SCALE_VECTORS_PER_EXPERT = "per_expert"
SCALE_GAINS_NONE = "none"
TRANSFORM_SCALED_HADAMARD = "scaled_hadamard"
TRANSFORM_NONE = "none"

PROJECTIONS = ("gate", "up", "down")

RATE_TENSOR = "b12x_trellis.rate"
INPUT_SCALES_TENSOR = "b12x_trellis.input_scales"
INTERMEDIATE_SCALES_TENSOR = "b12x_trellis.intermediate_scales"
OUTPUT_SCALES_TENSOR = "b12x_trellis.output_scales"
METADATA_TENSORS = (
    RATE_TENSOR,
    INPUT_SCALES_TENSOR,
    INTERMEDIATE_SCALES_TENSOR,
    OUTPUT_SCALES_TENSOR,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_keys(
    mapping: dict, *, required: set[str], optional: set[str], where: str
) -> None:
    _require(isinstance(mapping, dict), f"{where} must be a JSON object")
    keys = set(mapping.keys())
    unknown = keys - required - optional
    _require(not unknown, f"{where} has unknown keys {sorted(unknown)}")
    missing = required - keys
    _require(not missing, f"{where} is missing keys {sorted(missing)}")


def rate_byte(bits: int) -> int:
    """The whole-matrix rate byte for one bitrate: low == high == bits."""

    return (int(bits) << 4) | int(bits)


def rate_byte_bits(code: int) -> int:
    """Decode a whole-matrix rate byte, rejecting split-nibble bytes."""

    low, high = (int(code) >> 4) & 0xF, int(code) & 0xF
    _require(
        low == high,
        f"b12x_trellis rate byte {int(code):#04x} has differing nibbles; "
        "whole-matrix rates require low == high",
    )
    return high


@dataclass(frozen=True)
class TrellisScaleFamily:
    vectors: str
    gains: str


@dataclass(frozen=True)
class TrellisTransform:
    projection_kind: str
    projection_block_size: int | None
    expert_kind: str


@dataclass(frozen=True)
class TrellisCheckpointConfig:
    """Parsed, validated `quantization_config.b12x_trellis` declarations."""

    version: int
    codebook: str
    rate_granularity: str
    input_scales: TrellisScaleFamily
    intermediate_scales: TrellisScaleFamily
    output_scales: TrellisScaleFamily
    transform: TrellisTransform

    @staticmethod
    def from_quantization_config(data: dict) -> "TrellisCheckpointConfig":
        _require(
            isinstance(data, dict) and data.get("quant_method") == QUANT_METHOD,
            f"quantization_config must declare quant_method {QUANT_METHOD!r}",
        )
        block = data.get(QUANT_METHOD)
        _require_keys(
            block if isinstance(block, dict) else {},
            required={"version", "codebook", "rate", "scale", "transform"},
            optional=set(),
            where="b12x_trellis",
        )
        _require(
            block["version"] == TRELLIS_CHECKPOINT_VERSION,
            f"b12x_trellis version must be {TRELLIS_CHECKPOINT_VERSION}, "
            f"got {block['version']!r}",
        )
        codebook = block["codebook"]
        _require(
            codebook in CODEBOOKS,
            f"b12x_trellis codebook must be one of {sorted(CODEBOOKS)}, "
            f"got {codebook!r}",
        )

        rate = block["rate"]
        _require_keys(
            rate,
            required={"granularity"},
            optional=set(),
            where="b12x_trellis rate",
        )
        _require(
            rate["granularity"] == RATE_GRANULARITY_PER_EXPERT_PROJECTION,
            "b12x_trellis rate granularity must be "
            f"{RATE_GRANULARITY_PER_EXPERT_PROJECTION!r}, got "
            f"{rate['granularity']!r}",
        )

        scale = block["scale"]
        _require_keys(
            scale,
            required={"input_scales", "intermediate_scales", "output_scales"},
            optional=set(),
            where="b12x_trellis scale",
        )

        def _family(name: str) -> TrellisScaleFamily:
            family = scale[name]
            _require_keys(
                family,
                required={"vectors", "gains"},
                optional=set(),
                where=f"b12x_trellis scale.{name}",
            )
            _require(
                family["vectors"]
                in (SCALE_VECTORS_PER_LAYER, SCALE_VECTORS_PER_EXPERT),
                f"b12x_trellis scale.{name}.vectors must be "
                f"'{SCALE_VECTORS_PER_LAYER}' or "
                f"'{SCALE_VECTORS_PER_EXPERT}', got {family['vectors']!r}",
            )
            _require(
                family["gains"] == SCALE_GAINS_NONE,
                f"b12x_trellis scale.{name}.gains supports only "
                f"'{SCALE_GAINS_NONE}', got {family['gains']!r}",
            )
            return TrellisScaleFamily(vectors=family["vectors"], gains=family["gains"])

        transform = block["transform"]
        _require_keys(
            transform,
            required={"projection", "expert"},
            optional=set(),
            where="b12x_trellis transform",
        )
        projection = transform["projection"]
        _require_keys(
            projection,
            required={"kind"},
            optional={"block_size"},
            where="b12x_trellis transform.projection",
        )
        _require(
            projection["kind"] == TRANSFORM_SCALED_HADAMARD,
            "b12x_trellis transform.projection.kind supports only "
            f"{TRANSFORM_SCALED_HADAMARD!r}, got {projection['kind']!r}",
        )
        block_size = projection.get("block_size")
        _require(
            isinstance(block_size, int) and block_size > 0 and block_size % 32 == 0,
            "b12x_trellis transform.projection.block_size must be a "
            "positive multiple of 32",
        )
        expert = transform["expert"]
        _require_keys(
            expert,
            required={"kind"},
            optional=set(),
            where="b12x_trellis transform.expert",
        )
        _require(
            expert["kind"] == TRANSFORM_NONE,
            "b12x_trellis transform.expert.kind supports only "
            f"{TRANSFORM_NONE!r}, got {expert['kind']!r}",
        )

        return TrellisCheckpointConfig(
            version=block["version"],
            codebook=codebook,
            rate_granularity=rate["granularity"],
            input_scales=_family("input_scales"),
            intermediate_scales=_family("intermediate_scales"),
            output_scales=_family("output_scales"),
            transform=TrellisTransform(
                projection_kind=projection["kind"],
                projection_block_size=block_size,
                expert_kind=expert["kind"],
            ),
        )

    def validate_bits(self, bits: int) -> None:
        validate_codebook_bits(self.codebook, bits)

    def rate_shape(self, num_layers: int, num_experts: int) -> tuple[int, ...]:
        return (num_layers, num_experts, len(PROJECTIONS))

    def hidden_scales_shape(
        self,
        family: TrellisScaleFamily,
        num_layers: int,
        num_experts: int,
        hidden_size: int,
    ) -> tuple[int, ...]:
        if family.vectors == SCALE_VECTORS_PER_LAYER:
            return (num_layers, hidden_size)
        return (num_layers, num_experts, hidden_size)

    def intermediate_scales_shape(
        self, num_layers: int, num_experts: int, intermediate_size: int
    ) -> tuple[int, ...]:
        if self.intermediate_scales.vectors == SCALE_VECTORS_PER_LAYER:
            return (num_layers, len(PROJECTIONS), intermediate_size)
        return (num_layers, num_experts, len(PROJECTIONS), intermediate_size)

    def validate_extent(
        self, first_channel: int, channel_count: int, intermediate_size: int
    ) -> None:
        """Reject rank extents the transform block width makes illegal."""

        block = self.transform.projection_block_size or 0
        _require(
            channel_count > 0 and first_channel >= 0,
            f"b12x_trellis extent [{first_channel}, "
            f"{first_channel + channel_count}) is empty or negative",
        )
        _require(
            first_channel + channel_count <= intermediate_size,
            f"b12x_trellis extent [{first_channel}, "
            f"{first_channel + channel_count}) exceeds "
            f"{intermediate_size} intermediate channels",
        )
        _require(
            block > 0 and first_channel % block == 0 and channel_count % block == 0,
            f"b12x_trellis extent [{first_channel}, "
            f"{first_channel + channel_count}) must align to the "
            f"{block}-channel transform block",
        )
