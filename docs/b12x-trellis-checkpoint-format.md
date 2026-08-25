# b12x_trellis checkpoint format (version 2)

`b12x_trellis` is the in-checkpoint declaration standard for trellis-coded
MoE expert weights. Unlike a sidecar container, a `b12x_trellis` checkpoint
is a normal Hugging-Face-style model directory: the routed-expert payload
stays in the model's own safetensors shards as topology-neutral
whole-matrix tensors, and the quantization contract lives in two places —
a `quantization_config` block in `config.json` and four global metadata
tensors. One stored artifact serves every tensor-parallel degree: a rank
slices the expert tensors and the intermediate-axis scales at load, and no
trellis symbol is ever decoded or re-encoded on the way in.

Schema id: `quant_method = "b12x_trellis"`, `version = 2`. The in-repo
implementation is `b12x/moe/_shared/trellis_checkpoint.py` (declarations)
and `b12x/moe/_shared/kernels/w4a16/trellis_checkpoint_reader.py`
(reading, slicing, and native tier assembly).

## Configuration block

```json
"quantization_config": {
  "quant_method": "b12x_trellis",
  "b12x_trellis": {
    "version": 2,
    "codebook": "mcg",
    "rate": { "granularity": "per_expert_projection" },
    "scale": {
      "input_scales":        { "vectors": "per_layer",  "gains": "none" },
      "intermediate_scales": { "vectors": "per_expert", "gains": "none" },
      "output_scales":       { "vectors": "per_layer",  "gains": "none" }
    },
    "transform": {
      "projection": { "kind": "scaled_hadamard", "block_size": 128 },
      "expert": { "kind": "none" }
    }
  }
}
```

Validation is fail-closed: unknown keys, unknown enum values, and every
cross-field inconsistency below are rejected before any tensor is read.

| field | contents |
|---|---|
| `version` | `2` |
| `codebook` | `"mcg"` — the registry codebook (`b12x/moe/_shared/trellis_codebooks.py`); the MCG multiplier is the registry constant, not restated per tensor |
| `rate.granularity` | `"per_expert_projection"` — one whole-matrix bitrate per (expert, projection) |
| `scale.<family>.vectors` | `"per_layer"` or `"per_expert"` — the granularity of that family's scale vectors |
| `scale.<family>.gains` | `"none"` — reserved for scalar gain factors layered on the vectors |
| `transform.projection` | `{"kind": "scaled_hadamard", "block_size": B}` — the activation-boundary transform; `B` is the Hadamard block width in channels and the resharding quantum |
| `transform.expert` | `{"kind": "none"}` — no per-expert transform |

## Metadata tensors

Four global tensors carry the whole model's rate and scale metadata. With
`L` MoE layers, `E` routed experts per layer, hidden width `H`, and
expert intermediate width `I`, and the projection axis always ordered
`[gate, up, down]`:

| tensor | dtype / shape | contents |
|---|---|---|
| `b12x_trellis.rate` | u8 `[L, E, 3]` | one rate byte per (layer, expert, projection); a rate byte is `(low_bits << 4) \| high_bits` with `low == high` for whole-matrix rates — `0x33`, `0x44`, `0x55` |
| `b12x_trellis.input_scales` | fp16 `[L, H]` (per_layer) or `[L, E, H]` (per_expert) | hidden-axis input-side scale vectors for gate and up |
| `b12x_trellis.intermediate_scales` | fp16 `[L, E, 3, I]` (per_expert) or `[L, 3, I]` (per_layer) | intermediate-axis boundary vectors: gate output-side, up output-side, down input-side |
| `b12x_trellis.output_scales` | fp16 `[L, H]` (per_layer) or `[L, E, H]` (per_expert) | hidden-axis output-side scale vectors for down |

The `[L, …]` axis indexes the model's MoE layers in ascending layer order;
the mapping to absolute layer indices comes from the model configuration
(`first_k_dense_replace` .. `num_hidden_layers - 1` for contiguous-MoE
models). Every rate byte must appear in the codebook's defined bitrate
range, and the reader rejects rate bytes whose low and high nibbles
disagree.

`vectors: "per_layer"` on a hidden-axis family declares that gate and up
(input side) or down (output side) share one vector per layer across all
experts — the broadcast-rotation serving layout. Checkpoints whose
per-expert hidden-axis vectors genuinely differ declare
`vectors: "per_expert"` and carry the `[L, E, H]` form.

## Payload contract

Routed-expert weights are ordinary exllamav3-family trellis tensors in
the model's own shards, whole-matrix and topology-neutral:

```
<prefix>.experts.<E>.<proj>.trellis    int16 [in/16, out/16, 16*K]
```

with `proj` in `gate_proj | up_proj | down_proj`, `in/out` the matrix's
input/output widths (`[H/16, I/16]` for gate/up, `[I/16, H/16]` for
down), and `K` the matrix's bitrate — which must equal the matrix's entry
in `b12x_trellis.rate`. Per-tensor scale/multiplier tensors from the
producing quantizer (`suh`/`svh`/`mcg`) may remain in the shards for
provenance; the metadata tensors are the serving contract, and readers do
not consult the per-tensor copies.

## Tensor-parallel slicing

The expert intermediate axis is expert-private, so a rank owns a
contiguous channel range `[first, first + count)` of `I` and loads:

- gate/up `trellis[:, first/16 : (first+count)/16, :]`
- down `trellis[first/16 : (first+count)/16, :, :]`
- `intermediate_scales[..., first : first + count]`

The range must align to `transform.projection.block_size` channels.
Hidden-axis tensors and `rate` are loaded whole. No decode, no
re-encode, no resharding artifact: the same checkpoint serves TP1
through TP-`I/block_size`.

## Preparation

`read_trellis_checkpoint_layer` loads one rank extent of one layer and
`assemble_trellis_projection_weights` restores native per-tier storage:
per projection and bitrate, int16 `[count, H/16, 2*slots, 16*K]` FC1 and
`[count, 2*slots, H/16, 16*K]` FC2 tensors (32-channel slots on the local
intermediate axis), plus per-projection tier-ID vectors in ascending
bitrate order with tier-local experts in ascending global-ID order —
exactly the inputs of the projection-mixed MCG K3/K4/K5 runtime's
`build_projection_tiered_maps` (PR #223). On trees without that runtime,
preparation fails closed with a status message.

## Support status

- **Declared, adapter-prepared**: `per_expert_projection` rates over MCG
  bitrate sets within {K3, K4, K5}, `scaled_hadamard` projection
  transform, `gains: none`. The declaration, validation, slicing, and
  native tier assembly are complete; fused execution binds through the
  three-tier runtime in
  `b12x/moe/_shared/kernels/w4a16/mixed_trellis.py` (PR #223). Where
  that runtime is absent, projection-tiered preparation is unsupported
  and fails closed.
- Other codebooks, bitrate sets, transform kinds, and gain declarations
  fail closed at validation with a status message naming the unsupported
  field.
- The BTX container (`docs/btx-checkpoint-format.md`) remains B12X's
  prepared-weight representation in the serving API; `b12x_trellis` is
  the distribution-side checkpoint standard that feeds it.
