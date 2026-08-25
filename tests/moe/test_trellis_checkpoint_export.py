"""EXL3-to-b12x_trellis exporter: elections, losslessness, fail-closed.

The exporter under test is ``scripts/export_exl3_to_b12x_trellis.py``.
Sources are synthesized as whole-matrix exllamav3-family checkpoints with
per-expert or layer-shared hidden-axis vectors. All tests are host-side.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest
import torch
from safetensors.torch import save_file

from b12x.moe._shared.kernels.w4a16.trellis_checkpoint_reader import (
    read_trellis_checkpoint,
    read_trellis_checkpoint_layer,
)
from b12x.moe._shared.trellis_codebooks import MCG_MULTIPLIER

_REPO = pathlib.Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "export_exl3_to_b12x_trellis",
    _REPO / "scripts" / "export_exl3_to_b12x_trellis.py",
)
exporter = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("export_exl3_to_b12x_trellis", exporter)
_SPEC.loader.exec_module(exporter)

_HIDDEN = 64
_INTERMEDIATE = 512
_EXPERTS = 3
_MOE_LAYERS = (1, 2)
_TRIPLES = ((3, 4, 5), (3, 3, 3), (4, 4, 3))


def _mcg() -> torch.Tensor:
    return torch.tensor(MCG_MULTIPLIER, dtype=torch.int64).to(torch.int32)


def _write_source(root: pathlib.Path, *, layout: str = "per_expert") -> None:
    generator = torch.Generator().manual_seed(20260824)
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": _HIDDEN,
                "moe_intermediate_size": _INTERMEDIATE,
                "n_routed_experts": _EXPERTS,
                "first_k_dense_replace": _MOE_LAYERS[0],
                "num_hidden_layers": _MOE_LAYERS[-1] + 1,
            }
        )
    )

    def _values(width: int) -> torch.Tensor:
        raw = torch.rand((width,), generator=generator, dtype=torch.float32)
        return (0.5 + raw).to(torch.float16)

    for layer in _MOE_LAYERS:
        prefix = f"model.layers.{layer}.mlp.experts"
        tensors: dict[str, torch.Tensor] = {}
        if layout == "shared":
            tensors[f"{prefix}.shared_vectors.gate_up_suh"] = _values(_HIDDEN)
            tensors[f"{prefix}.shared_vectors.down_svh"] = _values(_HIDDEN)
        for expert, triple in enumerate(_TRIPLES):
            gate_up_suh = _values(_HIDDEN)
            for proj, bits in zip(
                ("gate_proj", "up_proj", "down_proj"), triple, strict=True
            ):
                fc1 = proj != "down_proj"
                base = f"{prefix}.{expert}.{proj}"
                shape = (
                    (_HIDDEN // 16, _INTERMEDIATE // 16, 16 * bits)
                    if fc1
                    else (_INTERMEDIATE // 16, _HIDDEN // 16, 16 * bits)
                )
                tensors[f"{base}.trellis"] = torch.randint(
                    -(1 << 15),
                    1 << 15,
                    shape,
                    dtype=torch.int16,
                    generator=generator,
                )
                tensors[f"{base}.mcg"] = _mcg()
                if fc1:
                    tensors[f"{base}.svh"] = _values(_INTERMEDIATE)
                    if layout == "per_expert":
                        tensors[f"{base}.suh"] = gate_up_suh.clone()
                else:
                    tensors[f"{base}.suh"] = _values(_INTERMEDIATE)
                    if layout == "per_expert":
                        tensors[f"{base}.svh"] = _values(_HIDDEN)
        save_file(tensors, str(root / f"experts-layer-{layer:03d}.safetensors"))


def _run(source: pathlib.Path, output: pathlib.Path, *extra: str) -> dict:
    report_path = output / "report.json"
    code = exporter.main(
        [
            "--model-dir",
            str(source),
            "--output",
            str(output),
            "--verify",
            "full",
            "--report",
            str(report_path),
            *extra,
        ]
    )
    assert code == 0
    return json.loads(report_path.read_text())


def _as_v2_dir(source: pathlib.Path, output: pathlib.Path) -> pathlib.Path:
    """Assemble a readable v2 checkpoint view from source + exporter output."""

    view = output / "view"
    view.mkdir()
    for shard in source.glob("*.safetensors"):
        (view / shard.name).symlink_to(shard)
    (view / "b12x-trellis.safetensors").symlink_to(output / "b12x-trellis.safetensors")
    (view / "config.json").write_text((output / "config.patched.json").read_text())
    return view


@pytest.mark.parametrize("layout", ("per_expert", "shared"))
def test_export_elects_granularities_and_verifies(tmp_path, layout) -> None:
    source, output = tmp_path / "src", tmp_path / "out"
    _write_source(source, layout=layout)
    report = _run(source, output)

    assert report["verification"]["result"] == "byte-identical"
    assert report["elections"]["intermediate_scales"] == "per_expert"
    expected_hidden = "per_layer" if layout == "shared" else "per_expert"
    assert report["elections"]["input_scales"] == expected_hidden
    assert report["elections"]["output_scales"] == expected_hidden
    assert report["gate_up_divergent_assignments"] == 2  # expert 0, both layers

    checkpoint = read_trellis_checkpoint(_as_v2_dir(source, output))
    layer = read_trellis_checkpoint_layer(
        checkpoint, 2, first_channel=128, channel_count=256
    )
    for expert, triple in enumerate(_TRIPLES):
        for pi, bits in enumerate(triple):
            assert int(layer.rates[expert, pi]) == bits


def test_export_rejects_divergent_gate_up_input_vectors(tmp_path) -> None:
    source, output = tmp_path / "src", tmp_path / "out"
    _write_source(source, layout="per_expert")
    from safetensors import safe_open

    shard = source / "experts-layer-001.safetensors"
    with safe_open(str(shard), framework="pt") as handle:
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}
    name = "model.layers.1.mlp.experts.0.up_proj.suh"
    tensors[name] = tensors[name] + 1.0
    save_file(tensors, str(shard))
    with pytest.raises(SystemExit, match="gate and up"):
        _run(source, output)


def test_export_rejects_rank_sliced_payloads(tmp_path) -> None:
    source, output = tmp_path / "src", tmp_path / "out"
    _write_source(source, layout="per_expert")
    save_file(
        {
            "model.layers.1.mlp.experts.0.gate_proj.rank0.trellis": (
                torch.zeros((4, 8, 48), dtype=torch.int16)
            )
        },
        str(source / "sliced.safetensors"),
    )
    with pytest.raises(SystemExit, match="rank-sliced"):
        _run(source, output)


def test_export_rejects_foreign_mcg_multiplier(tmp_path) -> None:
    source, output = tmp_path / "src", tmp_path / "out"
    _write_source(source, layout="per_expert")
    from safetensors import safe_open

    shard = source / "experts-layer-001.safetensors"
    with safe_open(str(shard), framework="pt") as handle:
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}
    tensors["model.layers.1.mlp.experts.0.gate_proj.mcg"] = torch.tensor(
        1234567, dtype=torch.int32
    )
    save_file(tensors, str(shard))
    with pytest.raises(SystemExit, match="MCG multiplier"):
        _run(source, output)


def test_export_cross_checks_sidecar_bit_maps(tmp_path) -> None:
    source, output = tmp_path / "src", tmp_path / "out"
    _write_source(source, layout="per_expert")
    sidecar = {
        "bit_map": {"model.layers.1.mlp.experts.0.gate_proj": 5}
    }  # tensor carries K3
    (source / "experts-layer-001.json").write_text(json.dumps(sidecar))
    with pytest.raises(SystemExit, match="sidecar"):
        _run(source, output)

    sidecar["bit_map"]["model.layers.1.mlp.experts.0.gate_proj"] = 3
    (source / "experts-layer-001.json").write_text(json.dumps(sidecar))
    report = _run(source, output)
    assert report["checks"]["sidecar_bit_map_layers"] == 1


def test_export_rejects_partial_checkpoints(tmp_path) -> None:
    source, output = tmp_path / "src", tmp_path / "out"
    _write_source(source, layout="per_expert")
    (source / "experts-layer-002.safetensors").unlink()
    with pytest.raises(SystemExit, match="partial checkpoint"):
        _run(source, output)
