import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from bluepixel.adapters import PeftOverlay, SFTOverlay
from bluepixel.artifacts import SourceFileSnapshot
from bluepixel.conversion import TensorSource, convert_component


@pytest.fixture
def tiny_snapshot(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    directory = snapshot / "transformer"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text("{}", encoding="utf-8")
    save_file({"x_embedder.weight": torch.ones(4, 64)}, str(directory / "model.safetensors"))
    # Keep actual file access, conversion and atomic writer; bypass only the
    # fixed 9B shape/config contract so this regression needs a tiny CPU tensor.
    monkeypatch.setattr("bluepixel.conversion.validate_transformer_config", lambda config: None)
    monkeypatch.setattr("bluepixel.conversion.expected_transformer_shapes",
                        lambda config: {"img_in.weight": (4, 64)})
    return snapshot


def replace_file(path):
    replacement = path.with_name(path.name + ".replacement")
    replacement.write_bytes(path.read_bytes() + b" ")
    os.replace(replacement, path)


@pytest.mark.parametrize("changed", ["weight", "config", "index", "sft", "lora_weight", "lora_config",
                                    "tokenizer"])
def test_changed_input_aborts_before_publishing(tiny_snapshot, tmp_path, changed):
    directory = tiny_snapshot / "transformer"
    watched = directory / "model.safetensors"
    options = {"precision": "original", "overwrite": True}
    if changed == "config":
        watched = directory / "config.json"
    elif changed == "index":
        watched = directory / "model.safetensors.index.json"
        watched.write_text(json.dumps({"weight_map": {"x_embedder.weight": "model.safetensors"}}))
    elif changed == "sft":
        watched = tmp_path / "sft.safetensors"
        save_file({"x_embedder.weight": torch.full((4, 64), 3.0)}, str(watched))
        options["sft_checkpoint"] = watched
    elif changed.startswith("lora_"):
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(json.dumps({"r": 1, "lora_alpha": 1}))
        save_file({"x_embedder.lora_A.weight": torch.ones(1, 64),
                   "x_embedder.lora_B.weight": torch.ones(4, 1)},
                  str(adapter / "adapter_model.safetensors"))
        options["peft_directory"] = adapter
        watched = adapter / ("adapter_config.json" if changed == "lora_config"
                             else "adapter_model.safetensors")
    elif changed == "tokenizer":
        text_encoder = tiny_snapshot / "text_encoder"
        text_encoder.mkdir()
        (text_encoder / "config.json").write_text("{}")
        save_file({"model.embed_tokens.weight": torch.ones(4, 64)},
                  str(text_encoder / "model.safetensors"))
        tokenizer = tiny_snapshot / "tokenizer"
        tokenizer.mkdir()
        watched = tokenizer / "tokenizer.json"
        watched.write_text('{"version":"1.0"}')
        options["component"] = "text_encoder"
    output = tmp_path / "output.safetensors"
    manifest = Path(str(output) + ".manifest.json")
    output.write_bytes(b"previous checkpoint")
    manifest.write_bytes(b"previous manifest")

    def progress(event):
        assert event["stage"] == "tensor"
        replace_file(watched)

    with pytest.raises(ValueError, match="Conversion input changed"):
        convert_component(tiny_snapshot, output, progress=progress, **options)
    assert output.read_bytes() == b"previous checkpoint"
    assert manifest.read_bytes() == b"previous manifest"
    assert sorted(path.name for path in tmp_path.glob("output.safetensors*")) == [output.name, manifest.name]


def test_unchanged_inputs_produce_portable_verified_provenance(tiny_snapshot, tmp_path):
    directory = tiny_snapshot / "transformer"
    index = directory / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"x_embedder.weight": "model.safetensors"}}))
    output = tmp_path / "converted.safetensors"
    report = convert_component(tiny_snapshot, output, precision="original")
    assert report["source_files"] == [{"path": "model.safetensors", "sha256":
        hashlib.sha256((directory / "model.safetensors").read_bytes()).hexdigest()}]
    assert report["source_aux_files"] == [
        {"path": "transformer/config.json", "sha256": hashlib.sha256(b"{}").hexdigest()},
        {"path": "transformer/model.safetensors.index.json", "sha256":
         hashlib.sha256(index.read_bytes()).hexdigest()},
    ]
    assert json.loads(Path(str(output) + ".manifest.json").read_text()) == report
    with safe_open(str(output), framework="pt", device="cpu") as reader:
        assert torch.equal(reader.get_tensor("img_in.weight"), torch.ones(4, 64))
    assert str(tmp_path) not in json.dumps(report)


def test_change_during_source_hash_aborts_before_publishing(tiny_snapshot, tmp_path, monkeypatch):
    from bluepixel import artifacts
    original_hash = artifacts.sha256_file
    source = tiny_snapshot / "transformer" / "model.safetensors"
    output = tmp_path / "not-published.safetensors"

    def hash_then_replace(path, *args, **kwargs):
        value = original_hash(path, *args, **kwargs)
        if Path(path) == source:
            replace_file(source)
        return value

    monkeypatch.setattr(artifacts, "sha256_file", hash_then_replace)
    with pytest.raises(ValueError, match="Conversion input changed"):
        convert_component(tiny_snapshot, output, precision="original")
    assert not output.exists()
    assert not Path(str(output) + ".manifest.json").exists()
    assert not list(tmp_path.glob(output.name + "*"))


@pytest.mark.parametrize("indexed", [False, True])
def test_replacement_while_opening_source_is_not_adopted_as_initial_snapshot(
        tiny_snapshot, monkeypatch, indexed):
    from bluepixel import conversion
    directory = tiny_snapshot / "transformer"
    if indexed:
        (directory / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"x_embedder.weight": "model.safetensors"}}))

    @contextmanager
    def open_then_replace(path, *args, **kwargs):
        with safe_open(path, *args, **kwargs) as reader:
            replace_file(Path(path))
            yield reader

    monkeypatch.setattr(conversion, "safe_open", open_then_replace)
    with pytest.raises(ValueError, match="Conversion input changed"):
        with TensorSource(directory) as source:
            source.get("x_embedder.weight")


def test_deleted_source_reports_change(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"source")
    snapshot = SourceFileSnapshot(source)
    source.unlink()
    with pytest.raises(ValueError, match="changed or became unavailable"):
        snapshot.sha256()


@pytest.mark.parametrize("overlay_type", ["sft", "peft"])
def test_adapter_replacement_during_open_is_not_adopted_as_initial_snapshot(
        tmp_path, monkeypatch, overlay_type):
    from bluepixel import adapters
    if overlay_type == "sft":
        weights = tmp_path / "sft.safetensors"
        save_file({"block.weight": torch.ones(4, 64)}, str(weights))
    else:
        weights = tmp_path / "adapter_model.safetensors"
        (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 1, "lora_alpha": 1}))
        save_file({"block.lora_A.weight": torch.ones(1, 64),
                   "block.lora_B.weight": torch.ones(4, 1)}, str(weights))

    def open_then_replace(path, *args, **kwargs):
        reader = safe_open(path, *args, **kwargs)
        replace_file(Path(path))
        return reader

    monkeypatch.setattr(adapters, "safe_open", open_then_replace)
    with pytest.raises(ValueError, match="Conversion input changed"):
        if overlay_type == "sft":
            SFTOverlay(weights, {"block.weight"})
        else:
            PeftOverlay(tmp_path, {"block.weight"})
