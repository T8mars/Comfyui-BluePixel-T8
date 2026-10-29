import json
import struct
import torch
import pytest
from safetensors import safe_open
from safetensors.torch import save_file

from bluepixel.artifacts import StreamingSafeWriter, read_header
from bluepixel.conversion import TensorSource, quantize_convrot, transformer_mapping


def test_streaming_file_is_readable_and_lossless(tmp_path):
    tensors = {"fp32": torch.randn(13, 17), "bf16": torch.randn(4, 7).bfloat16(),
               "counter": torch.tensor(42), "codes": torch.tensor([-128, 0, 127], dtype=torch.int8)}
    path = tmp_path / "model.safetensors"
    with StreamingSafeWriter(path, {"source": "fixture"}) as writer:
        for key, tensor in tensors.items():
            writer.add(key, tensor)
    with safe_open(str(path), framework="pt") as source:
        assert source.metadata() == {"source": "fixture"}
        for key, expected in tensors.items():
            actual = source.get_tensor(key)
            assert actual.dtype == expected.dtype
            assert actual.shape == expected.shape
            assert torch.equal(actual, expected)
    assert not list(tmp_path.glob("*.payload"))
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(FileExistsError):
        with StreamingSafeWriter(path):
            pass


def test_failed_write_is_never_published(tmp_path):
    path = tmp_path / "bad.safetensors"
    with pytest.raises(ValueError):
        with StreamingSafeWriter(path) as writer:
            writer.add("w", torch.ones(3))
            writer.add("w", torch.ones(3))
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_tensor_source_checks_index_and_prevents_path_escape(tmp_path):
    save_file({"first": torch.ones(3)}, str(tmp_path / "a.safetensors"))
    save_file({"second": torch.zeros(4)}, str(tmp_path / "b.safetensors"))
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"first": "a.safetensors", "second": "b.safetensors"}}))
    with TensorSource(tmp_path) as source:
        assert torch.equal(source.get("first"), torch.ones(3))
        assert torch.equal(source.get("second"), torch.zeros(4))
    index.write_text(json.dumps({"weight_map": {"first": "../escape.safetensors"}}))
    with pytest.raises(ValueError, match="escapes"):
        TensorSource(tmp_path)


def test_qkv_mapping_enforces_order_and_no_silent_keys():
    mapping = transformer_mapping([f"transformer_blocks.0.attn.to_{letter}.weight" for letter in "vkq"])
    assert [item[1].split(".")[-2] for item in mapping["double_blocks.0.img_attn.qkv.weight"]] == ["to_q", "to_k", "to_v"]
    with pytest.raises(ValueError, match="Incomplete"):
        transformer_mapping(["transformer_blocks.0.attn.to_q.weight"])
    with pytest.raises(ValueError, match="Unmapped"):
        transformer_mapping(["unrecognized.weight"])


@pytest.mark.parametrize("width,group", [(256, 256), (128, 64)])
def test_native_convrot_round_trip_and_linear(tmp_path, width, group):
    from comfy_kitchen.tensor import TensorWiseINT8Layout, QuantizedTensor
    torch.manual_seed(117)
    weight = torch.randn(64, width)
    codes, scale = quantize_convrot(weight, group, row_chunk=13)
    params = TensorWiseINT8Layout.Params(scale=scale, orig_dtype=torch.float32,
        orig_shape=tuple(weight.shape), convrot=True, convrot_groupsize=group)
    reconstructed = TensorWiseINT8Layout.dequantize(codes, params)
    relative_error = torch.linalg.vector_norm(reconstructed - weight) / torch.linalg.vector_norm(weight)
    assert codes.dtype == torch.int8
    assert relative_error < 0.02
    # Prove this is a rotation, not an ordinary INT8 cast with a false metadata flag.
    assert not torch.allclose(reconstructed, codes.float() * scale, atol=0.1)
    quantized = QuantizedTensor(codes, "TensorWiseINT8Layout", params)
    x = torch.randn(4, width)
    y = torch.nn.functional.linear(x, quantized)
    expected = torch.nn.functional.linear(x, weight)
    assert (torch.linalg.vector_norm(y - expected) / torch.linalg.vector_norm(expected)) < 0.04
    path = tmp_path / "int8.safetensors"
    with StreamingSafeWriter(path) as writer:
        writer.add("linear.weight", codes)
        writer.add("linear.weight_scale", scale)
    assert read_header(path)["linear.weight"]["dtype"] == "I8"
