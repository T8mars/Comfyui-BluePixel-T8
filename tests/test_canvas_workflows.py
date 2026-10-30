import importlib.util
import json
from pathlib import Path
from uuid import UUID

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_eight_canvas_workflows_are_shipped():
    paths = sorted((ROOT / "workflows").glob("*.json"))
    assert len(paths) == 8 and not any(path.name.endswith(".api.json") for path in paths)
    graphs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    assert len({UUID(graph["id"]) for graph in graphs}) == 8
    for graph in graphs:
        assert graph["version"] == 0.4 and graph["nodes"] and graph["links"]
        assert graph["extra"]["ds"]["scale"] > 0


def test_preview_widget_defaults_match_its_connected_size_and_text_layers():
    graph = json.loads((ROOT / "workflows/02_edit_1k_preview.json").read_text(encoding="utf-8"))
    nodes = {node["type"]: node for node in graph["nodes"]}
    assert nodes["BluePixelEmptyLatent"]["widgets_values"] == [1024, 1024, 1]
    assert nodes["BluePixelTextEncode"]["widgets_values"][-1] == "9,18,27"
    assert [socket["widget"]["name"] for socket in nodes["BluePixelEmptyLatent"]["inputs"]] == ["width", "height"]
    assert nodes["SaveImage"]["outputs"][0]["type"] == "IMAGE"


@pytest.mark.parametrize("failure", [None, "backlink", "target_slot", "type"])
def test_canvas_verifier_detects_broken_links(failure):
    builder, verifier = load_tool("build_workflows"), load_tool("verify_workflows")
    graph = builder.to_ui({
        "1": {"class_type": "LoadImage", "inputs": {"image": "input.png"}},
        "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "Test"}},
    })
    schemas = {
        "LoadImage": {"input": {"required": {"image": (["input.png"], {})}}, "output": ["IMAGE", "MASK"]},
        "SaveImage": {"input": {"required": {"images": ("IMAGE",), "filename_prefix": ("STRING", {})}}, "output": ["IMAGE"]},
    }
    if failure == "backlink":
        graph["nodes"][0]["outputs"][0]["links"] = []
    elif failure == "target_slot":
        graph["links"][0][4] = 1
    elif failure == "type":
        graph["links"][0][5] = "MASK"
    if failure:
        with pytest.raises(ValueError):
            verifier.validate_canvas(graph, schemas)
    else:
        assert verifier.validate_canvas(graph, schemas) == 2
