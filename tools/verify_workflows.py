"""Validate shipped canvas workflows and their real ComfyUI node connections."""

import argparse
import asyncio
import json
from pathlib import Path
import sys


def validate_canvas(graph, schemas):
    if graph.get("version") != 0.4 or not isinstance(graph.get("nodes"), list) or not graph["nodes"]:
        raise ValueError("Expected a non-empty canvas workflow with nodes, links and version 0.4")
    nodes = {node["id"]: node for node in graph["nodes"]}
    if len(nodes) != len(graph["nodes"]):
        raise ValueError("Duplicate canvas node IDs")
    links = {link[0]: link for link in graph["links"]}
    if len(links) != len(graph["links"]):
        raise ValueError("Duplicate canvas link IDs")
    for node in nodes.values():
        schema = schemas[node["type"]]
        inputs = {**schema["input"].get("required", {}), **schema["input"].get("optional", {})}
        if len(node["outputs"]) != len(schema["output"]):
            raise ValueError(f"{node['id']}: output count differs from the node definition")
        for slot, output in enumerate(node["outputs"]):
            if output["type"] != schema["output"][slot]:
                raise ValueError(f"{node['id']}: incorrect output type")
            expected = sorted(link[0] for link in links.values() if link[1:3] == [node["id"], slot])
            if sorted(output.get("links") or []) != expected:
                raise ValueError(f"{node['id']}: output links disagree with the canvas links")
        sockets = {value["name"]: value for value in node["inputs"]}
        if set(sockets) - inputs.keys():
            raise ValueError(f"{node['id']}: unknown input sockets")
        widget_index = 0
        for name, specification in inputs.items():
            kind = specification[0]
            options = specification[1] if len(specification) > 1 else {}
            if isinstance(kind, list) or kind in {"INT", "FLOAT", "STRING", "BOOLEAN"}:
                if widget_index >= len(node["widgets_values"]):
                    raise ValueError(f"{node['id']}: missing widget value for {name}")
                widget_index += 1 + bool(options.get("control_after_generate"))
            elif name in schema["input"].get("required", {}) and name not in sockets:
                raise ValueError(f"{node['id']}: missing required socket {name}")
        for slot, socket in enumerate(node["inputs"]):
            link = links.get(socket["link"])
            if link is None or link[3:5] != [node["id"], slot]:
                raise ValueError(f"{node['id']}: input link targets the wrong socket")
            source = nodes[link[1]]
            actual = schemas[source["type"]]["output"][link[2]]
            expected = inputs[socket["name"]][0]
            if actual != expected or link[5] != actual or socket["type"] != actual:
                raise ValueError(f"{node['id']}:{socket['name']}: incompatible canvas connection")
    return len(nodes)


async def verify(comfyui):
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    comfy.cli_args.args.use_pytorch_cross_attention = True
    comfy.cli_args.args.disable_xformers = True
    import nodes
    await nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)
    root = Path(__file__).resolve().parents[1]
    await nodes.load_custom_node(str(root))
    shipped = sorted((root / "workflows").glob("*.json"))
    if not shipped or any(path.name.endswith(".api.json") for path in shipped):
        raise ValueError("Only canvas workflow files should be shipped")
    graphs = [json.loads(path.read_text(encoding="utf-8")) for path in shipped]
    names = {node["type"] for graph in graphs for node in graph["nodes"]}
    schemas = {name: {"input": nodes.NODE_CLASS_MAPPINGS[name].INPUT_TYPES(),
                      "output": nodes.NODE_CLASS_MAPPINGS[name].RETURN_TYPES} for name in names}
    checked = sum(validate_canvas(graph, schemas) for graph in graphs)
    print(json.dumps({"canvas_workflows": len(shipped), "nodes_type_checked": checked,
                      "models_loaded": False, "sampled": False}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.comfyui.resolve(strict=True)))
    asyncio.run(verify(args.comfyui))


if __name__ == "__main__":
    main()
