"""Check shipped API graphs against real ComfyUI node schemas without sampling."""

import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
import sys


async def verify(comfyui):
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    comfy.cli_args.args.use_pytorch_cross_attention = True
    comfy.cli_args.args.disable_xformers = True
    import nodes
    await nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)
    root = Path(__file__).resolve().parents[1]
    await nodes.load_custom_node(str(root))
    shipped = sorted((root / "workflows").glob("*.api.json"))
    checked = 0
    for path in shipped:
        graph = json.loads(path.read_text(encoding="utf-8"))
        for node_id, data in graph.items():
            cls = nodes.NODE_CLASS_MAPPINGS[data["class_type"]]
            schema = cls.INPUT_TYPES()
            required = schema.get("required", {})
            optional = schema.get("optional", {})
            inputs = data["inputs"]
            missing = set(required) - set(inputs)
            unknown = set(inputs) - set(required) - set(optional)
            if missing or unknown:
                raise ValueError(f"{path.name}:{node_id}: missing={missing}, unknown={unknown}")
            for name, value in inputs.items():
                if not isinstance(value, list):
                    continue
                source, slot = value
                source_cls = nodes.NODE_CLASS_MAPPINGS[graph[source]["class_type"]]
                output_type = source_cls.RETURN_TYPES[slot]
                expected = (required | optional)[name][0]
                if output_type != expected:
                    raise ValueError(f"{path.name}:{node_id}:{name}: {output_type} != {expected}")
            checked += 1
    print(json.dumps({"api_workflows": len(shipped), "nodes_type_checked": checked,
                      "models_loaded": False, "sampled": False}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.comfyui.resolve(strict=True)))
    asyncio.run(verify(args.comfyui))


if __name__ == "__main__":
    main()
