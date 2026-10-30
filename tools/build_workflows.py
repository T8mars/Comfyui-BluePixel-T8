"""Generate editable ComfyUI canvas workflows."""

from copy import deepcopy
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5


def workflow():
    return {
        "1": {"class_type": "BluePixelDiffusionLoader", "inputs": {"model_name": "bluepixel_turbo_int8_convrot.safetensors", "dtype": "default", "disable_dynamic": False}},
        "2": {"class_type": "BluePixelCLIPLoader", "inputs": {"model_name": "bluepixel_qwen3_8b_original.safetensors", "device": "default"}},
        "3": {"class_type": "BluePixelVAELoader", "inputs": {"model_name": "bluepixel_dcae_f64.safetensors", "tiling": True}},
        "4": {"class_type": "LoadImage", "inputs": {"image": "bluepixel_test_input.jpg"}},
        "5": {"class_type": "BluePixelPrepareImage", "inputs": {"image": ["4", 0], "target_pixels": 16777216, "output_width": 4096, "output_height": 4096}},
        "6": {"class_type": "BluePixelTextEncode", "inputs": {"clip": ["2", 0], "instruction": "Change the gray sweater to burgundy red.", "max_sequence_length": 512}},
        "7": {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["3", 0]}},
        "8": {"class_type": "BluePixelEditConditioning", "inputs": {"positive": ["6", 0], "negative": ["6", 1], "reference": ["7", 0]}},
        "9": {"class_type": "BluePixelEmptyLatent", "inputs": {"width": ["5", 1], "height": ["5", 2], "batch_size": 1}},
        "10": {"class_type": "BluePixelScheduler", "inputs": {"latent": ["9", 0], "steps": 4, "mu_mode": "empirical", "shift": 0.0}},
        "11": {"class_type": "BluePixelNoise", "inputs": {"seed": 42, "mode": "upstream_cuda_bf16"}},
        "12": {"class_type": "BluePixelCFGGuider", "inputs": {"model": ["1", 0], "positive": ["8", 0], "negative": ["8", 1], "guidance": 3.0}},
        "13": {"class_type": "BluePixelEulerSampler", "inputs": {}},
        "14": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["11", 0], "guider": ["12", 0], "sampler": ["13", 0], "sigmas": ["10", 0], "latent_image": ["9", 0]}},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["3", 0]}},
        "16": {"class_type": "SaveImage", "inputs": {"images": ["15", 0], "filename_prefix": "BluePixel/INT8_ConvRot"}},
    }


OUTPUTS = {
    "BluePixelDiffusionLoader": [("MODEL", "MODEL")], "BluePixelCLIPLoader": [("CLIP", "CLIP")],
    "BluePixelVAELoader": [("VAE", "VAE")], "LoadImage": [("IMAGE", "IMAGE"), ("MASK", "MASK")],
    "BluePixelPrepareImage": [("reference_image", "IMAGE"), ("output_width", "INT"), ("output_height", "INT"), ("metadata", "STRING")],
    "BluePixelTextEncode": [("positive", "CONDITIONING"), ("empty_instruction", "CONDITIONING")],
    "VAEEncode": [("LATENT", "LATENT")], "BluePixelEditConditioning": [("positive", "CONDITIONING"), ("negative", "CONDITIONING")],
    "BluePixelEmptyLatent": [("LATENT", "LATENT")], "BluePixelScheduler": [("sigmas", "SIGMAS"), ("schedule_metadata", "STRING")],
    "BluePixelNoise": [("NOISE", "NOISE")], "BluePixelCFGGuider": [("GUIDER", "GUIDER")],
    "KSamplerSelect": [("SAMPLER", "SAMPLER")], "SamplerCustomAdvanced": [("output", "LATENT"), ("denoised_output", "LATENT")],
    "BluePixelEulerSampler": [("SAMPLER", "SAMPLER")],
    "BluePixelSeedListNoise": [("NOISE", "NOISE")],
    "VAEDecode": [("IMAGE", "IMAGE")], "SaveImage": [("images", "IMAGE")],
}


WIDGETS = {
    "BluePixelDiffusionLoader": [("model_name", ""), ("dtype", "default"), ("disable_dynamic", False)],
    "BluePixelCLIPLoader": [("model_name", ""), ("device", "default")],
    "BluePixelVAELoader": [("model_name", ""), ("tiling", True)],
    "LoadImage": [("image", "")],
    "BluePixelPrepareImage": [("target_pixels", 16777216), ("output_width", 0), ("output_height", 0)],
    "BluePixelTextEncode": [("instruction", ""), ("max_sequence_length", 512), ("hidden_layers", "9,18,27")],
    "BluePixelEmptyLatent": [("width", 1024), ("height", 1024), ("batch_size", 1)],
    "BluePixelScheduler": [("steps", 4), ("mu_mode", "empirical"), ("shift", 0.0)],
    "BluePixelNoise": [("seed", 42), ("mode", "upstream_cuda_bf16")],
    "BluePixelCFGGuider": [("guidance", 3.0)],
    "BluePixelSeedListNoise": [("seeds", "42,43"), ("mode", "upstream_cuda_bf16")],
    "SaveImage": [("filename_prefix", "BluePixel")],
}


def to_ui(graph, workflow_name="bluepixel"):
    nodes, lookup, links = [], {}, []
    for order, (node_id, specification) in enumerate(graph.items()):
        outputs = [{"name": name, "type": kind, "links": [], "slot_index": slot}
                   for slot, (name, kind) in enumerate(OUTPUTS[specification["class_type"]])]
        node = {"id": int(node_id), "type": specification["class_type"],
                "pos": [(order % 4) * 440, (order // 4) * 340], "size": [390, 280],
                "flags": {}, "order": order, "mode": 0, "inputs": [], "outputs": outputs,
                "properties": {"Node name for S&R": specification["class_type"]}, "widgets_values": []}
        if specification["class_type"].startswith("BluePixel"):
            node["properties"]["cnr_id"] = "bluepixel-t8"
        for widget_name, default in WIDGETS.get(specification["class_type"], []):
            value = specification["inputs"].get(widget_name, default)
            if isinstance(value, list):
                source = graph[value[0]]["inputs"]
                value = source.get("output_" + widget_name, default) or default
            node["widgets_values"].append(value)
            if widget_name == "seed":
                node["widgets_values"].append("fixed")
        if specification["class_type"] == "LoadImage":
            node["widgets_values"].append("image")
        nodes.append(node)
        lookup[node_id] = node
    for node_id, specification in graph.items():
        node = lookup[node_id]
        for name, value in specification["inputs"].items():
            if not isinstance(value, list):
                continue
            origin, slot = value
            output = lookup[origin]["outputs"][slot]
            link_id = len(links) + 1
            links.append([link_id, int(origin), slot, int(node_id), len(node["inputs"]), output["type"]])
            node["inputs"].append({"name": name, "type": output["type"], "link": link_id})
            if specification["class_type"] == "BluePixelEmptyLatent" and name in {"width", "height"}:
                node["inputs"][-1]["widget"] = {"name": name}
            output["links"].append(link_id)
    return {"id": str(uuid5(NAMESPACE_URL, "https://github.com/T8mars/Comfyui-BluePixel-T8/" + workflow_name)),
            "revision": 0, "last_node_id": max(int(key) for key in graph), "last_link_id": len(links),
            "nodes": nodes, "links": links, "groups": [], "config": {},
            "extra": {"ds": {"scale": 0.65, "offset": [40, 60]}}, "version": 0.4}


def main():
    directory = Path(__file__).resolve().parents[1] / "workflows"
    directory.mkdir(exist_ok=True)
    base = workflow()
    versions = {"01_edit_4k_int8_convrot": base}
    preview = deepcopy(base)
    preview["5"]["inputs"].update(target_pixels=1048576, output_width=1024, output_height=1024)
    versions["02_edit_1k_preview"] = preview
    rectangle = deepcopy(base)
    rectangle["5"]["inputs"].update(output_width=0, output_height=0)
    versions["03_preserve_source_aspect"] = rectangle
    training = deepcopy(base)
    training["10"]["inputs"]["mu_mode"] = "training"
    versions["04_training_mu"] = training
    fixed = deepcopy(base)
    fixed["10"]["inputs"]["shift"] = 3.0
    versions["05_fixed_shift"] = fixed
    multiple = deepcopy(preview)
    multiple["17"] = {"class_type": "LoadImage", "inputs": {"image": "bluepixel_test_forest.jpg"}}
    multiple["18"] = {"class_type": "BluePixelPrepareImage", "inputs": {"image": ["17", 0], "target_pixels": 1048576}}
    multiple["19"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["18", 0], "vae": ["3", 0]}}
    multiple["20"] = {"class_type": "BluePixelEditConditioning", "inputs": {"positive": ["8", 0], "negative": ["8", 1], "reference": ["19", 0]}}
    multiple["12"]["inputs"].update(positive=["20", 0], negative=["20", 1])
    multiple["6"]["inputs"]["instruction"] = "Place the person from the first image in the forest shown in the second image."
    versions["06_multiple_references"] = multiple
    text_only = deepcopy(preview)
    for key in ("4", "5", "7", "8"):
        del text_only[key]
    text_only["9"]["inputs"].update(width=1024, height=1024)
    text_only["12"]["inputs"].update(positive=["6", 0], negative=["6", 1])
    text_only["6"]["inputs"]["instruction"] = "A cozy cabin beside a forest lake at sunrise."
    versions["07_text_only_experimental"] = text_only
    batch = deepcopy(preview)
    batch["9"]["inputs"]["batch_size"] = 2
    batch["11"] = {"class_type": "BluePixelSeedListNoise", "inputs": {"seeds": "42,43", "mode": "upstream_cuda_bf16"}}
    versions["08_seed_list_batch"] = batch
    for name, graph in versions.items():
        (directory / (name + ".json")).write_text(json.dumps(to_ui(graph, name), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"canvas_workflows": len(versions), "files": len(versions)}))


if __name__ == "__main__":
    main()
