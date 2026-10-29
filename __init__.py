"""ComfyUI-BluePixel: split native models and INT8 ConvRot inference."""

import importlib.util

# The offline converter and its tests do not require a ComfyUI installation.
# When ComfyUI is present, errors in node imports are intentionally propagated.
if importlib.util.find_spec("folder_paths") is not None:
    from .bluepixel_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
else:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
