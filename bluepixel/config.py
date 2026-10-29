"""Versioned architecture and artifact contracts, independent of ComfyUI."""

SOURCE_REPO = "openvivo/BluePixel-Turbo"
SOURCE_REVISION = "e9c7a641a4bcd9851eef50c1ce9b07fa44f6044c"
CODE_REVISION = "3a39b77133b1e08d724a0f45b316ccfa5c3b2acc"
FORMAT_VERSION = 1
SPATIAL_RATIO = 64
LATENT_CHANNELS = 128
LATENT_SCALE = 0.4538
CONTEXT_DIM = 12288
DEFAULT_PIXELS = 4096 * 4096
DEFAULT_LAYERS = (9, 18, 27)
CHAT_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def validate_transformer_config(config):
    expected = {
        "in_channels": 128, "patch_size": 1,
        "joint_attention_dim": 12288, "num_attention_heads": 32,
        "attention_head_dim": 128, "num_layers": 8,
        "num_single_layers": 24, "guidance_embeds": False,
        "axes_dims_rope": [32, 32, 32, 32], "rope_theta": 2000,
    }
    mismatches = {k: (config.get(k), v) for k, v in expected.items() if config.get(k) != v}
    if mismatches:
        raise ValueError(f"Not a supported BluePixel transformer configuration: {mismatches}")
