"""Instance-local Qwen3 conditioning precision conventions."""

import types
import threading
import torch

_SDPA_PRECISION_LOCK = threading.RLock()


def reference_frequencies(head_dim, theta, position_ids, device):
    inverse = 1.0 / (float(theta) ** (torch.arange(0, head_dim, 2,
        dtype=torch.float32, device="cpu") / head_dim))
    with torch.autocast(device_type=torch.device(device).type, enabled=False):
        frequencies = (inverse[None, :, None].to(device).expand(position_ids.shape[0], -1, 1)
                       @ position_ids[:, None, :].float()).transpose(1, 2)
        angles = torch.cat((frequencies, frequencies), dim=-1)
        cos, sin = angles.cos().unsqueeze(1), angles.sin().unsqueeze(1)
    half = sin.shape[-1] // 2
    return cos, sin[..., :half], -sin[..., half:]


def reference_sdpa(query, key, value, **kwargs):
    if not query.is_cuda:
        return torch.nn.functional.scaled_dot_product_attention(query, key, value, **kwargs)
    # Match PyTorch's source SDPA accumulation while retaining its native BF16
    # tensor path. Restore the host preference even when dispatch raises.
    with _SDPA_PRECISION_LOCK:
        previous = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
        try:
            if previous:
                torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
            return torch.nn.functional.scaled_dot_product_attention(query, key, value, **kwargs)
        finally:
            if previous:
                torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(previous)


def source_rms_norm(module, values):
    # Qwen3 rounds the normalized values to the activation dtype BEFORE the
    # learned scale multiplication; fused RMS kernels may round only afterwards.
    normalized = values.float() * torch.rsqrt(values.float().square().mean(-1, keepdim=True) + module.eps)
    return module.scale().to(values) * normalized.to(values.dtype)


def source_attention(module, hidden_states, attention_mask=None, freqs_cis=None,
                     optimized_attention=None, past_key_value=None, sliding_window=None):
    if past_key_value is not None:
        raise ValueError("BluePixel text conditioning does not use a generation KV cache")
    batch, length, _ = hidden_states.shape
    query = module.q_proj(hidden_states).view(batch, length, module.num_heads, module.head_dim)
    key = module.k_proj(hidden_states).view(batch, length, module.num_kv_heads, module.head_dim)
    value = module.v_proj(hidden_states).view(batch, length, module.num_kv_heads, module.head_dim)
    query = module.q_norm(query).transpose(1, 2)
    key = module.k_norm(key).transpose(1, 2)
    value = value.transpose(1, 2)
    cos, half_sin, negative_half_sin = freqs_cis
    cos = cos.to(query.dtype)
    sin = torch.cat((half_sin, -negative_half_sin), dim=-1).to(query.dtype)

    def rotate_half(tensor):
        first, second = tensor.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    # Keep the source's BF16 multiply/add round points instead of the generic
    # fused FP32 RoPE implementation used by other ComfyUI language models.
    query = query * cos + rotate_half(query) * sin
    key = key * cos + rotate_half(key) * sin
    if attention_mask is not None:
        attention_mask = attention_mask.to(query.dtype)
        attention_mask = attention_mask.masked_fill(attention_mask != 0, torch.finfo(query.dtype).min)
        # Transformers elides an all-attended mask and selects SDPA's causal
        # path. This matters for truncated prompts with no remaining padding.
        if not torch.any(attention_mask[..., -1, :] != 0).item():
            attention_mask = None
    version = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
    use_gqa = attention_mask is None and version >= (2, 5)
    if not use_gqa:
        repeats = module.num_heads // module.num_kv_heads
        key = key[:, :, None].expand(batch, module.num_kv_heads, repeats, length, module.head_dim).reshape(batch, module.num_heads, length, module.head_dim)
        value = value[:, :, None].expand(batch, module.num_kv_heads, repeats, length, module.head_dim).reshape(batch, module.num_heads, length, module.head_dim)
    kwargs = {"attn_mask": attention_mask, "dropout_p": 0.0,
              "is_causal": attention_mask is None and length > 1,
              **({"enable_gqa": True} if use_gqa else {})}
    output = reference_sdpa(query, key, value, **kwargs)
    output = output.transpose(1, 2).contiguous().reshape(batch, length, module.inner_size)
    return module.o_proj(output), None


def install_reference_layer_options(text_model):
    original_options = text_model.set_clip_options

    def source_options(self, options):
        original_options(options)
        layers = options.get("layer")
        # Generic Qwen preserves its fixed default layer list even when a CLIP
        # clone requests another list. BluePixel exposes those taps explicitly.
        if isinstance(layers, list):
            self.layer = layers.copy()

    text_model.set_clip_options = types.MethodType(source_options, text_model)


def install_reference_precision(transformer):
    from comfy.text_encoders.llama import RMSNorm, Attention
    for module in transformer.modules():
        if isinstance(module, RMSNorm):
            module.forward = types.MethodType(source_rms_norm, module)
        elif isinstance(module, Attention):
            module.forward = types.MethodType(source_attention, module)
    # HF initializes RoPE inverse frequencies on CPU before moving the module
    # to CUDA. CUDA pow differs just enough to cross a BF16 sine rounding tie
    # at token459 (channels18/82). Preserve the source CPU constant, per model.
    def source_frequencies(self, position_ids, device):
        return reference_frequencies(self.config.head_dim, self.config.rope_theta, position_ids, device)

    transformer.model.compute_freqs_cis = types.MethodType(source_frequencies, transformer.model)
    original_forward = transformer.forward

    def source_precision_forward(self, *args, **kwargs):
        if kwargs.get("embeds") is not None:
            kwargs["embeds"] = kwargs["embeds"].to(torch.bfloat16)
        kwargs["dtype"] = torch.bfloat16
        return original_forward(*args, **kwargs)

    transformer.forward = types.MethodType(source_precision_forward, transformer)
