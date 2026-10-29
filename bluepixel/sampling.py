"""Source BF16 time/CFG/Euler math on standard ComfyUI sampling interfaces."""

import torch


def seed_list_noise(shape, seeds, device, mode="upstream_cuda_bf16"):
    if len(seeds) != shape[0]:
        raise ValueError("Seed list length must equal latent batch size")
    if mode not in {"upstream_cuda_bf16", "comfy_cpu"}:
        raise ValueError("Unknown noise mode")
    if mode == "upstream_cuda_bf16" and torch.device(device).type != "cuda":
        raise ValueError("Source CUDA BF16 noise requires CUDA")
    dtype = torch.bfloat16 if mode == "upstream_cuda_bf16" else torch.float32
    return torch.cat([torch.randn((1, *shape[1:]), device=device, dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed)) for seed in seeds], dim=0)


def source_timestep(module, sigma):
    # Source pipeline: scheduler t -> BF16 -> /1000; transformer: BF16 *1000.
    # Comfy's Flux embedding multiplies a FP32 time by1000, so precompute the
    # source-rounded embedding time and undo that final factor in FP32.
    time = (sigma * 1000.0).to(torch.bfloat16)
    time = (time / 1000.0) * 1000.0
    return time.float() / 1000.0


def source_post_cfg(arguments):
    sample = arguments["input"].float()
    sigma = arguments["sigma"].reshape((-1,) + (1,) * (sample.ndim - 1))
    conditional = ((sample - arguments["cond_denoised"].float()) / sigma).to(torch.bfloat16)
    scale = arguments["cond_scale"]
    if scale > 1 and arguments["uncond_denoised"] is not None:
        empty = ((sample - arguments["uncond_denoised"].float()) / sigma).to(torch.bfloat16)
        velocity = empty + scale * (conditional - empty)
    else:
        velocity = conditional
    return sample - sigma * velocity.float()


@torch.no_grad()
def sample_source_euler(model, x, sigmas, extra_args=None, callback=None, disable=None, step_controller=None):
    extra_args = extra_args or {}
    x = x.to(torch.bfloat16)
    ones = torch.ones(x.shape[0], device=x.device, dtype=torch.float32)
    for index in range(len(sigmas) - 1):
        sigma = sigmas[index]
        denoised = model(x, sigma * ones, **extra_args)
        velocity = ((x.float() - denoised.float()) / sigma).to(torch.bfloat16)
        if callback is not None:
            callback({"x": x, "i": index, "sigma": sigma, "sigma_hat": sigma, "denoised": denoised})
        dt = sigmas[index + 1] - sigma
        # Keep the source's scalar*tensor round point, FP32 sample addition,
        # then BF16 sample storage on EVERY step, including the last step.
        x = (x.float() + dt * velocity).to(torch.bfloat16)
        if step_controller is not None:
            x = step_controller.step_end(index, sigma * 1000.0, x)
    return x


def make_sampler(step_controller=None):
    import comfy.samplers
    return comfy.samplers.KSAMPLER(sample_source_euler,
        extra_options={"step_controller": step_controller} if step_controller is not None else {})
