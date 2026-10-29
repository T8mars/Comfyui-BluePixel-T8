import numpy as np
import torch
from bluepixel.sampling import source_timestep, source_post_cfg, sample_source_euler, seed_list_noise


def test_source_time_cast_sequence():
    sigmas = torch.tensor([1.0, 0.96738399, 0.90814392, 0.76719996])
    pipeline_time = (sigmas * 1000).bfloat16() / 1000
    transformer_time = (pipeline_time * 1000).float()
    torch.testing.assert_close(source_timestep(None, sigmas) * 1000, transformer_time, atol=1e-4, rtol=1e-7)


def test_seed_list_noise_matches_independent_generators():
    shape = (3, 128, 2, 3)
    result = seed_list_noise(shape, [42, 43, 42], "cpu", "comfy_cpu")
    assert torch.equal(result[0], result[2])
    assert not torch.equal(result[0], result[1])
    expected = torch.randn(shape[1:], generator=torch.Generator().manual_seed(43))
    assert torch.equal(result[1], expected)


def test_bf16_cfg_round_points():
    torch.manual_seed(19)
    sample = torch.randn(1, 128, 2, 2).bfloat16()
    positive = torch.randn_like(sample)
    empty = torch.randn_like(sample)
    sigma = torch.tensor([0.96738399])
    shaped = sigma.view(1, 1, 1, 1)
    result = source_post_cfg({"input": sample, "sigma": sigma, "cond_scale": 3.0,
        "cond_denoised": sample.float() - shaped * positive.float(),
        "uncond_denoised": sample.float() - shaped * empty.float()})
    expected = sample.float() - shaped * (empty + 3.0 * (positive - empty)).float()
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)


def test_bf16_euler_steps_match_diffusers():
    from diffusers import FlowMatchEulerDiscreteScheduler
    torch.manual_seed(20)
    scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
    scheduler.set_timesteps(sigmas=np.linspace(1.0, 0.25, 4), mu=2.2911798941)
    sigmas = scheduler.sigmas
    sample = torch.randn(1, 128, 2, 2).bfloat16()
    velocities = [torch.randn_like(sample) for _ in range(4)]
    expected = sample.clone()
    for index, timestep in enumerate(scheduler.timesteps):
        expected = scheduler.step(velocities[index], timestep, expected, return_dict=False)[0]
    position = [0]

    def model(x, sigma):
        velocity = velocities[position[0]]
        position[0] += 1
        return x.float() - sigma.reshape(1, 1, 1, 1) * velocity.float()

    actual = sample_source_euler(model, sample.float(), sigmas)
    assert torch.equal(actual, expected)
