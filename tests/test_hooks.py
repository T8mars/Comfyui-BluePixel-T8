import torch
import pytest
from bluepixel.hooks import StepController
from bluepixel.sampling import sample_source_euler


def test_step_callback_updates_last_latent_and_positive_only():
    prompt = torch.zeros(1, 2, 12288)
    seen = []
    def callback(controller, index, time, values):
        seen.append((index, float(time), tuple(values["latents"].shape)))
        return {"latents": values["latents"] + 1, "prompt_embeds": prompt + 2}
    controller = StepController(callback, prompt)
    def model(x, sigma):
        return x.float()
    result = sample_source_euler(model, torch.zeros(1, 128, 2, 3), torch.tensor([1., .5, 0.]), step_controller=controller)
    assert torch.all(result == 2)
    assert seen == [(0, 1000., (1, 6, 128)), (1, 500., (1, 6, 128))]
    arguments = {"input": None, "timestep": None, "c": {"c_crossattn": torch.zeros(2, 2, 12288)}, "cond_or_uncond": [1, 0]}
    context = controller.diffusion_wrapper()(lambda x, t, **c: c["c_crossattn"], arguments)
    assert torch.all(context[0] == 0) and torch.all(context[1] == 2)
    assert torch.all(arguments["c"]["c_crossattn"] == 0)


def test_callback_rejects_nonfinite_updates():
    controller = StepController(lambda *args: {"latents": torch.full((1, 4, 128), float("nan"))}, torch.zeros(1, 2, 12288))
    with pytest.raises(ValueError, match="finite"):
        controller.step_end(0, 1000, torch.zeros(1, 128, 2, 2))
