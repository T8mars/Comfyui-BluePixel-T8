import math
import numpy as np
import pytest
import torch
from bluepixel.image import select_size, prepare_image
from bluepixel.scheduling import schedule_values, compute_mu


@pytest.mark.parametrize("length", [1, 256, 4096, 4300, 4301, 16384])
@pytest.mark.parametrize("steps", [1, 4, 10, 50])
@pytest.mark.parametrize("mode,shift", [("empirical", 0), ("training", 0), ("empirical", 3.0)])
def test_schedule_against_diffusers(length, steps, mode, shift):
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.flux2.pipeline_flux2 import compute_empirical_mu
    ours, metadata = schedule_values(steps, length, mode, shift)
    mu = compute_empirical_mu(length, steps) if mode == "empirical" else math.log1p(length) / (math.log(2) + length)
    oracle = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=shift == 0,
        time_shift_type="exponential", shift=shift or 3.0)
    oracle.set_timesteps(sigmas=np.linspace(1.0, 1.0 / steps, steps), mu=mu)
    assert torch.allclose(torch.tensor(ours), oracle.sigmas.cpu(), atol=1e-6, rtol=1e-6)
    assert ours[-1] == 0
    assert all(a > b for a, b in zip(ours, ours[1:]))


def test_invalid_schedule_combinations():
    for arguments in [(0, 4096, "empirical", 0), (4, 0, "empirical", 0),
                      (4, 4096, "training", 1), (4, 4096, "empirical", float("nan")),
                      (4, 4096, "unknown", 0)]:
        with pytest.raises(ValueError):
            schedule_values(*arguments)


def test_preprocessing_reports_selected_and_processed_sizes():
    image = torch.rand(1, 700, 1100, 3)
    result, width, height, meta = prepare_image(image, 1024 * 1024)
    assert width % 64 == height % 64 == 0
    assert result.shape[1] % 128 == result.shape[2] % 128 == 0
    assert meta["output_size"] == [width, height]
    assert meta["processed_reference_size"] == [result.shape[2], result.shape[1]]
    assert meta["image_seq_len"] == width * height // 4096
    assert result.min() >= 0 and result.max() <= 1
    # Source tie-breaking chooses the first equal-aspect candidate (64px smaller).
    assert select_size(4096, 4096, 4096 * 4096) == (4032, 4032)
    _, width, height, meta = prepare_image(torch.rand(1, 128, 128, 3), 4096 * 4096, 4096, 4096)
    assert (width, height) == (4096, 4096)
    assert meta["image_seq_len"] == 4096
