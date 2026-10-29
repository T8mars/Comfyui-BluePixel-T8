"""BluePixel's three FlowMatch Euler schedules; no model imports."""

import math


def positive_int(value, name):
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def empirical_mu(image_seq_len, steps):
    length = positive_int(image_seq_len, "image_seq_len")
    steps = positive_int(steps, "steps")
    m200 = 0.00016927 * length + 0.45666666
    if length > 4300:
        return m200
    m10 = 8.73809524e-05 * length + 1.89833333
    slope = (m200 - m10) / 190.0
    return slope * steps + m200 - 200.0 * slope


def compute_mu(image_seq_len, steps, mode="empirical"):
    length = positive_int(image_seq_len, "image_seq_len")
    if mode == "empirical":
        return empirical_mu(length, steps)
    if mode == "training":
        return math.log1p(length) / (math.log(2.0) + length)
    raise ValueError(f"Unknown mu_mode: {mode}")


def schedule_values(steps, image_seq_len, mu_mode="empirical", shift=0.0):
    steps = positive_int(steps, "steps")
    shift = float(shift)
    if not math.isfinite(shift) or shift < 0:
        raise ValueError("shift must be finite and non-negative")
    if mu_mode == "training" and shift != 0:
        raise ValueError("mu_mode=training requires shift=0")
    mu = compute_mu(image_seq_len, steps, mu_mode)
    factor = shift if shift > 0 else math.exp(mu)
    sigmas = []
    for index in range(steps):
        sigma = (steps - index) / steps
        sigmas.append(factor * sigma / (1.0 + (factor - 1.0) * sigma))
    sigmas.append(0.0)
    return sigmas, {
        "mu": None if shift > 0 else mu,
        "mu_mode": "fixed" if shift > 0 else mu_mode,
        "shift": shift, "image_seq_len": image_seq_len, "steps": steps,
    }
