"""Per-edit source-style callbacks, carried by an instance-local model clone."""

import torch


class StepController:
    def __init__(self, callback, prompt_embeds, tensor_inputs=("latents",)):
        if set(tensor_inputs) - {"latents", "prompt_embeds"}:
            raise ValueError("Step callback tensor inputs are latents and prompt_embeds")
        self.callback = callback
        self.prompt_embeds = prompt_embeds.detach().to(torch.bfloat16).clone()
        self.tensor_inputs = tuple(tensor_inputs)
        self.prompt_changed = False

    def diffusion_wrapper(self, previous=None):
        def apply(function, arguments):
            if self.prompt_changed:
                arguments = arguments.copy()
                arguments["c"] = arguments["c"].copy()
                context = arguments["c"]["c_crossattn"]
                branches = arguments["cond_or_uncond"]
                chunks = context.chunk(len(branches))
                replacement = []
                for branch, chunk in zip(branches, chunks):
                    if branch == 0:
                        prompt = self.prompt_embeds.to(chunk)
                        if prompt.shape[0] not in (1, chunk.shape[0]) or prompt.shape[1:] != chunk.shape[1:]:
                            raise ValueError("Callback prompt batch/length must match native conditioning")
                        chunk = prompt.expand(chunk.shape[0], -1, -1)
                    replacement.append(chunk)
                arguments["c"]["c_crossattn"] = torch.cat(replacement)
            if previous is not None:
                return previous(function, arguments)
            return function(arguments["input"], arguments["timestep"], **arguments["c"])
        return apply

    def step_end(self, index, timestep, samples):
        packed = samples.flatten(2).transpose(1, 2)
        available = {"latents": packed, "prompt_embeds": self.prompt_embeds}
        values = {key: available[key] for key in self.tensor_inputs}
        updates = self.callback(self, index, timestep, values)
        if not isinstance(updates, dict) or set(updates) - {"latents", "prompt_embeds"}:
            raise ValueError("Step callback must return a dict containing latents and/or prompt_embeds")
        latent = updates.get("latents", packed)
        if not torch.is_tensor(latent) or latent.shape != packed.shape or not torch.isfinite(latent).all():
            raise ValueError("Callback latents must be finite source packed [B,L,128] with the same shape")
        prompt = updates.get("prompt_embeds", self.prompt_embeds)
        if not torch.is_tensor(prompt) or prompt.shape != self.prompt_embeds.shape or not torch.isfinite(prompt).all():
            raise ValueError("Callback prompt_embeds must retain shape and be finite")
        if "prompt_embeds" in updates or "prompt_embeds" in self.tensor_inputs:
            self.prompt_changed = True
            self.prompt_embeds = prompt
        return latent.transpose(1, 2).reshape_as(samples).to(samples)
