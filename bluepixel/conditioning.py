"""Reference conditioning with append-only tensors and fixed routing metadata."""


def append_references(conditioning, references):
    result = []
    for embedding, metadata in conditioning:
        previous = metadata.get("reference_latents")
        previous = [] if previous is None else previous
        if not isinstance(previous, (list, tuple)):
            raise ValueError("reference_latents metadata must be a list of spatial tensors")
        values = metadata.copy()
        values["reference_latents"] = list(previous) + list(references)
        # Generic append=True concatenates strings too (index -> indexindex).
        # Routing must remain a scalar while reference tensors accumulate.
        values["reference_latents_method"] = "index"
        result.append([embedding, values])
    return result
