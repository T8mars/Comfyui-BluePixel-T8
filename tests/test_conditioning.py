from bluepixel.conditioning import append_references


def test_chained_references_keep_index_routing_and_original_conditioning():
    embedding, first, second = object(), object(), object()
    original = [[embedding, {"attention_mask": "keep"}]]
    one = append_references(original, [first])
    two = append_references(one, [second])
    assert two[0][0] is embedding
    assert two[0][1]["reference_latents"] == [first, second]
    assert two[0][1]["reference_latents_method"] == "index"
    assert two[0][1]["attention_mask"] == "keep"
    assert one[0][1]["reference_latents"] == [first]
    assert "reference_latents" not in original[0][1]
