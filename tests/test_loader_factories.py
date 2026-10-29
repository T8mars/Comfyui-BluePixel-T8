from types import SimpleNamespace

from bluepixel.identity import register_patcher_loader


def adapted_loader(path, dtype="default", disable_dynamic=False):
    return SimpleNamespace(path=path, dtype=dtype, disable_dynamic=disable_dynamic,
                           spatial_ratio=64, source_timestep=True)


def test_reload_factory_preserves_the_custom_loader_and_options(tmp_path):
    patcher = SimpleNamespace()
    path = tmp_path / "model.safetensors"
    register_patcher_loader(patcher, adapted_loader, path, dtype="bf16", disable_dynamic=True)
    loader, arguments = patcher.cached_patcher_init
    clone = loader(*arguments)
    assert clone.path == str(path.resolve())
    assert clone.dtype == "bf16" and clone.disable_dynamic
    assert clone.spatial_ratio == 64 and clone.source_timestep


def test_native_clone_can_override_dynamic_loading(tmp_path):
    patcher = SimpleNamespace()
    register_patcher_loader(patcher, adapted_loader, tmp_path / "model", dtype="fp32", disable_dynamic=False)
    loader, arguments = patcher.cached_patcher_init
    assert not loader(*arguments).disable_dynamic
    assert loader(*arguments, disable_dynamic=True).disable_dynamic


def test_reload_factory_remains_valid_after_working_directory_changes(tmp_path, monkeypatch):
    patcher = SimpleNamespace()
    monkeypatch.chdir(tmp_path)
    register_patcher_loader(patcher, adapted_loader, "model.safetensors")
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    loader, arguments = patcher.cached_patcher_init
    assert loader(*arguments).path == str(tmp_path / "model.safetensors")
