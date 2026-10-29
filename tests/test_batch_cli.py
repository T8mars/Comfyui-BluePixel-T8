import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from PIL import Image
import pytest
from bluepixel import execution


@pytest.fixture
def cli_batch(tmp_path, monkeypatch):
    # main() selects a device for its dedicated process; restore the test process.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    Image.new("RGB", (2, 2)).save(tmp_path / "input.png")
    source = tmp_path / "tasks.jsonl"
    source.write_text(json.dumps({"image_name": "input.png", "edit_prompt": "edit"}) + "\n")
    for name in ("diffusion", "text", "vae"):
        (tmp_path / name).write_bytes(name.encode())
    comfy = ModuleType("comfy")
    comfy.__path__ = []
    options = ModuleType("comfy.options")
    options.args_parsing = False
    comfy.options = options
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.options", options)
    native = ModuleType("bluepixel.native")
    loaded = []

    def load(*args, **kwargs):
        loaded.append(args)
        return object()

    native.load_diffusion = native.load_clip = native.BluePixelVAE = load
    monkeypatch.setitem(sys.modules, "bluepixel.native", native)
    script = Path(__file__).resolve().parents[1] / "tools/run_batch.py"
    spec = importlib.util.spec_from_file_location("bluepixel_batch_cli_test", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(sys, "argv", ["run_batch.py", "--comfyui", str(tmp_path),
        "--diffusion", str(tmp_path / "diffusion"), "--text-encoder", str(tmp_path / "text"),
        "--vae", str(tmp_path / "vae"), "--jsonl", str(source), "--out-dir", str(tmp_path / "output")])
    return SimpleNamespace(cli=cli, source=source, loaded=loaded)


@pytest.mark.parametrize("failed", [0, 1])
def test_cli_exit_status_reports_per_task_failures(cli_batch, monkeypatch, capsys, failed):
    monkeypatch.setattr(execution, "run_tasks", lambda *args, **kwargs: {
        "success": 1 - failed, "failed": failed, "skipped": 0,
        "total_tasks": 1, "done_tasks": 1 - failed, "remaining_tasks": failed})
    if failed:
        with pytest.raises(SystemExit) as stopped:
            cli_batch.cli.main()
        assert stopped.value.code == 1
    else:
        assert cli_batch.cli.main() is None
    final = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert final["stage"] == "complete" and final["failed"] == failed


def test_empty_force_rerun_does_not_load_models(cli_batch, monkeypatch, capsys):
    cli_batch.source.write_text("")
    sys.argv.append("--no-resume")

    def unexpected(*args, **kwargs):
        raise AssertionError("An empty batch must not execute")

    monkeypatch.setattr(execution, "run_tasks", unexpected)
    cli_batch.cli.main()
    assert not cli_batch.loaded
    final = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert final["models_loaded"] is False and final["total_tasks"] == 0
