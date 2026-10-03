import pytest

from jym import logits, players


def test_models_toml_names_the_players():
    known = players.known(env="flight")
    assert {f"{name}@cpu" for name in players.featured()} <= set(known)
    cpu, gpu = known["maverick@cpu"], known["maverick@gpu"]
    assert (cpu.title, gpu.title) == ("Maverick", "Maverick (GPU)") and "llama.cpp (GPU)" in gpu.deployed
    model = "Flight-Autopilot-Decision-Model-Maverick1.0"
    assert cpu.model == f"projectj/{model}@main/{model}-Q8_0.gguf"
    assert players.player(name="maverick", env="flight").id == "maverick@cpu"
    assert (known["reference"].title, known["reference"].per_run) == ("Autopilot", True)  # the environment's own
    with pytest.raises(ValueError, match="unknown player 'jev@gpu'"):
        players.player(name="jev@gpu", env="flight")


def test_any_gguf_file_plays_by_its_path(tmp_path, monkeypatch):
    (tmp_path / "my-model-Q4.gguf").write_bytes(b"GGUF")
    monkeypatch.chdir(tmp_path.parent)  # a path is relative to the current directory
    monkeypatch.setattr(logits, "LogitsPolicy", lambda *arguments: arguments)
    player = players.player(name=f"{tmp_path.name}/my-model-Q4.gguf@gpu", env="flight")
    assert (player.id, player.title) == ("my-model-Q4@gpu", "my-model-Q4 (GPU)")
    name, model, device = player.make(None)
    assert (name, model.path, device) == ("my-model-Q4@gpu", tmp_path / "my-model-Q4.gguf", "gpu")
    assert model.identity().startswith("my-model-Q4.gguf@")  # known by its content
    with pytest.raises(ValueError, match="no GGUF file"):
        players.player(name="missing.gguf", env="flight")
