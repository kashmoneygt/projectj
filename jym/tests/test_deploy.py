import re
import tomllib
from pathlib import Path

from jym import players

DEPLOY = Path(__file__).parents[1] / "deploy"


def test_deploy_ports_agree():
    port = re.search(r"ENV PORT=(\d+)", (DEPLOY / "Dockerfile").read_text()).group(1)
    assert set(re.findall(r"\b(?:targetPort|port): (\d+)", (DEPLOY / "azure.sh").read_text())) == {port}


def test_hosted_players_are_known():
    with open(DEPLOY / "jym.toml", "rb") as file:
        settings = tomllib.load(file)
    models = players.models()["models"]
    for player in [*settings["players"], *settings["recorded"]]:
        name, _, device = player.partition("@")
        assert player == "reference" or (name in models and device in ("", *players.DEVICES)), player


def test_container_serves_its_models_through_jym():
    dockerfile = (DEPLOY / "Dockerfile").read_text()
    server = re.search(r"LLAMA_SERVER=(\S+)", dockerfile).group(1)
    assert f"COPY --from=llama /app {Path(server).parent}\n" in dockerfile  # the pinned llama.cpp build
    mount = re.search(r"mountPath: (\S+)", (DEPLOY / "azure.sh").read_text()).group(1)
    assert re.search(rf"JYM_MODELS={mount}/", dockerfile)  # downloaded once to the file share
