import json

from typer.testing import CliRunner

from jym import cli


def test_run_prints_the_result(tmp_path):
    arguments = ["run", "--task", "drone-takeoff", "--player", "reference"]
    result = CliRunner().invoke(cli.app, [*arguments, "--runs-dir", str(tmp_path)])
    assert result.exit_code == 0 and json.loads(result.output)["status"] == "success"


def test_ui_rejects_unknown_players():
    result = CliRunner().invoke(cli.app, ["ui", "--players", "maverick,nobody"])
    assert result.exit_code == 2 and "unknown player 'nobody'" in result.output
