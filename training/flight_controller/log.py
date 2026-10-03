import json
import time
from collections.abc import Callable
from pathlib import Path


def logger(folder: Path) -> Callable[[dict], None]:
    folder.mkdir(parents=True, exist_ok=True)

    def log(record: dict) -> None:
        record = {"time": round(time.time()), **record}
        print(json.dumps(record), flush=True)
        with (folder / "log.jsonl").open("a") as file:
            file.write(json.dumps(record) + "\n")

    return log
