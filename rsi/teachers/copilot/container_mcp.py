import json
import subprocess
import sys
import threading

OUTPUT_CHARS = 15_000  # per tool result (Copilot only previews results over ~20 KB)
DEFAULT_TIMEOUT_S = 600
MAX_TIMEOUT_S = 3600
READ_LINES = 400

TOOLS = [
    {
        "name": "bash",
        "description": (
            "Run a bash command in the Linux container you work in. Each call starts a fresh shell in /work, which "
            "is read-only (write under /out or /tmp), so cd and variables don't carry over between calls. It has "
            "git, python3, pytest, node, pi, check-variant and try-variant, and no internet. Returns the exit code "
            "and the output (stdout and stderr together; long output keeps its start and end)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "the bash command"},
                "timeout_s": {
                    "type": "integer",
                    "description": f"seconds before the command is killed (default {DEFAULT_TIMEOUT_S}, "
                    f"at most {MAX_TIMEOUT_S})",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": (
            f"Read a text file in the container, with line numbers: {READ_LINES} lines from start_line unless you "
            f"ask for more, and at most about {OUTPUT_CHARS:,} characters. The end of the result says where to "
            "continue."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "absolute path"},
                "start_line": {"type": "integer", "description": "first line to show, from 1"},
                "lines": {"type": "integer", "description": "how many lines to show"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Create or overwrite a file in the container with this content (its folders are made too).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "absolute path"},
                "content": {"type": "string", "description": "the whole new content"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Replace one exact piece of text in a file in the container. old_text has to occur exactly once; "
            "include enough surrounding lines to make it unique."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "absolute path"},
                "old_text": {"type": "string", "description": "the exact text to replace"},
                "new_text": {"type": "string", "description": "what to put in its place"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
]


def in_container(container: str, args: list[str], stdin: bytes | None, timeout_s: int) -> tuple[int, bytes]:
    """Run a command in the container, under its own `timeout` so nothing outlives us"""
    command = ["docker", "exec", "-i", container, "timeout", "-k", "5", str(timeout_s), *args]
    try:
        done = subprocess.run(command, input=stdin, capture_output=True, timeout=timeout_s + 30, check=False)
    except subprocess.TimeoutExpired:
        return 124, b"(the command did not return in time)"
    return done.returncode, done.stdout + done.stderr


def clip(text: str, limit: int) -> str:
    """Keep the start and end of text over the limit"""
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n… ({len(text) - limit} characters left out) …\n{text[-half:]}"


def bash(container: str, arguments: dict) -> tuple[str, bool]:
    timeout_s = max(1, min(int(arguments.get("timeout_s") or DEFAULT_TIMEOUT_S), MAX_TIMEOUT_S))
    command = ["bash", "-lc", f"cd /work && {arguments['command']}"]
    code, output = in_container(container=container, args=command, stdin=None, timeout_s=timeout_s)
    note = f" (killed after {timeout_s} s)" if code == 124 else ""
    return f"exit code {code}{note}\n{clip(text=output.decode(errors='replace'), limit=OUTPUT_CHARS)}", code != 0


def read_file(container: str, arguments: dict) -> tuple[str, bool]:
    code, output = in_container(container=container, args=["cat", "--", arguments["path"]], stdin=None, timeout_s=60)
    text = output.decode(errors="replace")
    if code != 0:
        return text, True
    lines = text.splitlines()
    start = max(1, int(arguments.get("start_line") or 1))
    last = min(len(lines), start - 1 + max(1, int(arguments.get("lines") or READ_LINES)))
    shown: list[str] = []
    size = 0
    for number in range(start, last + 1):
        line = f"{number}. {lines[number - 1]}"
        if shown and size + len(line) > OUTPUT_CHARS:
            break
        shown.append(line)
        size += len(line) + 1
    after = start + len(shown)
    tail = f"\n(continue at start_line={after}; the file has {len(lines)} lines)" if after <= len(lines) else ""
    return clip(text="\n".join(shown), limit=OUTPUT_CHARS) + tail, False


def write_file(container: str, arguments: dict) -> tuple[str, bool]:
    script = 'mkdir -p "$(dirname "$1")" && cat > "$1"'
    code, output = in_container(
        container=container,
        args=["sh", "-c", script, "sh", arguments["path"]],
        stdin=arguments["content"].encode(),
        timeout_s=60,
    )
    if code != 0:
        return output.decode(errors="replace"), True
    return f"wrote {arguments['path']}", False


def edit_file(container: str, arguments: dict) -> tuple[str, bool]:
    code, output = in_container(container=container, args=["cat", "--", arguments["path"]], stdin=None, timeout_s=60)
    if code != 0:
        return output.decode(errors="replace"), True
    text = output.decode()
    found = text.count(arguments["old_text"])
    if found != 1:
        return f"old_text occurs {found} times in {arguments['path']}; it has to occur exactly once", True
    return write_file(
        container=container,
        arguments={"path": arguments["path"], "content": text.replace(arguments["old_text"], arguments["new_text"])},
    )


HANDLERS = {"bash": bash, "read_file": read_file, "write_file": write_file, "edit_file": edit_file}
_write_lock = threading.Lock()


def send(message: dict) -> None:
    with _write_lock:
        sys.stdout.write(json.dumps(message) + "\n")
        sys.stdout.flush()


def handle(container: str, message: dict) -> None:
    if "id" not in message:
        return  # a notification, like notifications/initialized
    reply: dict = {"jsonrpc": "2.0", "id": message["id"]}
    try:
        reply |= answer(container=container, method=message.get("method"), params=message.get("params") or {})
    except Exception as exc:  # noqa: BLE001
        reply["error"] = {"code": -32603, "message": f"{type(exc).__name__}: {exc}"}
    send(message=reply)


def answer(container: str, method: object, params: dict) -> dict:
    if method == "initialize":
        return {
            "result": {
                "protocolVersion": params.get("protocolVersion", "2025-06-18"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "container", "version": "1"},
            }
        }
    if method == "tools/list":
        return {"result": {"tools": TOOLS}}
    if method == "tools/call" and params.get("name") in HANDLERS:
        try:
            text, is_error = HANDLERS[params["name"]](container=container, arguments=params.get("arguments") or {})
        except Exception as exc:  # noqa: BLE001
            text, is_error = f"{type(exc).__name__}: {exc}", True
        return {"result": {"content": [{"type": "text", "text": text}], "isError": is_error}}
    if method == "tools/call":
        return {"error": {"code": -32601, "message": "unknown tool"}}
    if method == "ping":
        return {"result": {}}
    return {"error": {"code": -32601, "message": f"no method {method}"}}


def main() -> None:
    container = sys.argv[1]
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except ValueError as exc:
            send(message={"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": f"parse error: {exc}"}})
            continue
        if not isinstance(message, dict):
            send(message={"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "not a request"}})
            continue
        # a tool call can take minutes, so each request gets its own thread
        threading.Thread(target=handle, kwargs={"container": container, "message": message}, daemon=True).start()


if __name__ == "__main__":
    main()
