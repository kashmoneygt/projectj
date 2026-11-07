import asyncio
import collections
import dataclasses
import json
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Protocol

# forwards a task's model requests as is and logs each one, since it's the task's only way out
# runs on its own in the pi image, so stick to python 3.11

# request lines leave out the conversation, and log longer parameters by their size
LEFT_OUT = ("messages", "tools", "functions")
LONGEST = 1000


class Log(Protocol):
    def __call__(self, entry: dict) -> None: ...


Read = Callable[..., Awaitable[None]]  # read_requests or read_responses


def request(method: str, path: str, body: bytes) -> dict:
    line = {"method": method, "path": path}
    if not body:
        return line
    try:
        params = json.loads(body)
    except ValueError:
        params = None
    if not isinstance(params, dict):
        return {**line, "unread": "body is not a JSON object"}
    kept = {}
    for key, value in params.items():
        if key not in LEFT_OUT:
            size = len(json.dumps(value))
            kept[key] = value if size <= LONGEST else f"({size} bytes)"
    return {**line, "params": kept}


def usage(raw: bytes, log: Log) -> None:
    """Log the usage a response body or stream event reports"""
    if b'"usage"' not in raw:
        return
    try:
        data = json.loads(raw)
    except ValueError:
        return
    found = data.get("usage") if isinstance(data, dict) else None
    if isinstance(found, dict):
        log(entry={"usage": found})


async def head(stream: asyncio.StreamReader) -> tuple[list[str], dict[str, str]] | None:
    """The start line and headers, or None if the connection ends"""
    try:
        raw = await stream.readuntil(b"\r\n\r\n")
    except asyncio.IncompleteReadError as exc:
        if exc.partial.strip():
            raise
        return None
    start, *fields = raw.decode("latin-1").split("\r\n")
    headers = {}
    for field in fields:
        name, _, value = field.partition(":")
        headers[name.strip().lower()] = value.strip()
    return start.split(" ", 2), headers


async def body(stream: asyncio.StreamReader, headers: dict[str, str]) -> AsyncIterator[bytes]:
    """The body as it arrives: chunked, by length, or until the connection ends"""
    if "chunked" in headers.get("transfer-encoding", "").lower():
        while size := int((await stream.readuntil(b"\r\n")).split(b";")[0], 16):
            yield await stream.readexactly(size)
            await stream.readexactly(2)
        while await stream.readuntil(b"\r\n") != b"\r\n":
            pass
    elif "content-length" in headers:
        yield await stream.readexactly(int(headers["content-length"]))
    else:
        while piece := await stream.read(65536):
            yield piece


async def read_requests(stream: asyncio.StreamReader, methods: collections.deque[str], log: Log) -> None:
    while (message := await head(stream=stream)) is not None:
        (method, path, _), headers = message
        methods.append(method)
        if "chunked" in headers.get("transfer-encoding", "").lower():
            # the next request starts somewhere in chunks we don't read
            log(entry={"method": method, "path": path, "unread": "chunked body"})
            return
        sent = await stream.readexactly(int(headers.get("content-length", 0)))
        log(entry=request(method=method, path=path, body=sent))


async def read_responses(stream: asyncio.StreamReader, methods: collections.deque[str], log: Log) -> None:
    while (message := await head(stream=stream)) is not None:
        (_, status, *_), headers = message
        if status == "101":
            return  # the connection leaves HTTP
        if status.startswith("1"):
            continue  # an interim response, before the real one
        method = methods.popleft() if methods else ""
        if method == "HEAD" or status in ("204", "304"):
            continue
        events = headers.get("content-type", "").lower().startswith("text/event-stream")
        rest = b""
        async for piece in body(stream=stream, headers=headers):
            rest += piece
            if events:
                *lines, rest = rest.split(b"\n")
                for line in lines:
                    usage(raw=line.removeprefix(b"data:"), log=log)
        usage(raw=rest.removeprefix(b"data:"), log=log)


class Tap:
    """Reads a copy of one direction of a connection without holding up its bytes"""

    def __init__(self, read: Read, methods: collections.deque[str], log: Log) -> None:
        self.stream: asyncio.StreamReader | None = asyncio.StreamReader()
        self.done = asyncio.create_task(self.watch(read=read, methods=methods, log=log))

    async def watch(self, read: Read, methods: collections.deque[str], log: Log) -> None:
        try:
            await read(stream=self.stream, methods=methods, log=log)
        except asyncio.IncompleteReadError:
            pass  # the connection ended inside a message
        except Exception as exc:  # noqa: BLE001
            log(entry={"unread": f"{read.__name__}: {type(exc).__name__}: {exc}"})
        finally:
            self.stream = None

    def feed(self, data: bytes) -> None:
        if self.stream is not None:
            self.stream.feed_data(data)

    def end(self) -> None:
        if self.stream is not None:
            self.stream.feed_eof()


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, tap: Tap) -> None:
    try:
        while data := await reader.read(65536):
            writer.write(data)
            tap.feed(data=data)
            await writer.drain()
    except ConnectionError:
        pass
    finally:
        tap.end()
        writer.close()


def logger(path: str, ip: str) -> Log:
    def log(entry: dict) -> None:
        # reopened per line, so a moved or deleted log starts again
        with open(path, "a") as fh:
            fh.write(json.dumps({"time": round(time.time(), 3), "ip": ip, **entry}) + "\n")

    return log


@dataclasses.dataclass(frozen=True)
class Upstream:
    """The model server, and the log of what each container sent it"""

    host: str
    port: int
    log_path: str

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            upstream_reader, upstream_writer = await asyncio.open_connection(self.host, self.port)
        except OSError:
            writer.close()
            return
        log = logger(path=self.log_path, ip=writer.get_extra_info("peername")[0])
        methods: collections.deque[str] = collections.deque()
        sent = Tap(read=read_requests, methods=methods, log=log)
        received = Tap(read=read_responses, methods=methods, log=log)
        await asyncio.gather(
            pipe(reader=reader, writer=upstream_writer, tap=sent),
            pipe(reader=upstream_reader, writer=writer, tap=received),
        )
        await asyncio.gather(sent.done, received.done)


async def serve(upstream: Upstream) -> None:
    server = await asyncio.start_server(upstream.handle, "0.0.0.0", upstream.port)
    await server.serve_forever()


if __name__ == "__main__":
    host, port, log_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    open(log_path, "a").close()  # there from the start, so a run can tell this forwarder from one that logs nothing
    asyncio.run(serve(upstream=Upstream(host=host, port=port, log_path=log_path)))
