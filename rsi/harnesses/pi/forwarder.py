import asyncio
import collections
import dataclasses
import json
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Protocol

# forward model requests unchanged and log each one, since this is the only way out of the sandbox network
# runs by itself in the pi image, so stick to python 3.11 and its standard library

# leave messages and tool definitions out of request lines, and log long params by their size only
LEFT_OUT = ("messages", "tools", "functions")
LONGEST = 1000
# Docker Desktop's route to the host sometimes leaves a new connection hanging for 30+ s, while a retry on a new
# socket gets through at once. So retry, waiting a bit longer each time
CONNECT_TRIES = 5
CONNECT_WAIT_S = 2.0


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


def usage(raw: bytes, log: Log) -> bool:
    """Log the token usage a response body or stream event reports, and return whether it had any"""
    if b'"usage"' not in raw:
        return False
    try:
        data = json.loads(raw)
    except ValueError:
        return False
    found = data.get("usage") if isinstance(data, dict) else None
    if isinstance(found, dict):
        log(entry={"usage": found})
        return True
    return False


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
    """Yield the body as it arrives, whether it's chunked, sized by content-length, or runs until the connection ends"""
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
            # stop, since we don't parse chunked request bodies and so can't tell where the next request starts
            log(entry={"method": method, "path": path, "unread": "chunked body"})
            return
        sent = await stream.readexactly(int(headers.get("content-length", 0)))
        log(entry=request(method=method, path=path, body=sent))


async def read_responses(stream: asyncio.StreamReader, methods: collections.deque[str], log: Log) -> None:
    while (message := await head(stream=stream)) is not None:
        (_, status, *_), headers = message
        if status == "101":
            return  # switching protocols, so the rest isn't HTTP
        if status.startswith("1"):
            continue  # an interim response, before the real one
        method = methods.popleft() if methods else ""
        if method == "HEAD" or status in ("204", "304"):
            continue
        events = headers.get("content-type", "").lower().startswith("text/event-stream")
        rest = b""
        chunks, reported = 0, False
        try:
            async for piece in body(stream=stream, headers=headers):
                rest += piece
                if events:
                    *lines, rest = rest.split(b"\n")
                    for line in lines:
                        reported |= usage(raw=line.removeprefix(b"data:"), log=log)
                        if b'"delta"' in line:
                            chunks += 1
            reported |= usage(raw=rest.removeprefix(b"data:"), log=log)
        finally:
            # if a stream ends before the server reports usage (e.g. the harness cut the response or the task timed
            # out), still count its tokens from the number of chunks, since the servers send about one token per chunk
            if chunks and not reported:
                log(entry={"usage": {"completion_tokens": chunks}, "from": "stream chunks"})


class Tap:
    """Reads a copy of the bytes going one way on a connection, without holding them up"""

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


async def connect(host: str, port: int, log: Log) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    start = time.monotonic()
    tries = 1
    while True:
        try:
            streams = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=CONNECT_WAIT_S * tries)
            break
        except TimeoutError:
            if tries == CONNECT_TRIES:
                raise
            tries += 1
    if tries > 1:
        log(entry={"connect": {"tries": tries, "seconds": round(time.monotonic() - start, 1)}})
    return streams


def logger(path: str, ip: str) -> Log:
    def log(entry: dict) -> None:
        # reopen the file for each line, so a moved or deleted log gets created again
        with open(path, "a") as fh:
            fh.write(json.dumps({"time": round(time.time(), 3), "ip": ip, **entry}) + "\n")

    return log


@dataclasses.dataclass(frozen=True)
class Upstream:
    """The model server to forward to, and where to log what each container sends it"""

    host: str
    port: int
    log_path: str

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        log = logger(path=self.log_path, ip=writer.get_extra_info("peername")[0])
        try:
            upstream_reader, upstream_writer = await connect(host=self.host, port=self.port, log=log)
        except OSError:  # TimeoutError is an OSError too
            writer.close()
            return
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
    open(log_path, "a").close()  # create the log now, so a run can tell this forwarder from one that logs nothing
    asyncio.run(serve(upstream=Upstream(host=host, port=port, log_path=log_path)))
