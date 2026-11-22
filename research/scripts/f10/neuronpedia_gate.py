#!/usr/bin/env python
"""Forward 1 TCP port to the local Neuronpedia viewer, for the login node only.

The viewer has no login, and the node has a routable address. Thus the viewer listens
on 127.0.0.1, and this gate accepts connections from the listed addresses only. The
gate logs each refused address, thus the log shows a login node with a new address.

Use:
    python research/scripts/f10/neuronpedia_gate.py --port 3000 --target 3001 --allow 128.84.96.23
"""
from __future__ import annotations

import argparse
import asyncio


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except ConnectionError:
        pass
    finally:
        writer.close()


def serve(target: int, allowed: set[str]):
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")[0].removeprefix("::ffff:")
        if peer not in allowed:
            print(f"[gate] refused {peer}", flush=True)
            writer.close()
            return
        try:
            up_reader, up_writer = await asyncio.open_connection("127.0.0.1", target)
        except OSError:
            writer.close()
            return
        await asyncio.gather(pipe(reader, up_writer), pipe(up_reader, writer))

    return handle


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--allow", nargs="+", required=True)
    args = ap.parse_args()
    server = await asyncio.start_server(serve(args.target, {"127.0.0.1", *args.allow}), "0.0.0.0", args.port)
    print(f"[gate] port {args.port} to 127.0.0.1:{args.target}, allow {args.allow}", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
