"""Local OpenPI-compatible server, with bounded messages and serialized inference."""
import asyncio
import logging


def serve_policy(policy, port):
    from openpi_client import msgpack_numpy
    from websockets.asyncio.server import serve

    async def run():
        lock = asyncio.Lock()

        async def handler(socket):
            packer = msgpack_numpy.Packer()
            await socket.send(packer.pack(policy.metadata))
            async for raw in socket:
                try:
                    observation = msgpack_numpy.unpackb(raw)
                    async with lock:
                        result = await asyncio.to_thread(policy.infer, observation)
                    await socket.send(packer.pack(result))
                except Exception as exc:
                    logging.exception('CLOAK inference failed')
                    await socket.send(str(exc))
                    await socket.close(code=1011, reason='Inference failed')
                    return

        async with serve(handler, '127.0.0.1', port, compression=None, max_size=16 * 1024 * 1024,
                         ping_timeout=60) as server:
            await server.serve_forever()

    asyncio.run(run())
