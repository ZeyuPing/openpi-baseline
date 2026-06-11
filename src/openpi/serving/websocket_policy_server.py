import asyncio
import http
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._infer_lock = asyncio.Lock()
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            ping_interval=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))
        logger.info("Sent metadata to %s: %s", websocket.remote_address, self._metadata)

        prev_total_time = None
        request_count = 0
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())
                request_count += 1
                images = obs.get("images", {}) if isinstance(obs, dict) else {}
                image_shapes = {
                    key: getattr(value, "shape", None)
                    for key, value in images.items()
                } if isinstance(images, dict) else {}
                logger.info(
                    "Request #%d from %s: image_shapes=%s state_shape=%s prompt=%r",
                    request_count,
                    websocket.remote_address,
                    image_shapes,
                    getattr(obs.get("state"), "shape", None) if isinstance(obs, dict) else None,
                    obs.get("prompt") if isinstance(obs, dict) else None,
                )

                infer_time = time.monotonic()
                async with self._infer_lock:
                    action = await asyncio.to_thread(self._policy.infer, obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time
                actions = action.get("actions") if isinstance(action, dict) else None
                logger.info(
                    "Response #%d to %s: actions_shape=%s actions_dtype=%s infer_ms=%.1f total_ms=%.1f",
                    request_count,
                    websocket.remote_address,
                    getattr(actions, "shape", None),
                    getattr(actions, "dtype", None),
                    infer_time * 1000,
                    prev_total_time * 1000,
                )

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                logger.exception("Request #%d from %s failed", request_count, websocket.remote_address)
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
