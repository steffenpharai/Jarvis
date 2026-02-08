"""Bridge between FastAPI WebSocket clients and the Jarvis orchestrator.

Holds a shared query queue (same one the orchestrator reads from) and a
broadcast set of connected WebSocket clients.  Thread-safe: the orchestrator
may run in its own thread while FastAPI runs on the async event loop.
"""

import asyncio
import json
import logging
import threading
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class Bridge:
    """Glue between WebSocket clients and the orchestrator loop."""

    def __init__(self) -> None:
        # asyncio.Queue shared with the orchestrator (text queries)
        self._query_queue: asyncio.Queue[str] | None = None
        # Connected WebSocket clients
        self._clients: set[WebSocket] = set()
        self._clients_lock = threading.Lock()
        # Event loop reference (set once at startup)
        self._loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def set_query_queue(self, q: asyncio.Queue[str]) -> None:
        self._query_queue = q

    @property
    def query_queue(self) -> asyncio.Queue[str]:
        if self._query_queue is None:
            raise RuntimeError("Bridge.query_queue not set; call set_query_queue first")
        return self._query_queue

    # ------------------------------------------------------------------
    # Client management
    # ------------------------------------------------------------------

    def add_client(self, ws: WebSocket) -> None:
        with self._clients_lock:
            self._clients.add(ws)
        logger.info("WS client connected (%d total)", len(self._clients))

    def remove_client(self, ws: WebSocket) -> None:
        with self._clients_lock:
            self._clients.discard(ws)
        logger.info("WS client disconnected (%d total)", len(self._clients))

    # ------------------------------------------------------------------
    # Broadcasting (server → all clients)
    # ------------------------------------------------------------------

    async def _send_json(self, ws: WebSocket, data: dict) -> None:
        try:
            await ws.send_json(data)
        except Exception:
            self.remove_client(ws)

    async def broadcast(self, data: dict) -> None:
        """Send JSON payload to every connected client."""
        with self._clients_lock:
            targets = list(self._clients)
        if not targets:
            return
        await asyncio.gather(*(self._send_json(ws, data) for ws in targets))

    def broadcast_threadsafe(self, data: dict) -> None:
        """Call from any thread to broadcast to all clients."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(asyncio.ensure_future, self.broadcast(data))

    # Convenience helpers for common message types -----------------------

    async def send_status(self, status: str) -> None:
        await self.broadcast({"type": "status", "status": status})

    def send_status_threadsafe(self, status: str) -> None:
        self.broadcast_threadsafe({"type": "status", "status": status})

    async def send_reply(self, text: str) -> None:
        await self.broadcast({"type": "reply", "text": text})

    async def send_transcript(self, text: str, final: bool = True) -> None:
        msg_type = "transcript_final" if final else "transcript_interim"
        await self.broadcast({"type": msg_type, "text": text})

    async def send_detections(self, detections: list[dict], description: str = "") -> None:
        await self.broadcast({"type": "detections", "detections": detections, "description": description})

    async def send_error(self, message: str) -> None:
        await self.broadcast({"type": "error", "message": message})

    async def send_wake(self) -> None:
        await self.broadcast({"type": "wake"})

    async def send_proactive(self, text: str) -> None:
        await self.broadcast({"type": "proactive", "text": text})

    # ------------------------------------------------------------------
    # Inbound: client → orchestrator
    # ------------------------------------------------------------------

    async def inject_text(self, text: str) -> None:
        """Put a user text query onto the orchestrator's queue."""
        await self.query_queue.put(text)
        logger.debug("Injected text into query queue: %r", text[:80])

    async def handle_client_message(self, raw: str | bytes) -> None:
        """Parse and dispatch a JSON message from a WS client."""
        try:
            msg: dict[str, Any] = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Invalid WS message: %r", raw[:120] if isinstance(raw, (str, bytes)) else raw)
            return

        msg_type = msg.get("type") or msg.get("command") or ""

        if msg_type == "text":
            text = (msg.get("text") or "").strip()
            if text:
                await self.inject_text(text)

        elif msg_type in ("start_listening", "stop_listening"):
            # Forward to orchestrator status if needed (currently informational)
            logger.debug("Client command: %s", msg_type)

        elif msg_type == "scan":
            # Trigger vision_analyze and send detections back
            await self._handle_scan()

        elif msg_type == "get_status":
            await self._handle_get_status()

        elif msg_type == "interrupt":
            logger.debug("Client requested interrupt (not yet implemented)")

        elif msg_type == "sarcasm_toggle":
            enabled = msg.get("enabled", False)
            from tools import toggle_sarcasm
            result = toggle_sarcasm(enabled)
            await self.broadcast({"type": "reply", "text": result})

        else:
            logger.debug("Unknown WS message type: %r", msg_type)

    async def _handle_scan(self) -> None:
        """Run vision_analyze tool and broadcast detections + description."""
        from tools import run_tool
        description = await asyncio.get_running_loop().run_in_executor(
            None, run_tool, "vision_analyze", {}
        )
        await self.broadcast({
            "type": "scan_result",
            "description": description,
        })

    async def _handle_get_status(self) -> None:
        """Return Jetson system status via WS."""
        from tools import run_tool
        status = await asyncio.get_running_loop().run_in_executor(
            None, run_tool, "get_jetson_status", {}
        )
        await self.broadcast({"type": "system_status", "status": status})


# Module-level singleton
bridge = Bridge()
