"""Bridge between FastAPI WebSocket clients and the Jarvis orchestrator.

Holds a shared query queue (same one the orchestrator reads from) and a
broadcast set of connected WebSocket clients.  Thread-safe: the orchestrator
may run in its own thread while FastAPI runs on the async event loop.

Enhanced with:
  - Message sequence numbers for ordering (prevents WS race conditions)
  - Acknowledgement support (PWA can confirm receipt)
  - Rate limiting for broadcasts (prevents flood on slow networks)
"""

import asyncio
import json
import logging
import threading
import time
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
        # Message sequence counter for ordering
        self._seq: int = 0
        self._seq_lock = threading.Lock()
        # Rate limiting: track last broadcast time per message type
        self._last_broadcast: dict[str, float] = {}
        self._rate_limit_ms: dict[str, int] = {
            "detections": 500,
            "vitals": 1000,
            "threat": 1000,
            "thinking_step": 100,
        }

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

    def _next_seq(self) -> int:
        """Atomically increment and return the next sequence number."""
        with self._seq_lock:
            self._seq += 1
            return self._seq

    def _should_rate_limit(self, msg_type: str) -> bool:
        """Check if this message type should be rate-limited."""
        limit_ms = self._rate_limit_ms.get(msg_type, 0)
        if limit_ms <= 0:
            return False
        now = time.monotonic()
        last = self._last_broadcast.get(msg_type, 0)
        if (now - last) * 1000 < limit_ms:
            return True
        self._last_broadcast[msg_type] = now
        return False

    async def broadcast(self, data: dict) -> None:
        """Send JSON payload to every connected client with sequence number."""
        msg_type = data.get("type", "")
        if self._should_rate_limit(msg_type):
            return

        with self._clients_lock:
            targets = list(self._clients)
        if not targets:
            return
        # Inject sequence number for ordering on the client side
        data["_seq"] = self._next_seq()
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

    async def send_hologram(self, data: dict) -> None:
        await self.broadcast({"type": "hologram", "data": data})

    async def send_vitals(self, data: dict) -> None:
        await self.broadcast({"type": "vitals", "data": data})

    async def send_threat(self, data: dict) -> None:
        await self.broadcast({"type": "threat", "data": data})

    async def send_thinking_step(self, step: str, detail: str = "") -> None:
        """Broadcast an orchestration thinking step to all clients.

        The PWA renders these as a live activity feed so the user always
        sees what Jarvis is doing ("What is Jarvis thinking?").
        """
        await self.broadcast({
            "type": "thinking_step",
            "step": step,
            "detail": detail,
        })

    def send_thinking_step_threadsafe(self, step: str, detail: str = "") -> None:
        """Thread-safe variant for the wake/STT thread."""
        self.broadcast_threadsafe({
            "type": "thinking_step",
            "step": step,
            "detail": detail,
        })

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
            # Interrupt: clear the query queue so pending queries are dropped,
            # and notify all clients that the current action was interrupted.
            if self._query_queue is not None:
                while not self._query_queue.empty():
                    try:
                        self._query_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
            await self.broadcast({"type": "status", "status": "Listening"})
            await self.broadcast({"type": "reply", "text": "Very well, sir."})
            logger.debug("Client requested interrupt — queue cleared")

        elif msg_type == "sarcasm_toggle":
            enabled = msg.get("enabled", False)
            from tools import toggle_sarcasm
            result = toggle_sarcasm(enabled)
            await self.broadcast({"type": "reply", "text": result})

        elif msg_type == "hologram_request":
            await self._handle_hologram_request()

        elif msg_type == "vitals_request":
            await self._handle_vitals_request()

        elif msg_type == "ping":
            # Heartbeat response for connection health
            await self.broadcast({"type": "pong"})

        else:
            logger.debug("Unknown WS message type: %r", msg_type)

        # Send acknowledgement if client included a request_id
        req_id = msg.get("request_id")
        if req_id:
            await self.broadcast({"type": "ack", "request_id": req_id})

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

    async def _handle_hologram_request(self) -> None:
        """Generate hologram data and broadcast to all clients."""
        try:
            from tools import vision_analyze_full

            data = await asyncio.get_running_loop().run_in_executor(
                None, vision_analyze_full, None,
            )
            hologram_payload = {
                "point_cloud": data.get("point_cloud", [])[:3000],
                "tracked_objects": data.get("tracked", []),
                "description": data.get("description", ""),
            }
            await self.send_hologram(hologram_payload)
        except Exception as e:
            logger.warning("Hologram request failed: %s", e)
            await self.send_error(f"Hologram generation failed: {e}")

    async def _handle_vitals_request(self) -> None:
        """Return latest vitals snapshot via WS."""
        try:
            from vision.shared import get_vitals_analyzer

            analyzer = await asyncio.get_running_loop().run_in_executor(
                None, get_vitals_analyzer,
            )
            if analyzer is None:
                await self.send_vitals({
                    "fatigue": "unknown",
                    "posture": "unknown",
                    "heart_rate": None,
                    "hr_confidence": 0,
                    "alerts": [],
                })
                return

            result = analyzer.last_result
            await self.send_vitals({
                "fatigue": result.fatigue_level,
                "posture": result.posture_label,
                "heart_rate": result.heart_rate_bpm,
                "hr_confidence": result.heart_rate_confidence,
                "alerts": result.alerts,
            })
        except Exception as e:
            logger.warning("Vitals request failed: %s", e)


# Module-level singleton
bridge = Bridge()
