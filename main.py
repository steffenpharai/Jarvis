#!/usr/bin/env python3
"""Jarvis – offline voice assistant on Jetson Orin Nano. Entry: parse args, init modules, run main loop."""

import argparse
import logging
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

from config import settings
from config.prompts import JARVIS_SYSTEM_PROMPT
from utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Jarvis – offline voice assistant (Jetson Orin Nano)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--dry-run", action="store_true", help="Only validate config and exit")
    parser.add_argument(
        "--e2e",
        action="store_true",
        help="Run full loop: wake → STT → LLM → TTS (real hardware)",
    )
    parser.add_argument(
        "--voice-only",
        action="store_true",
        help="Phase 1 test: wake → play TTS 'At your service, sir' (no STT/LLM)",
    )
    parser.add_argument(
        "--test-audio",
        action="store_true",
        help="List audio devices and default sink/source, then exit",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Show status overlay (Listening / Thinking / Speaking)",
    )
    parser.add_argument(
        "--one-shot",
        metavar="PROMPT",
        nargs="?",
        const="What time is it?",
        default=None,
        help="E2E test without mic: text → LLM → TTS → play (default prompt: What time is it?)",
    )
    parser.add_argument(
        "--yolo-visualize",
        action="store_true",
        help="Live camera + YOLOE-26N detections in OpenCV window (press q to quit).",
    )
    parser.add_argument(
        "--orchestrator",
        action="store_true",
        help="Run agentic orchestrator: wake → STT → LLM with tools + context → TTS.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run FastAPI server + orchestrator: exposes WebSocket bridge, REST API, and MJPEG stream for the PWA.",
    )
    parser.add_argument(
        "--portable",
        action="store_true",
        help="Portable/walk-around mode: lower res (320x320), 10 FPS, frame skipping for battery/thermal.",
    )
    return parser.parse_args()


def _set_gui_status(status: str) -> None:
    try:
        from gui.overlay import set_status

        set_status(status)
    except Exception:
        pass


def _handle_voice_only():
    """Phase 1: on wake, play a canned TTS 'At your service, sir.'"""
    from audio.output import play_wav
    from voice.tts import synthesize
    from voice.wakeword import run_wake_loop

    def on_wake():
        _set_gui_status("Speaking")
        wav = synthesize("At your service, sir.", voice=settings.TTS_VOICE)
        if wav:
            play_wav(wav)
        else:
            logger.warning("TTS failed, skipping playback")
        _set_gui_status("Listening")

    logger.info("Voice-only test: say wake word to hear 'At your service, sir.'")
    _set_gui_status("Listening")
    stop = run_wake_loop(on_wake)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop.set()


def _handle_one_shot(prompt: str):
    """E2E without mic: prompt → LLM → TTS → play. For testing without camera/BT."""
    from audio.output import play_wav
    from llm.context import build_messages
    from llm.ollama_client import chat, is_ollama_available, is_ollama_model_available
    from utils.reminders import format_reminders_for_llm, load_reminders
    from voice.tts import synthesize

    reminders_path = Path(settings.PROJECT_ROOT) / "data"
    reminders_path.mkdir(parents=True, exist_ok=True)
    reminders = load_reminders(reminders_path)
    rem_text = format_reminders_for_llm(reminders)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        from utils.power import get_system_stats

        sys_stats = get_system_stats()
    except Exception:
        sys_stats = None
    system = JARVIS_SYSTEM_PROMPT
    if settings.SARCASM_ENABLED:
        system += " Sarcasm mode is on; you may be dry and slightly sarcastic."
    messages = build_messages(
        system,
        prompt.strip(),
        vision_description=None,
        reminders_text=rem_text,
        current_time=current_time,
        system_stats=sys_stats,
    )
    logger.info("One-shot E2E: prompt=%r", prompt.strip())
    if not is_ollama_available(settings.OLLAMA_BASE_URL):
        logger.error("Ollama not available at %s. Start: bash scripts/start-ollama.sh", settings.OLLAMA_BASE_URL)
        return
    if not is_ollama_model_available(settings.OLLAMA_BASE_URL, settings.OLLAMA_MODEL):
        logger.error(
            "Ollama model %r not pulled. Run: ollama pull %s",
            settings.OLLAMA_MODEL,
            settings.OLLAMA_MODEL,
        )
        return
    _set_gui_status("Thinking (LLM)")
    reply = chat(
        settings.OLLAMA_BASE_URL,
        settings.OLLAMA_MODEL,
        messages,
        stream=False,
        num_ctx=settings.OLLAMA_NUM_CTX,
    )
    if not (reply and reply.strip()):
        reply = "I'm afraid I have nothing to report on that, sir."
    _set_gui_status("Speaking")
    wav = synthesize(reply.strip(), voice=settings.TTS_VOICE)
    if wav:
        play_wav(wav)
        logger.info("One-shot E2E: played TTS")
    else:
        logger.warning("One-shot E2E: TTS failed")
    _set_gui_status("Idle")


def _handle_e2e():
    """Full loop: wake → record → STT → LLM → TTS → play; always with vision."""
    from audio.input import get_default_input_index, record_to_file
    from audio.output import play_wav
    from llm.context import build_messages
    from llm.ollama_client import chat, is_ollama_available, is_ollama_model_available
    from utils.reminders import format_reminders_for_llm, load_reminders
    from voice.stt import transcribe
    from voice.tts import synthesize
    from voice.wakeword import run_wake_loop

    vision_description: str | None = None
    vision_lock = threading.Lock()

    def vision_thread_fn():
        from vision.detector_mediapipe import detect_faces
        from vision.scene import describe_scene
        from vision.shared import get_face_detector, get_yolo, read_frame, run_inference_shared

        # Trigger lazy init so we fail-fast with a clear message
        engine, class_names = get_yolo()
        if not engine:
            logger.error("Vision thread: YOLOE engine not loaded")
            return
        face_det = get_face_detector()
        frame_count = 0
        while True:
            frame = read_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            yolo_dets = run_inference_shared(frame)
            faces = detect_faces(face_det, frame) if face_det else []
            desc = describe_scene(yolo_dets, face_count=len(faces), class_names=class_names)
            with vision_lock:
                nonlocal vision_description
                vision_description = desc
            frame_count += 1
            if frame_count % 15 == 0 and frame is not None:
                try:
                    import cv2
                    from vision.visualize import draw_detections_on_frame

                    preview_path = settings.JARVIS_PREVIEW_PATH
                    vis_frame = frame.copy()
                    draw_detections_on_frame(vis_frame, yolo_dets, class_names=class_names)
                    small = cv2.resize(vis_frame, (320, 180))
                    cv2.imwrite(preview_path, small)
                    try:
                        from gui.overlay import set_latest_frame_path

                        set_latest_frame_path(preview_path)
                    except Exception:
                        pass
                except Exception:
                    pass
            time.sleep(0.2)

    if not settings.yolo_engine_exists():
        logger.error(
            "Vision requires YOLOE engine at %s. Run: bash scripts/export_yolo_engine.sh",
            settings.YOLOE_ENGINE_PATH,
        )
        raise SystemExit(
            "Missing models/yoloe26n.engine. Build it with: bash scripts/export_yolo_engine.sh"
        )
    vt = threading.Thread(target=vision_thread_fn, daemon=True)
    vt.start()

    reminders_path = Path(settings.PROJECT_ROOT) / "data"
    reminders_path.mkdir(parents=True, exist_ok=True)

    def on_wake():
        _set_gui_status("Listening (recording)")
        device_index = get_default_input_index()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        try:
            if not record_to_file(
                wav_path,
                duration_sec=settings.RECORD_DURATION_SEC,
                device_index=device_index,
            ):
                logger.warning("Recording failed (check mic / BT HFP or use USB mic)")
                _set_gui_status("Listening")
                return
            _set_gui_status("Thinking (STT)")
            text = transcribe(wav_path, model_size=settings.STT_MODEL_SIZE)
            Path(wav_path).unlink(missing_ok=True)
            if not (text and text.strip()):
                reply = "My apologies, sir. I didn't quite catch that."
            else:
                with vision_lock:
                    v_desc = vision_description
                reminders = load_reminders(reminders_path)
                rem_text = format_reminders_for_llm(reminders)
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                try:
                    from utils.power import get_system_stats

                    sys_stats = get_system_stats()
                except Exception:
                    sys_stats = None
                system = JARVIS_SYSTEM_PROMPT
                if settings.SARCASM_ENABLED:
                    system += " Sarcasm mode is on; you may be dry and slightly sarcastic."
                messages = build_messages(
                    system,
                    text.strip(),
                    vision_description=v_desc,
                    reminders_text=rem_text,
                    current_time=current_time,
                    system_stats=sys_stats,
                )
                if not is_ollama_available(settings.OLLAMA_BASE_URL):
                    reply = "I'm afraid my language systems are offline, sir. The service requires a restart."
                elif not is_ollama_model_available(settings.OLLAMA_BASE_URL, settings.OLLAMA_MODEL):
                    reply = "I'm afraid the required language model hasn't been installed yet, sir."
                else:
                    _set_gui_status("Thinking (LLM)")
                    reply = ""
                    try:
                        reply = chat(
                            settings.OLLAMA_BASE_URL,
                            settings.OLLAMA_MODEL,
                            messages,
                            stream=False,
                            num_ctx=settings.OLLAMA_NUM_CTX,
                        )
                    except Exception as e:
                        if (
                            "memory" in str(e).lower()
                            or "oom" in str(e).lower()
                            or "resource" in str(e).lower()
                        ) and settings.OLLAMA_FALLBACK_MODEL != settings.OLLAMA_MODEL:
                            try:
                                reply = chat(
                                    settings.OLLAMA_BASE_URL,
                                    settings.OLLAMA_FALLBACK_MODEL,
                                    messages,
                                    stream=False,
                                    num_ctx=settings.OLLAMA_NUM_CTX,
                                )
                                logger.info("Used fallback model after OOM")
                            except Exception as e2:
                                logger.exception("Fallback model also failed: %s", e2)
                                reply = "I appear to be overtaxed at the moment, sir. Might I suggest a lighter configuration."
                        else:
                            if "memory" in str(e).lower() or "oom" in str(e).lower():
                                logger.warning("OOM during LLM; try a smaller model")
                            reply = "I've encountered a slight complication, sir. Shall we try again, perhaps with a lighter configuration."
                            logger.exception("Ollama error: %s", e)
                    if not (reply and reply.strip()):
                        reply = "I'm afraid I have nothing of substance to offer on that, sir."
            _set_gui_status("Speaking")
            wav_out = synthesize(reply.strip(), voice=settings.TTS_VOICE)
            if wav_out:
                play_wav(wav_out)
            else:
                logger.warning("TTS failed for reply")
        except Exception as e:
            logger.exception("E2E round failed: %s", e)
        finally:
            Path(wav_path).unlink(missing_ok=True)
            _set_gui_status("Listening")

    try:
        from utils.power import get_thermal_warning

        thermal = get_thermal_warning()
        if thermal:
            logger.warning("Thermal: %s", thermal)
    except Exception:
        pass
    logger.info("E2E: full loop (vision=on). Say wake word to ask.")
    stop = run_wake_loop(on_wake)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop.set()


def _handle_serve():
    """Run FastAPI (uvicorn) + orchestrator in one process.

    The orchestrator runs as an asyncio task alongside the ASGI server.
    The bridge shares the orchestrator's query_queue so WebSocket clients
    can inject text and receive status/reply broadcasts.

    Vision is always enabled.  Status updates are propagated to both the
    GUI overlay and all WebSocket clients via a combined callback.
    """
    import asyncio

    import uvicorn
    from orchestrator import _gui_status, run_orchestrator
    from server.app import app
    from server.bridge import bridge

    Path(settings.DATA_DIR).mkdir(parents=True, exist_ok=True)

    def _combined_status(status: str) -> None:
        """Forward status to GUI overlay *and* WebSocket clients."""
        _gui_status(status)
        bridge.send_status_threadsafe(status)

    async def _run_all():
        loop = asyncio.get_running_loop()
        bridge.set_loop(loop)

        # Shared query queue between orchestrator and bridge
        query_queue: asyncio.Queue = asyncio.Queue()
        bridge.set_query_queue(query_queue)

        # Uvicorn config (run as an asyncio server inside our loop)
        ssl_kwargs = {}
        if settings.JARVIS_HTTPS_CERT and settings.JARVIS_HTTPS_KEY:
            ssl_kwargs["ssl_certfile"] = settings.JARVIS_HTTPS_CERT
            ssl_kwargs["ssl_keyfile"] = settings.JARVIS_HTTPS_KEY

        config = uvicorn.Config(
            app,
            host=settings.JARVIS_SERVE_HOST,
            port=settings.JARVIS_SERVE_PORT,
            log_level="info",
            **ssl_kwargs,
        )
        server = uvicorn.Server(config)

        # Run orchestrator and uvicorn concurrently
        orch_task = asyncio.create_task(
            run_orchestrator(
                query_queue=query_queue,
                bridge=bridge,
                status_callback=_combined_status,
            )
        )
        server_task = asyncio.create_task(server.serve())

        _combined_status("Listening")
        logger.info(
            "Jarvis serving on %s:%s (orchestrator + API + WS)",
            settings.JARVIS_SERVE_HOST,
            settings.JARVIS_SERVE_PORT,
        )
        try:
            await asyncio.gather(server_task, orch_task)
        except (KeyboardInterrupt, asyncio.CancelledError):
            server.should_exit = True
            orch_task.cancel()

    asyncio.run(_run_all())


def main() -> int:
    args = parse_args()
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    # Portable mode: override camera settings for low-power walk-around
    if args.portable:
        settings.PORTABLE_MODE = True
        settings.CAMERA_WIDTH = settings.PORTABLE_WIDTH
        settings.CAMERA_HEIGHT = settings.PORTABLE_HEIGHT
        settings.CAMERA_FPS = settings.PORTABLE_FPS
        logger.info(
            "Portable mode: %dx%d @ %d FPS, depth skip=%d, vitals skip=%d",
            settings.PORTABLE_WIDTH, settings.PORTABLE_HEIGHT,
            settings.PORTABLE_FPS, settings.PORTABLE_DEPTH_SKIP,
            settings.PORTABLE_VITALS_SKIP,
        )

    if args.dry_run:
        logger.info(
            "Dry run: config OK, project_root=%s, ollama=%s, model=%s",
            settings.PROJECT_ROOT,
            settings.OLLAMA_BASE_URL,
            settings.OLLAMA_MODEL,
        )
        if settings.yolo_engine_exists():
            logger.info("YOLOE engine present: %s", settings.YOLOE_ENGINE_PATH)
        else:
            logger.warning(
                "YOLOE engine missing: %s (required for --e2e / --serve)",
                settings.YOLOE_ENGINE_PATH,
            )
        return 0

    if args.test_audio:
        from audio import bluetooth
        from audio import input as audio_input

        print("Input devices:", audio_input.list_input_devices())
        print("Default input index:", audio_input.get_default_input_index())
        print("Default sink:", bluetooth.get_default_sink_name())
        print("Default source:", bluetooth.get_default_source_name())
        return 0

    if args.gui:
        from gui.overlay import run_overlay

        gui_thread = threading.Thread(target=run_overlay, daemon=True)
        gui_thread.start()

    if args.voice_only:
        _handle_voice_only()
        return 0

    if args.one_shot is not None:
        _handle_one_shot(args.one_shot)
        return 0

    if args.yolo_visualize:
        from vision.visualize import run_live_visualization

        run_live_visualization()
        return 0

    if args.e2e:
        from vision.shared import check_cuda

        ok, msg = check_cuda()
        if not ok:
            logger.error("CUDA required for vision: %s", msg)
            return 1
        _set_gui_status("Listening")
        _handle_e2e()
        return 0

    if args.orchestrator:
        import asyncio

        from orchestrator import run_orchestrator

        Path(settings.DATA_DIR).mkdir(parents=True, exist_ok=True)
        _set_gui_status("Listening")
        asyncio.run(run_orchestrator())
        return 0

    if args.serve:
        from vision.shared import check_cuda

        ok, msg = check_cuda()
        if not ok:
            logger.error("CUDA required for vision: %s", msg)
            return 1
        logger.info("Starting Jarvis server with all subsystems...")
        _handle_serve()
        return 0

    logger.info(
        "Jarvis idle. Use --one-shot [PROMPT], --voice-only, --e2e (full loop), --orchestrator, --serve, or --test-audio."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
