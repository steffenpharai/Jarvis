"""``python -m server`` – run the FastAPI server standalone (no orchestrator)."""

import uvicorn
from config import settings

if __name__ == "__main__":
    ssl_kwargs = {}
    if settings.JARVIS_HTTPS_CERT and settings.JARVIS_HTTPS_KEY:
        ssl_kwargs["ssl_certfile"] = settings.JARVIS_HTTPS_CERT
        ssl_kwargs["ssl_keyfile"] = settings.JARVIS_HTTPS_KEY

    uvicorn.run(
        "server.app:app",
        host=settings.JARVIS_SERVE_HOST,
        port=settings.JARVIS_SERVE_PORT,
        log_level="info",
        **ssl_kwargs,
    )
