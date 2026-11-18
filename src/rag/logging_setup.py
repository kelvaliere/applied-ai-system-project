"""One place that configures logging: console + rotating file at logs/app.log."""
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
_configured = False


def setup(level: int = logging.INFO) -> None:
    """Idempotent. Never log secrets (API keys are read only inside generate.py and never formatted into messages)."""
    global _configured
    if _configured:
        return
    LOG_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    root = logging.getLogger("rag")
    root.setLevel(level)
    for handler in (logging.StreamHandler(), RotatingFileHandler(LOG_DIR / "app.log", maxBytes=1_000_000, backupCount=3)):
        handler.setFormatter(fmt)
        root.addHandler(handler)
    _configured = True
