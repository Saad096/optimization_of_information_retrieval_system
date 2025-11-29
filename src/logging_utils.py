import logging
from pathlib import Path

from .config import load_config


def setup_logging():
    config = load_config()
    log_cfg = config.get("logging", {})
    log_file = log_cfg.get("log_file", "logs/ir_system.log")
    level_str = log_cfg.get("level", "INFO")

    level = getattr(logging, level_str.upper(), logging.INFO)

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    logger = logging.getLogger("IRSystem")
    logger.info("Logging initialized.")
    return logger
