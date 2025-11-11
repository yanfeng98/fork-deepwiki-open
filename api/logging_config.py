import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler


class IgnoreLogChangeDetectedFilter(logging.Filter):
    def filter(self, record: logging.LogRecord):
        return "Detected file change in" not in record.getMessage()


def setup_logging(format: str = None):
    base_dir: Path = Path(__file__).parent
    log_dir: Path = base_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    default_log_file: Path = log_dir / "application.log"

    log_level_str: str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    log_file_path: Path = Path(os.environ.get("LOG_FILE_PATH", str(default_log_file)))

    log_dir_resolved = log_dir.resolve()
    resolved_path = log_file_path.resolve()
    if not str(resolved_path).startswith(str(log_dir_resolved) + os.sep):
        raise ValueError(f"LOG_FILE_PATH '{log_file_path}' is outside the trusted log directory '{log_dir_resolved}'")

    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        max_mb: int = int(os.environ.get("LOG_MAX_SIZE", 10))
        max_bytes: int = max_mb * 1024 * 1024
    except (TypeError, ValueError):
        max_bytes = 10 * 1024 * 1024

    try:
        backup_count: int = int(os.environ.get("LOG_BACKUP_COUNT", 5))
    except ValueError:
        backup_count: int = 5

    log_format: str = format or "%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s"

    file_handler = RotatingFileHandler(resolved_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    file_handler.addFilter(IgnoreLogChangeDetectedFilter())
    console_handler.addFilter(IgnoreLogChangeDetectedFilter())

    logging.basicConfig(level=log_level, handlers=[file_handler, console_handler], force=True)

    logger = logging.getLogger(__name__)
    logger.debug(
        f"Logging configured: level={log_level_str}, "
        f"file={resolved_path}, max_size={max_bytes} bytes, "
        f"backup_count={backup_count}"
    )
