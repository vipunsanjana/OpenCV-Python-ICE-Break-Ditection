# utils/logger.py
from pathlib import Path
import logging
import time

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

log_file = log_dir / f"ice_break_{time.strftime('%Y%m%d_%H%M%S')}.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode='w')
    ]
)

logger = logging.getLogger("ice_break_logger")  # create a named logger
logger.info("Logging setup complete. Logs will be saved to %s", log_file)
