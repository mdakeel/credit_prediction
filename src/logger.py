import logging
import os
from datetime import datetime

LOG_File = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

logs_path = os.path.join(os.getcwd(), "logs", LOG_File)

os.makedirs(logs_path, exist_ok=True)

LOGS_FILE_PATH = os.path.join(logs_path, LOG_File)

logging.basicConfig(
    filename = LOGS_FILE_PATH,
    level = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s "
)