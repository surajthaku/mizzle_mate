import json
from datetime import datetime

import logging
import os

from from_root import from_root
from datetime import datetime

LOG_PATH = "logs/interactions.log"

def log_interaction(user_id: str, message: str, response: dict):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_id": user_id,
        "message": message,
        "response": response
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(log_entry) + "\n")




LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

log_dir = 'logs'

logs_path = os.path.join(from_root(), log_dir, LOG_FILE)

os.makedirs(log_dir, exist_ok=True)


logging.basicConfig(
    filename=logs_path,
    format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
)