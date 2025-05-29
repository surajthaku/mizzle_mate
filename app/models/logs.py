import re
from datetime import datetime, timedelta

class LogsModel:
    def handle(self, message: str) -> str:
        if "yesterday" in message.lower():
            day = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            return f"Showing logs for {day}."
        elif "today" in message.lower():
            day = datetime.now().strftime("%Y-%m-%d")
            return f"Showing logs for {day}."
        return "Showing recent logs."
