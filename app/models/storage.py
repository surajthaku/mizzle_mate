import re

class StorageModel:
    def handle(self, message: str) -> str:
        match = re.search(r'add\s+(\d+)\s*GB', message, re.IGNORECASE)
        if match:
            volume = match.group(1)
            return f"Added {volume}GB to your current storage volume."
        return "Storage updated successfully."
