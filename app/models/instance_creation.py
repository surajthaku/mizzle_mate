import re

class InstanceCreationModel:
    def handle(self, message: str) -> str:
        # Simple regex to extract resources
        ram_match = re.search(r'(\d+GB|\d+ GB)', message)
        cpu_match = re.search(r'(\d+)\s*CPU', message, re.IGNORECASE)

        ram = ram_match.group(1) if ram_match else "2GB"
        cpu = cpu_match.group(1) if cpu_match else "1"

        return f"Instance launched with {ram} RAM and {cpu} CPUs."
