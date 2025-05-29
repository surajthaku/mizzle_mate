class MonitoringModel:
    def handle(self, message: str) -> str:
        if "cpu" in message.lower():
            return "Current CPU usage is at 42%."
        elif "memory" in message.lower():
            return "Memory usage is at 65%."
        return "Monitoring data is available."
