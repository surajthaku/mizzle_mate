class CICDModel:
    def handle(self, message: str) -> str:
        
        if "frontend" in message:
            return "CI/CD pipeline 'frontend' created successfully!"
        return "CI/CD task completed."
