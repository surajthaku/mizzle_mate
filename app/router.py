from app.classifier import classify_intent
from app.config import MODULE_REGISTRY
from app.exception import MizzleException
import logging,sys

from app.logger import logging

_loaded_modules = {}

def route_message(user_id: str, message: str) -> dict:

   
    try:
        intent = classify_intent(message)

        if intent not in MODULE_REGISTRY:
            return {
                "reply": "Sorry, I don't understand that request.",
                "module": "unknown",
                "status": "error",
                "http_status": 400
            }

        
        if intent not in _loaded_modules:
            _loaded_modules[intent] = MODULE_REGISTRY[intent]()

        module_handler = _loaded_modules[intent]

        result = module_handler.handle(message)

        return {
            "reply": result,
            "module": intent,
            "status": "success",
            "http_status": 200
        }

    except Exception as e:
        logging.error(f"Error routing message for user {user_id}: {e}", exc_info=True)
        
        
        return {
            "reply": "An internal error occurred while processing your request.",
            "module": "internal_error",
            "status": "error",
            "http_status": 500
        }
        
