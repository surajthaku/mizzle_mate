

from fastapi import HTTPException
from app.exception import MizzleException
import joblib
import os,sys
from app.logger import logging



try:
    model_path = os.path.join("intent_model", "model.pkl")
    vectorizer_path = os.path.join("intent_model", "vectorizer.pkl")
    clf = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
except Exception as e:
    logging.error("Failed to load intent classifier model or vectorizer", exc_info=True)
    
    raise MizzleException(e,sys)

def classify_intent(message: str) -> str:
    try:
        vec = vectorizer.transform([message])
        intent = clf.predict(vec)[0]
        return intent
    except Exception as e:
        logging.error(f"Error classifying intent: {e}", exc_info=True)
        
        raise MizzleException(e,sys)
