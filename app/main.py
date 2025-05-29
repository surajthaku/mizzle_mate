from fastapi import FastAPI, Request, HTTPException
from app.router import route_message
from app.logger import log_interaction,logging
from app.schemas import ChatInput
from app.exception import MizzleException
import sys
import uvicorn


app = FastAPI(title="Mizzle Mate - DevOps Chatbot")




@app.post("/chat")
async def chat_handler(input: ChatInput):
    
    try:
        user_id = input.user_id
        message = input.message

        response = route_message(user_id, message)

        
        log_interaction(user_id, message, response)

        return response

    except Exception as e:
        logging.error(f"Error processing chat message: {e}")
        raise MizzleException(e,sys)


@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
