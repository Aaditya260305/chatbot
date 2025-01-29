# api/routes.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from utils.model_manager import ModelManager  
from fastapi.middleware.cors import CORSMiddleware  

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

model_manager = ModelManager()

class Query(BaseModel):
    question: str

class TrainingData(BaseModel):
    documents: List[str]

class ApiKeyInput(BaseModel):  # Add this for API key input
    api_key: str

@app.post("/initialize")
async def initialize_chatbot(api_key_input: ApiKeyInput):
    try:
        model_manager.initialize_chatbot(api_key_input.api_key)
        return {"status": "success", "message": "Chatbot initialized and trained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_model(query: Query):
    try:
        print(query)
        chatbot = model_manager.get_chatbot()
        response = chatbot.query(query.question )
        return {"status": "success", "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))