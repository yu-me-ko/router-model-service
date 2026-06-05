from fastapi import FastAPI
from pydantic import BaseModel

from app.predictor import router_predictor


app = FastAPI(title="Router Model Service")


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {
        "message": "Router Model Service is running"
    }


@app.post("/route")
def route_question(req: QuestionRequest):
    return router_predictor.predict(req.question)