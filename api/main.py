from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from api.model import get_model_response

model_name = "Bóson Higgs detection"
version = "v1.0.0"

app = FastAPI()

# Input for data validation
class Input(BaseModel):
    zero: float
    one: float
    two: float
    three: float
    four: float
    five: float
    six: float
    seven: float
    eight: str
    nine: float
    ten: float
    eleven: float
    twelve: str
    thirteen: float
    fourteen: float
    fifteen: float
    sixteen: str
    seventeen: float
    eighteen: float
    nineteen: float
    twenty: str
    twentyone: float
    twentytwo: float
    twentythree: float
    twentyfour: float
    twentyfive: float
    twentysix: float
    twentyseven: float

# Ouput for data validation
class Output(BaseModel):
    proba_0: float
    proba_1: float
    recommended_threshold: float


@app.get('/health')
def service_health():
    """Return service health"""
    return {
        "ok"
    }


@app.post('/predict', response_model=Output)
def model_predict(input: Input):
    """Predict with input"""
    response = get_model_response(input)
    return response