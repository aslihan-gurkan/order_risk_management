from typing import Dict, Any
from pydantic import BaseModel
from datetime import datetime
from typing import Dict, Any, List
from pydantic import BaseModel


class PredictionLogResponse(BaseModel):
    id: int
    input_data: Dict[str, Any]
    risk_probability: float
    prediction: int
    risk_level: str
    recommended_action: str
    created_at: datetime

    class Config:
        from_attributes = True

class PredictionRequest(BaseModel):
    input_data: Dict[str, Any]


class PredictionResponse(BaseModel):
    risk_probability: float
    prediction: int
    risk_level: str
    recommended_action: str