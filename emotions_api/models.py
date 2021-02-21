
from typing import List, Dict, Optional
from pydantic import BaseModel


class Prediction(BaseModel):
    provider: str
    confidences: Optional[Dict[str, float]]
    predominant_emotion: Optional[str]
    error: Optional[str]


class PredictorResponse(BaseModel):
    provider_predictions: List[Prediction]
    predominant_emotion: Optional[str]
