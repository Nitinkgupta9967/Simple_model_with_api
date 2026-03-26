from pydantic import BaseModel,Field,computed_field,field_validator
from typing import Annotated

class PredictionResponse(BaseModel):
    insurance_premium_category: str
    confidence: Annotated[float, Field(..., ge=0.0, le=1.0)]
    class_probabilities: dict[str, float]