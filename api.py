from pydantic import BaseModel,Field,computed_field
from typing import Annotated
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from main import Pipe
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("model.pkl")
    
class client(BaseModel):
    age : Annotated[int,Field(...,ge=0, le=120, description="Age must be between 0 and 120")]
    weight : Annotated[float,Field(...,ge=0, description="Weight must be a positive number")]
    height : Annotated[float,Field(...,ge=0, description="Height must be a positive number")]
    income_lpa : Annotated[float,Field(...,ge=0, description="Income must be a positive number")]
    city : Annotated[str,Field(...,description="City of residence")]
    occupation : Annotated[str,Field(...,description="Occupation of the client")]
    smoker : Annotated[bool,Field(...,description="Whether the client is a smoker or not")]

    @computed_field
    @property
    def lifestyle_risk(self) -> str:
        if self.smoker and self.bmi > 30:
            return "high"
        elif self.smoker or self.bmi > 27:
            return "medium"
        else:
            return "low"
        
    @computed_field
    @property
    def bmi(self) -> float:
        return self.weight/(self.height**2)
    
    @computed_field
    @property
    def age_group(self) -> str:
        if self.age < 18:
            return "child"
        elif self.age < 60:
            return "adult"
        else:        
            return "senior"

    @computed_field
    @property
    def city_tier(self) -> str:
        tier_1_cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"]
        tier_2_cities = [
            "Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi", "Visakhapatnam", "Coimbatore",
            "Bhopal", "Nagpur", "Vadodara", "Surat", "Rajkot", "Jodhpur", "Raipur", "Amritsar", "Varanasi",
            "Agra", "Dehradun", "Mysore", "Jabalpur", "Guwahati", "Thiruvananthapuram", "Ludhiana", "Nashik",
            "Allahabad", "Udaipur", "Aurangabad", "Hubli", "Belgaum", "Salem", "Vijayawada", "Tiruchirappalli",
            "Bhavnagar", "Gwalior", "Dhanbad", "Bareilly", "Aligarh", "Gaya", "Kozhikode", "Warangal",
            "Kolhapur", "Bilaspur", "Jalandhar", "Noida", "Guntur", "Asansol", "Siliguri"
        ]
        if self.city in tier_1_cities:
            return "Tier_1"
        elif self.city in tier_2_cities:
            return "Tier_2"
        else:   
            return "Tier_3"

@app.post("/predict")
def predict_insurance_premium(c: client):

    bmi = c.weight/(c.height**2)
    df = pd.DataFrame([{
        "income_lpa": c.income_lpa,
        "lifestyle_risk": c.lifestyle_risk,
        "age_group": c.age_group,
        "city_tier": c.city_tier,
        "occupation": c.occupation,
        "bmi": bmi
    }])

    prediction = model.predict(df)
    return JSONResponse({"insurance_premium_category": prediction[0]})

