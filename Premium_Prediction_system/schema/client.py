from pydantic import BaseModel,Field,computed_field,field_validator
from typing import Annotated
from config.city_tier import tier_1_cities, tier_2_cities

class Client(BaseModel):
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
    @field_validator("city")
    @classmethod
    def validate_city(cls, v):
        return v.strip().title()
    

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
        if self.city in tier_1_cities:
            return "Tier_1"
        elif self.city in tier_2_cities:
            return "Tier_2"
        else:   
            return "Tier_3"


