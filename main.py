import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
# Load the dataset
data = pd.read_csv('data.csv')
dff = data.copy()

def age_group(age):
    if age < 18:
        return "child"
    elif age < 60:
        return "adult"
    else:        
        return "senior"
    
def lifestyle_risk(row):
    if row["smoker"] and row["bmi"] > 30:
        return "high"
    elif row["smoker"] or row["bmi"] > 27:
        return "medium"
    else:
        return "low"

tier_1_cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"]

tier_2_cities = [
    "Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi", "Visakhapatnam", "Coimbatore",
    "Bhopal", "Nagpur", "Vadodara", "Surat", "Rajkot", "Jodhpur", "Raipur", "Amritsar", "Varanasi",
    "Agra", "Dehradun", "Mysore", "Jabalpur", "Guwahati", "Thiruvananthapuram", "Ludhiana", "Nashik",
    "Allahabad", "Udaipur", "Aurangabad", "Hubli", "Belgaum", "Salem", "Vijayawada", "Tiruchirappalli",
    "Bhavnagar", "Gwalior", "Dhanbad", "Bareilly", "Aligarh", "Gaya", "Kozhikode", "Warangal",
    "Kolhapur", "Bilaspur", "Jalandhar", "Noida", "Guntur", "Asansol", "Siliguri"
]

def categorise_city(city):
    if city in tier_1_cities:
        return "Tier_1"
    elif city in tier_2_cities:
        return "Tier_2"
    else:   
        return "Tier_3"
dff["bmi"] = dff["weight"]/(dff["height"]**2)
# print(dff.head())
dff["lifestyle_risk"] = dff.apply(lifestyle_risk, axis=1)
dff["age_group"] = dff["age"].apply(age_group)
dff["city_tier"] = dff["city"].apply(categorise_city)
dff.drop(columns = ["age","weight","height","smoker","city"],inplace = True)

x = dff[["income_lpa", "lifestyle_risk", "age_group", "city_tier","occupation","bmi"]]
y = dff["insurance_premium_category"]
# print(x.head())
preprocessing = ColumnTransformer(
    transformers=[("cat",OneHotEncoder(),["lifestyle_risk", "age_group", "city_tier","occupation"]),
                   ("num", "passthrough",["income_lpa","bmi"])]
)
Pipe = Pipeline(steps=[
    ("preprocessing", preprocessing),
    ("model", RandomForestClassifier(random_state=42))
])
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
Pipe.fit(X_train, y_train)
print(X_test.head())
y_pred = Pipe.predict(X_test)
print(accuracy_score(y_test, y_pred))

import pickle
pickle_model_path = "model.pkl"
with open(pickle_model_path, "wb") as file:
    pickle.dump(Pipe, file)