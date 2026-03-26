import joblib
import pandas as pd

model = joblib.load("Model/model.pkl")
class_labels = model.classes_.tolist()

def predict_output(Client : dict):
    input_df = pd.DataFrame(Client)

    output = model.predict(input_df)[0]
    predicted_classes = model.predict_proba(input_df)[0]
    confidence = max(predicted_classes)
    class_probs = dict(zip(class_labels,map(lambda x: round(x, 4), predicted_classes)))
    return {"predicted_category": output, "confidence": round(confidence, 4), "class_probabilities": class_probs}