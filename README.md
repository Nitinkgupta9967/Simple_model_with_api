# Insurance Premium Category Prediction API

This project trains a machine learning model to classify insurance premium category (`Low`, `Medium`, `High`) and exposes it through a FastAPI endpoint.

## Project Structure

- `main.py` - trains a `RandomForestClassifier` pipeline and saves the model as `model.pkl`
- `api.py` - FastAPI app that loads `model.pkl` and serves predictions at `POST /predict`
- `data.csv` - training dataset

## How It Works

The training script creates engineered features:

- `bmi` from `weight / (height^2)`
- `lifestyle_risk` from smoker status and BMI
- `age_group` from age (`child`, `adult`, `senior`)
- `city_tier` from city (`Tier_1`, `Tier_2`, `Tier_3`)

Model input features used for prediction:

- `income_lpa`
- `lifestyle_risk`
- `age_group`
- `city_tier`
- `occupation`
- `bmi`

## Requirements

Install dependencies:

```bash
pip install fastapi uvicorn pandas scikit-learn joblib pydantic matplotlib numpy
```

## Run Locally

1. Train and save the model:

```bash
python main.py
```

This generates `model.pkl` in the project root.

2. Start the API server:

```bash
uvicorn api:app --reload
```

3. Open API docs:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## API Usage

### Endpoint

- `POST /predict`

### Request Body

```json
{
  "age": 35,
  "weight": 78.5,
  "height": 1.75,
  "income_lpa": 12.0,
  "city": "Delhi",
  "occupation": "private_job",
  "smoker": false
}
```

### Example cURL

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"age\":35,\"weight\":78.5,\"height\":1.75,\"income_lpa\":12.0,\"city\":\"Delhi\",\"occupation\":\"private_job\",\"smoker\":false}"
```

### Sample Response

```json
{
  "insurance_premium_category": "Medium"
}
```

## Notes

- Ensure `model.pkl` exists before starting the API.
- Input validation is handled through Pydantic in `api.py`.
- City tier logic is currently hard-coded in both training and API scripts.
