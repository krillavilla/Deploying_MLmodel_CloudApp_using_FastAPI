import os

# If on Heroku (DYNO env variable) or Render.com (RENDER env variable) and there's a .dvc folder, pull data.
# This snippet is adapted from dvc_on_heroku_instructions.md
if ("DYNO" in os.environ or "RENDER" in os.environ) and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Adjust if your files are in a different path
from starter.ml.data import process_data
from starter.ml.model import inference

# Create the FastAPI application
app = FastAPI(
    title="Census Income Prediction API",
    description="Predicts whether income is <=50K or >50K based on Census data.",
    version="1.0.0",
)

# Load artifacts (model, encoder, lb) from disk
with open("starter/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("starter/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
with open("starter/lb.pkl", "rb") as f:
    lb = pickle.load(f)

# The same categorical features used in training
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

class CensusData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    class Config:
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States"
            }
        }

@app.get("/")
def read_root():
    """
    A simple GET endpoint returning a welcome message.
    """
    return {"greeting": "Welcome to the Census Income Prediction API!"}

@app.post("/predict")
def predict(data: CensusData):
    """
    POST endpoint to predict whether income is <=50K or >50K.
    """
    # Convert request body to dict, then DataFrame
    raw_data = data.dict(by_alias=True)
    df = pd.DataFrame([raw_data])

    # Process using the same process_data function
    X, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Inference
    preds = inference(model, X)
    # Convert 0/1 predictions back to string labels (<=50K or >50K)
    prediction_str = lb.inverse_transform(preds)[0]
    # Strip any leading/trailing whitespace
    prediction_str = prediction_str.strip()

    return {"prediction": prediction_str}
