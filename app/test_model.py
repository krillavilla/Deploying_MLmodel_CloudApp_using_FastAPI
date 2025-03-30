import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference


def test_process_data():
    # Construct path to data file that works regardless of where test is run from
    data_path = os.path.join(os.path.dirname(__file__), "data", "census.csv")
    df = pd.read_csv(data_path)
    # Clean column names (strip whitespace)
    df.columns = df.columns.str.strip()
    train, _ = train_test_split(df, test_size=0.2, random_state=42)
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=[
            "workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country"
        ],
        label="salary",
        training=True
    )
    assert X_train.shape[0] == y_train.shape[0]
    assert X_train.shape[0] > 0


def test_train_model():
    X = np.array([[0, 1], [1, 1], [1, 0]])
    y = np.array([0, 1, 1])
    model = train_model(X, y)
    assert model is not None


def test_compute_model_metrics():
    y = np.array([0, 1, 1])
    preds = np.array([0, 1, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision is not None
    assert recall is not None
    assert fbeta is not None


def test_inference():
    X = np.array([[0, 1], [1, 1]])
    y = np.array([0, 1])
    model = train_model(X, y)
    preds = inference(model, X)
    assert len(preds) == 2
