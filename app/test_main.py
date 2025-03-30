from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "Welcome" in r.json()["greeting"]

def test_post_predict_greater_than_50k():
    sample = {
        "age": 42,
        "workclass": "Private",
        "fnlgt": 123456,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 4000,
        "capital-loss": 0,
        "hours-per-week": 60,
        "native-country": "United-States"
    }
    r = client.post("/predict", json=sample)
    assert r.status_code == 200
    assert r.json()["prediction"] in [">50K", "<=50K"]

def test_post_predict_less_than_equal_50k():
    sample = {
        "age": 21,
        "workclass": "Private",
        "fnlgt": 654321,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Sales",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 20,
        "native-country": "United-States"
    }
    r = client.post("/predict", json=sample)
    assert r.status_code == 200
    assert r.json()["prediction"] in [">50K", "<=50K"]
