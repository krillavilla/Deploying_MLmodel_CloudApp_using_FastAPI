import requests

# URL of the deployed API
# Replace this with the actual URL of your deployed API
# For Render.com, the URL will be something like:
# https://your-app-name.onrender.com
API_URL = "https://census-income-predictor.onrender.com"

# Sample data for prediction
sample_data = {
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

# Test the GET endpoint on root
def test_get_root():
    response = requests.get(f"{API_URL}/")
    print("GET Root Response Status Code:", response.status_code)
    print("GET Root Response Body:", response.json())
    return response.status_code == 200


# Test the GET endpoint on predict
def test_get_predict():
    response = requests.get(f"{API_URL}/predict")
    print("GET Predict Response Status Code:", response.status_code)
    print("GET Predict Response Body:", response.json())
    return response.status_code == 200 and "message" in response.json()


# Test the POST endpoint
def test_post():
    response = requests.post(f"{API_URL}/predict", json=sample_data)
    print("POST Response Status Code:", response.status_code)
    print("POST Response Body:", response.json())
    return response.status_code == 200


if __name__ == "__main__":
    print("Testing the deployed API...")

    # Test GET endpoint on root
    print("\nTesting GET endpoint on root...")
    get_root_success = test_get_root()

    # Test GET endpoint on predict
    print("\nTesting GET endpoint on predict...")
    get_predict_success = test_get_predict()

    # Test POST endpoint
    print("\nTesting POST endpoint...")
    post_success = test_post()

    # Print overall result
    if get_root_success and get_predict_success and post_success:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed.")
