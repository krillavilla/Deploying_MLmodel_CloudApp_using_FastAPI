import requests
import json

# URL of the deployed API
# Replace this with the actual URL of your deployed API
# For Render.com, the URL will be something like: https://your-app-name.onrender.com
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

# Test the GET endpoint
def test_get():
    response = requests.get(f"{API_URL}/")
    print("GET Response Status Code:", response.status_code)
    print("GET Response Body:", response.json())
    return response.status_code == 200

# Test the POST endpoint
def test_post():
    response = requests.post(f"{API_URL}/predict", json=sample_data)
    print("POST Response Status Code:", response.status_code)
    print("POST Response Body:", response.json())
    return response.status_code == 200

if __name__ == "__main__":
    print("Testing the deployed API...")

    # Test GET endpoint
    print("\nTesting GET endpoint...")
    get_success = test_get()

    # Test POST endpoint
    print("\nTesting POST endpoint...")
    post_success = test_post()

    # Print overall result
    if get_success and post_success:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed.")
