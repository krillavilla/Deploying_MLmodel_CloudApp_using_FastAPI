# Instructions for Running Tests and Using the Application

This document provides instructions on how to set up the environment, train the model, run the API locally, run the tests, and deploy the API.

## Setting Up the Environment

1. Install the required packages:
   ```bash
   pip install -r app/requirements.txt
   ```

2. Navigate to the app directory:
   ```bash
   cd app
   ```

## Training the Model

Before running the API or tests, you need to train the model and generate the model artifacts:

1. Make sure you have the cleaned census data in `starter/data/census.csv`
2. Run the training script:
   ```bash
   python starter/train_model.py
   ```
   This will:
   - Train the model on the census data
   - Save the model artifacts (model.pkl, encoder.pkl, lb.pkl) in the current directory
   - Generate slice metrics in slice_output.txt

## Running the API Locally

1. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

2. Access the API documentation at http://127.0.0.1:8000/docs
   - You can test the GET endpoint by visiting http://127.0.0.1:8000/
   - You can test the POST endpoint using the Swagger UI at http://127.0.0.1:8000/docs

## Running the Tests

### Model Tests

Run the model tests to verify the model functionality:

```bash
pytest test_model.py -v
```

### API Tests

Run the API tests to verify the API functionality:

```bash
pytest test_main.py -v
```

### Sanity Check

Run the sanity check to ensure your test cases meet the requirements:

```bash
python sanitycheck.py
```
When prompted, enter the path to your test file: `test_main.py`

### Live API Tests

To test the deployed API:

1. Update the `API_URL` in `test_live_api.py` to point to your deployed API
2. Run the live API test:
   ```bash
   python test_live_api.py
   ```

## Deploying the API

### Render.com Deployment

1. Create a free Render.com account at https://render.com/
2. Make sure your repository contains the `render.yaml` file at the root
3. Connect your GitHub repository to Render.com:
   - Go to the Render.com dashboard
   - Click "New" and select "Blueprint"
   - Connect your GitHub account if you haven't already
   - Select your repository
   - Render will automatically detect the `render.yaml` file and set up the services

4. Push your code to GitHub:
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

5. Once deployed, you can test your live API using the `test_live_api.py` script (after updating the API_URL)

### Heroku Deployment

1. Create a free Heroku account
2. Install the Heroku CLI
3. Login to Heroku:
   ```bash
   heroku login
   ```
4. Create a new Heroku app:
   ```bash
   heroku create your-app-name
   ```
5. Set up your GitHub repository for deployment:
   - Go to the Heroku dashboard
   - Select your app
   - Go to the Deploy tab
   - Connect to your GitHub repository
   - Enable automatic deployments from the main branch
   - Check "Wait for CI to pass before deploy"

6. Push your code to GitHub:
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

7. Once deployed, you can test your live API using the `test_live_api.py` script (after updating the API_URL).

## Troubleshooting

- If you encounter path-related issues, make sure you're running commands from the correct directory.
- If model artifacts are not found, make sure you've run the training script first.
- If tests fail, check that your environment has all the required dependencies installed.
