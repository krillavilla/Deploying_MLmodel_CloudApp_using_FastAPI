# Screenshots for Project Submission

This directory contains placeholder files for the screenshots required for project submission. Replace these placeholder files with actual screenshots as described below.

## Required Screenshots

1. **continuous_integration.png**: 
   - Take a screenshot of the GitHub Actions page showing that the CI workflow is passing.
   - Navigate to your GitHub repository, click on the "Actions" tab, and capture a successful workflow run.

2. **continuous_deployment.png**: 
   - Take a screenshot of your Render.com dashboard showing that CD is enabled.
   - Navigate to your Render.com dashboard, select your service, go to the "Settings" tab, and capture the "Auto-Deploy" section showing it's enabled.

3. **example.png**: 
   - Take a screenshot of the FastAPI docs showing the example request body.
   - Run your FastAPI application locally with `uvicorn main:app --reload`, navigate to http://127.0.0.1:8000/docs, and capture the expanded POST /predict endpoint showing the example request body.

4. **live_get.png**: 
   - Take a screenshot of your browser receiving the contents of the GET endpoint from your deployed API.
   - Navigate to your deployed API's URL (e.g., https://census-income-predictor.onrender.com/) and capture the response.

5. **live_post.png**: 
   - Take a screenshot showing the result of a POST request to your live API.
   - Run the `test_live_api.py` script and capture the output showing a successful POST request and response.

## Instructions

1. Replace each placeholder file with an actual screenshot as described above.
2. Make sure the screenshots are clear and show all the required information.
3. Remove the placeholder files from the .gitignore file once you've replaced them with actual screenshots.
4. Commit and push the screenshots to your GitHub repository.