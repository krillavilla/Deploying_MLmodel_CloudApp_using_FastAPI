# Model Card

## Model Details
**Model Name**: Census Income Classifier  
**Model Version**: 1.0  
**Model Type**: Random Forest Classifier  
**Authors**: ML DevOps Engineer  
**Contact**: mldevops@example.com  

## Intended Use
Predict whether an individual's income exceeds $50K/year based on demographic data.  

## Training Data
Source: [UCI Machine Learning Repository's Adult Census dataset](https://archive.ics.uci.edu/ml/datasets/adult).  
Number of instances: ~32,000  
Number of features: 15  

The dataset contains census data with features such as:
- age
- workclass
- education
- marital-status
- occupation
- relationship
- race
- sex
- capital-gain
- capital-loss
- hours-per-week
- native-country

## Evaluation Data
- The data was split 80/20 for train/test.  
- No additional preprocessing was done beyond one-hot encoding categorical features and label binarization.

## Metrics
- Precision, Recall, F1 score are used to evaluate the model.
- Performance metrics will be updated after model training.
- Slice-based metrics are computed for categorical features to assess model fairness.

## Limitations
- The model is trained on census data from the 1990s, which may not reflect current demographic distributions.
- Potential biases in the underlying census data.  
- Not suitable for high-stakes decision making (e.g., credit scoring) without additional fairness testing.
- The model does not account for intersectional effects between features.

## Ethical Considerations
- The model may exhibit bias against certain demographic groups.
- Performance disparities across different demographic groups should be carefully monitored.
- Potential for disparate impact.  
- Must comply with relevant regulations (e.g., GDPR, anti-discrimination laws).  

## Caveats and Recommendations
- This model should be periodically retrained to maintain accuracy.  
- Before production use, confirm performance across demographic subgroups.
- Consider using more advanced fairness-aware algorithms for sensitive applications.
- Regularly monitor model performance in production to detect drift or degradation.