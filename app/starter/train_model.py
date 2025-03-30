import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

# Adjust these imports if your folder structure is different.
# E.g. if you have "starter.ml" or "ml" depends on how you structured it.
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference


def compute_slice_metrics(df, feature, model, encoder, lb, cat_features):
    """
    Compute performance metrics on slices of data for a categorical feature.
    Saves or returns them in a list so you can write them out to a file.

    df: the dataset (including label column).
    feature: (str) name of a categorical feature to slice on.
    model, encoder, lb: trained objects from training.
    cat_features: list of categorical feature names (same as training).
    """
    results = []
    unique_vals = df[feature].unique()

    for val in unique_vals:
        # Subset DataFrame
        df_temp = df[df[feature] == val]
        X_temp, y_temp, _, _ = process_data(
            df_temp,
            categorical_features=cat_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb
        )
        preds = inference(model, X_temp)
        precision, recall, fbeta = compute_model_metrics(y_temp, preds)
        line = (
            f"{feature}={val} -> "
            f"Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, "
            f"F1: {fbeta:.4f}"
        )
        results.append(line)
    return results


def main():
    # Load the data
    data = pd.read_csv("../data/census.csv")

    # Clean column names (strip whitespace)
    data.columns = data.columns.str.strip()

    # Train/test split
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    # Define your categorical features
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

    # Process the training data
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    # Process the test data
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary",
        training=False, encoder=encoder, lb=lb
    )

    # Train a model
    model = train_model(X_train, y_train)

    # Evaluate the model
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print(f"Overall test performance - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {fbeta:.4f}")

    # Compute slice-based metrics (example on 'education')
    slice_results = compute_slice_metrics(test, "education", model, encoder, lb, cat_features)
    with open("slice_output.txt", "w") as f:
        for line in slice_results:
            f.write(line + "\n")

    # Save the artifacts
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)
    with open("lb.pkl", "wb") as f:
        pickle.dump(lb, f)

    print("Training complete. Model artifacts saved, slice metrics written to slice_output.txt.")


if __name__ == "__main__":
    main()
