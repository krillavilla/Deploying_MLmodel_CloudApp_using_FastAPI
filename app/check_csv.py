import pandas as pd

# Load the data
data = pd.read_csv("data/census.csv")

# Print the column names
print("Column names:", data.columns.tolist())

# Print the first few rows
print("\nFirst few rows:")
print(data.head())