import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
file_name = 'datasets/penang_reviews.json'
df = pd.read_json(file_name)

# Define the split ratios
# Training: 60%, Validation: 20%, Test: 20%
TEST_SIZE = 0.2
VAL_SIZE = 0.2
RANDOM_STATE = 42 # Set for reproducibility

# --- Step 1: Split data into Training+Validation (80%) and Test (20%) ---
df_train_val, df_test = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

# --- Step 2: Split Training+Validation (80%) into Training (60% of total) and Validation (20% of total) ---
# Calculate the validation size relative to the train_val set: 0.2 / 0.8 = 0.25
val_relative_size = VAL_SIZE / (1 - TEST_SIZE)

df_train, df_val = train_test_split(
    df_train_val,
    test_size=val_relative_size,
    random_state=RANDOM_STATE
)

# Output the size of each split
print(f"Original dataset size: {len(df)}")
print(f"Training set size: {len(df_train)} ({len(df_train)/len(df):.2f})")
print(f"Validation set size: {len(df_val)} ({len(df_val)/len(df):.2f})")
print(f"Test set size: {len(df_test)} ({len(df_test)/len(df):.2f})")

# Save the split datasets to JSON files
# 'orient="records"' is a common format for saving a list of JSON objects
df_train.to_json("datasets/train.json", orient="records", indent=2)
df_val.to_json("datasets/valid.json", orient="records", indent=2)
df_test.to_json("datasets/test.json", orient="records", indent=2)

print("\nDatasets saved to: train.json, valid.json, test.json")