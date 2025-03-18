import pandas as pd
import numpy as np
import os

def transform_dataset(csv_path):
    """
    Transforms the dataset by:
    - Handling missing values.
    - Encoding categorical features.
    - Scaling numerical attributes.

    Parameters:
        csv_path (str): Path to the cleaned dataset.

    Returns:
        pd.DataFrame: Transformed dataset.
    """
    # Load the dataset
    df = pd.read_csv(csv_path, delimiter=",", low_memory=False)

    # Identify numerical and categorical columns
    numeric_features = df.select_dtypes(include=[np.number]).columns
    categorical_features = df.select_dtypes(exclude=[np.number]).columns

    print(f"Processing dataset: {csv_path}")
    print(f"Detected numerical features: {list(numeric_features)}")
    print(f"Detected categorical features: {list(categorical_features)}\n")

    # Fill missing values
    for feature in numeric_features:
        df[feature] = df[feature].fillna(df[feature].median())  # Replace with median

    for feature in categorical_features:
        if not df[feature].mode().empty:  # Ensure mode exists
            df[feature] = df[feature].fillna(df[feature].mode()[0])  # Replace with mode
        else:
            print(f"Warning: Unable to determine mode for {feature}, leaving NaN values.")

    # Normalize numerical data using Min-Max Scaling
    for feature in numeric_features:
        df[feature] = (df[feature] - df[feature].min()) / (df[feature].max() - df[feature].min())

    return df

# Define input and output directories
source_dir = "datasets/refined"
target_dir = "datasets/optimized"
os.makedirs(target_dir, exist_ok=True)

# Apply transformation to each dataset
for filename in os.listdir(source_dir):
    file_path = os.path.join(source_dir, filename)

    if filename.endswith(".csv"):
        try:
            modified_df = transform_dataset(file_path)
            output_path = os.path.join(target_dir, f"optimized_{filename}")
            modified_df.to_csv(output_path, index=False, sep=",")
            print(f"File processed and saved: {output_path}\n")
        except Exception as err:
            print(f"Error encountered while processing {filename}: {err}\n")

print("Dataset transformation completed.")
