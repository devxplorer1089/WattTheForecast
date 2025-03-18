import pandas as pd
import os

def refine_dataset(csv_path, time_columns=None, delimiter=";"):
    """
    Cleans and preprocesses the dataset by:
    - Identifying and converting datetime columns.
    - Managing missing values efficiently.
    - Sorting the dataset chronologically.

    Parameters:
        csv_path (str): Path to the CSV file.
        time_columns (list, optional): List of columns containing timestamps. Auto-detected if None.
        delimiter (str): Delimiter used in the CSV file.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    # Load the dataset
    df = pd.read_csv(csv_path, delimiter=delimiter, low_memory=False)

    # Detect timestamp columns if not explicitly provided
    if time_columns is None:
        time_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ["date", "time"])]

    # Convert detected datetime columns
    for col in time_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Log missing values before handling them
    missing_data = df[time_columns].isnull().sum()
    print(f"Reviewing missing timestamps in {csv_path}:\n{missing_data}")

    # Remove rows where all timestamp columns are NaN
    if time_columns:
        df.dropna(subset=time_columns, how="all", inplace=True)

    # Warn if dataset is empty post-cleaning
    if df.empty:
        print(f"Dataset {csv_path} has no usable data after processing.")

    # Sort by the earliest available timestamp column
    if time_columns and not df.empty:
        df.sort_values(by=time_columns[0], inplace=True, ignore_index=True)

    return df

# Define source and output directories
input_dir = "datasets/source"
output_dir = "datasets/refined"
os.makedirs(output_dir, exist_ok=True)

# Iterate over CSV files for processing
for filename in os.listdir(input_dir):
    if filename.endswith(".csv"):
        input_path = os.path.join(input_dir, filename)
        print(f"Starting cleanup for {filename}...")

        # Apply preprocessing
        refined_df = refine_dataset(input_path)

        if not refined_df.empty:  # Save only non-empty outputs
            output_path = os.path.join(output_dir, f"refined_{filename}")
            refined_df.to_csv(output_path, index=False, sep=",")
            print(f"Completed: Processed file saved at {output_path}\n")
        else:
            print(f"Skipped {filename} as the refined dataset is empty.\n")

print("Dataset refinement process completed.")
