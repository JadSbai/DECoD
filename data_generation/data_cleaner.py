import pandas as pd
import numpy as np
import random


def remove_values_to_threshold(file_path, columns_to_modify, missing_percentage):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Iterate over each specified column
    for column in columns_to_modify:
        # Check if the column exists in the dataset
        if column in df.columns:
            # Calculate the number of values to remove
            total_values = len(df[column])
            values_to_remove = int(total_values * missing_percentage / 100)

            # Select random indices to remove
            indices_to_remove = random.sample(range(total_values), values_to_remove)

            # Set the selected values to NaN
            df.loc[indices_to_remove, column] = np.nan
        else:
            print(f"Column '{column}' not found in dataset.")

    # Save the modified dataset
    df.to_csv('modified_dataset.csv', index=False)
    print("Modified dataset saved as 'modified_dataset.csv'.")


def remove_columns_from_csv(file_path, columns_to_remove, output_file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Remove the specified columns and save the result in a new DataFrame
    new_df = df.drop(columns_to_remove, axis=1, errors='ignore')

    # Save the new DataFrame to a new CSV file
    new_df.to_csv(output_file_path, index=False)
    print(f"New dataset saved as '{output_file_path}'.")


def report_missing_values(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Calculate the percentage of missing values for each column
    missing_percentages = df.isnull().mean() * 100

    # Print a formatted report
    print("Percentage of Missing Values for Each Variable:")
    print("-" * 50)
    for column, missing_percentage in missing_percentages.items():
        print(f"{column}: {missing_percentage:.2f}% missing")

