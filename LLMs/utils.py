import ast
import pickle
import re
import pandas as pd
import numpy as np
import random
from sklearn.metrics import confusion_matrix, mean_squared_error, precision_score, recall_score, fbeta_score, \
    accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer


def remove_values_to_threshold(file_path, columns_to_modify, missing_percentage, output_file_path):
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
    df.to_csv(output_file_path, index=False)
    print(f"Missing dataset saved as {output_file_path}.")


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


def generate_weight_matrix(categories):
    num_categories = len(categories)
    weights = np.zeros((num_categories, num_categories))

    for i in range(num_categories):
        for j in range(num_categories):
            # Example: weight based on the absolute difference in category indices
            weights[i, j] = abs(i - j)
    return weights


def extract_values_from_text(llm_output):
    # Regex pattern to extract '"column_row": value'
    pattern = r"\{(?:'\w+_\d+': '\w+'(?:, )?)+\}"
    # Find all matches
    matches = re.findall(pattern, llm_output)

    if matches:
        # Evaluate the first match to a dictionary
        python_dict = ast.literal_eval(matches[0])
        return python_dict
    else:
        print("No dictionary pattern found in the string.")
        return {}


def retrieve_true_values(true_dataset, imputed_values):
    true_values = {}
    for tag in imputed_values.keys():
        col, row = tag.split('_')
        true_value = true_dataset.at[int(row), col]
        true_values[tag] = true_value
    return true_values


def compute_categorical_error(true_values, imputed_values, categories):

    print("True Values: ", true_values)
    print("Imputed Values: ", imputed_values)
    print("Categories: ", categories)
    # # Create confusion matrix
    # cm = confusion_matrix(true_values, imputed_values, labels=categories)
    #
    # # Compute weighted error
    # weights = generate_weight_matrix(categories)
    # weighted_error = np.sum(cm * weights) / np.sum(cm)

    # Calculate Precision, Recall, F1 Score, and Matching Percentage (Accuracy)
    precision = precision_score(true_values, imputed_values, average='weighted', labels=['Yes', 'No'])
    recall = recall_score(true_values, imputed_values, average='weighted', labels=['Yes', 'No'])

    # true_values and predicted_values should be defined as your actual data
    fbeta = fbeta_score(true_values, imputed_values, beta=2, average='weighted', labels=['Yes', 'No'])
    mp = accuracy_score(true_values, imputed_values)

    # Return a dictionary of all the metrics
    return {
        # "Weighted Error": weighted_error,
        "Precision": precision,
        "Recall": recall,
        "Fβ Score (β=2):": fbeta,
        "Matching Percentage": mp,
    }


def compute_numerical_error(true_values, imputed_values):
    mse = mean_squared_error(true_values, imputed_values)
    print(f"Mean Squared Error: {mse}")
    return mse


def get_imputed_dataset(dataframe, imputed_values):
    for tag, value in imputed_values.items():
        col, row = tag.split('_')
        dataframe.at[int(row), col] = value
    return dataframe


def save_llm_output(output, filename):
    with open(f'outputs/{filename}.pkl', 'wb') as file:
        pickle.dump(output, file)


def load_llm_output(file_name):
    with open(f'outputs/{file_name}.pkl', 'rb') as file:
        output = pickle.load(file)
    return output


def get_corresponding_true_values(imputed_values, true_subset):
    # New code to print true values corresponding to imputed values
    true_values_corresponding = {}
    for key in imputed_values.keys():
        # Extract column name and row index from the key
        col, row_index = key.split('_')
        row_index = int(row_index)  # Convert row index to integer

        # Fetch the true value from the true_subset DataFrame
        if col in true_subset.columns:
            true_value = true_subset.at[row_index, col]
            true_values_corresponding[key] = true_value

    return true_values_corresponding
