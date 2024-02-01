from utils import extract_values_from_text, \
    compute_imputation_metrics, get_corresponding_true_values, get_imputed_dataset
import model_ids
import pandas as pd
        
def transform_row(row, category_dict, missing_columns):
    formatted_row = []
    for col in row.index:
        if col in missing_columns and pd.isna(row[col]):
            value = 'NaN'  # Replace missing values with 'NaN'
        else:
            value = row[col]

        # For categorical columns, check if value is in the categories list
        if col in category_dict and value not in ['NaN', None]:
            if value in category_dict[col]:
                value = value  # Use the value as it is (it's a valid category)
            else:
                value = 'Unknown'  # Value not found in the categories

        # Format and append the column info
        formatted_row.append(f"{col}: {value}")

    return '[' + ', '.join(formatted_row) + ']'



class ImputationManager:
    def __init__(self, dataset, missing_dataset, dataset_description, clinician_prior):
        self.data = dataset
        self.description = dataset_description
        self.prior = clinician_prior
        self.missing_data = missing_dataset
        self.imputed_data = None
        self.missing_columns = missing_dataset.columns[missing_dataset.isnull().any()].tolist()
        self.categorical_cols = dataset.select_dtypes(include=['object', 'category']).columns
        self.numerical_cols = dataset.select_dtypes(include=['number']).columns
        self.category_dict = {col: dataset[col].dropna().unique().tolist() for col in self.categorical_cols}  # Extract unique categories for each categorical column
        self.tag_missing_values()
        self.missing_subset = self.missing_data.sample(n=20)
        self.true_subset = self.data.loc[self.missing_subset.index]
        self.transformed_rows = self.missing_subset.apply(
            transform_row, axis=1, 
            category_dict=self.category_dict, 
            missing_columns=self.missing_columns
        ).tolist()




    def get_clinician_prior(self):
        return self.prior

    def get_prompt(self, model_id):
        clinician_prior = self.get_clinician_prior()

        if model_id == model_ids.tableLlama:
            prompt = (
                "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."

                "### Instruction:"
                "Task: Impute the missing values in the mental health dataset below. Use the provided clinician's insights to guide your imputations"
                f"Clinician's Prior: {clinician_prior}"
                "In the dataset that you can find below, focus on the values tagged as 'MISSING_Suicide_rownumber' and return the imputed values in a dictionary format"
                "where each key-value pair corresponds to a missing value. The key should be a string combining the column name and row index "
                "and the value should be the imputed value for that cell. The variable in question is whether the person has had suicidal ideation or not. So the only possible vaues it can take are Yes and No. Simply return the values."

                "### Input:"
                f"{self.missing_subset.to_string()}"

                "### Question:"
                f"What are the appropriate values for the missing values shown in the above dataset?"

                "### Response:"
            )
        

        elif model_id == model_ids.jellyfish:
            prompt = (
            "You are presented with mental health records for which some of them are missing values for a specific attribute: Suicide."
            "Your task is to deduce or infer the value of Suicide using the available information amongst the different records."
            "You may be provided with fields like Gender, Academic level, Religion and Depression Severity to help you in the inference."
            f"Records: {self.transformed_rows}"
            "Based on the provided records, what would you infer is the value for the missing attribute Suicide for each record?"
            "Give your answer in the following format: [{Record 1: Value1}, {Record2: Value2}...]. Report your answer only for the records where the Suicide value was unknown."
            )

        elif model_id == model_ids.mistral:
            prompt = (
            f"<s>[INST]Task: I am going to give you a mental helath dataset. {self.description}. "
            f"Dataset: {self.transformed_rows}"
            " "
            "I want you to look carefully at the dataset and try to deduce correlations between the various variables and derive semantical rules that you can then use later to then impute missing values."
            "Once you have a good understanding of the data and how variables correlate, I want you to infer the value for the missing attribute Suicide for each record?."
            "Give your answer in the following format: [{Record 1: Value1}, {Record2: Value2}...]. Report your answer only for the records where the Suicide value was unknown. [/INST]"
            )
        
        else:
            prompt = (
            "Task: Impute the missing values in the mental health dataset below. Use the provided clinician's insights to guide your imputations"
            f"Clinician's Prior: {clinician_prior}"
            f"Dataset: {self.missing_subset.to_string()}"
            "Please focus on the values tagged as 'MISSING_...' and return the imputed values in a dictionary format"
            "where each key-value pair corresponds to a missing value. The key should be a string combining the column name and row index "
            "and the value should be the imputed value for that cell. Simply return the values."
            )


        long_prompt = (
            f"[INST]I have a dataset described as follows: {self.description}. Within this dataset, "
            "there's a subset where a specific pattern, known as the 'clinician prior', is observed. "
            f"Here is a subset of my medical dataset:\n{self.missing_subset.to_string()}\n\n"
            "The missing values are tagged with their respective column names and row indices."
            f"The clinician prior is the following: {clinician_prior}. Your task is to impute missing values in the dataset, "
            f"using the clinician prior as a guiding principle. However, you should also consider the rest of the dataset to derive your own insights and identify patterns."
            f"The imputed values should reflect the insights and patterns of the data itself and those highlighted by the clinician prior.\n\n"
            "Please focus on the values tagged as 'MISSING_...' and return the imputed values in a dictionary format"
            "where each key-value pair corresponds to a missing value. The key should be a string combining the column name and row index "
            "and the value should be the imputed value for that cell. Please do not write any code. Simply return the values."
            "After returning the imputed values, explain your reasoning behind the imputed values and what patterns you identified."
            "Imputed values and explanations: [/INST]"
        )

        return prompt

    def tag_missing_values(self):
        for col in self.missing_data.columns:
            # Identify missing indices
            missing_indices = self.missing_data[col].isna()
            # Iterate over the Series
            for idx, is_missing in enumerate(missing_indices):
                if is_missing:
                    # Tag the missing value
                    self.missing_data.at[idx, col] = f"MISSING_{col}_{idx}"

    def compute_error(self, llm_output):
        imputed_values = extract_values_from_text(llm_output)
        print("Imputed: ", imputed_values)
        true = get_corresponding_true_values(imputed_values, self.true_subset)
        print("True: ", true)
        self.imputed_data = get_imputed_dataset(self.missing_subset, imputed_values)
        errors = compute_imputation_metrics(self.true_subset, self.imputed_data)
        print(errors)


