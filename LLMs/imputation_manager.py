from LLMs.utils import extract_values_from_text, \
    compute_categorical_error, compute_numerical_error, get_imputed_dataset, \
    get_corresponding_true_values


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
        self.missing_subset = self.missing_data.sample(n=10)
        self.true_subset = self.data.loc[self.missing_subset.index]

    def get_clinician_prior(self):
        return self.prior

    def get_prompt(self):
        clinician_prior = self.get_clinician_prior()

        prompt = (
            f"I have a dataset described as follows: {self.description}. Within this dataset, "
            "there's a subset where a specific pattern, known as the 'clinician prior', is observed. "
            f"Here is a subset of my medical dataset:\n{self.missing_subset.to_string()}\n\n"
            "The missing values are tagged with their respective column names and row indices."
            f"The clinician prior is the following: {clinician_prior}. Your task is to impute missing values in the dataset, "
            f"using the clinician prior as a guiding principle. However, you should also consider the rest of the dataset to derive your own insights and identify patterns."
            f"The imputed values should reflect the insights and patterns of the data itself and those highlighted by the clinician prior.\n\n"
            "Please focus on the values tagged as 'MISSING_...' and return the imputed values in a dictionary format"
            "where each key-value pair corresponds to a missing value. The key should be a string combining the column name and row index "
            "(e.g., 'Age_5'), and the value should be the imputed value for that cell. Please do not write any code. Simply return the values."
            "After returning the imputed values, explain your reasoning behind the imputed values and what patterns you identified."
            "Imputed values and explanations: [MASK]"
        )

        alternative_prompt = (
            "Task: Impute the missing values in the mental health dataset below. Use the provided clinician's insights to guide your imputations and explain your reasoning for each imputation."
            f"Clinician's Prior: {clinician_prior}"
            f"Dataset: {self.missing_subset.to_string()}"
            "return the imputed values in a dictionary format"
            "where each key-value pair corresponds to a missing value. The key should be a string combining the column name and row index "
            "(e.g., 'Age_5'), and the value should be the imputed value for that cell. Please do not write any code. Simply return the values."
            "Imputed values and explanations: [MASK]"
        )
        return alternative_prompt

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
        # self.imputed_data = get_imputed_dataset(self.missing_subset, imputed_values)
        imputed_list = [imputed_values[key] for key in imputed_values]
        true_list = [true[key] for key in imputed_values]
        errors = compute_categorical_error(true_list, imputed_list, self.category_dict)
        # for col in self.missing_columns:
        #     if col in self.categorical_cols:
        #         errors[col] = compute_categorical_error(self.true_subset[col], self.imputed_data[col], self.category_dict[col])
        #     elif col in self.numerical_cols:
        #         errors[col] = compute_numerical_error(self.true_subset[col], self.imputed_data[col])
        print(errors)


