import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from interpret import show
import matplotlib.pyplot as plt
import seaborn as sns


class EBMImputer:
    def __init__(self, original_data, missing_data):
        self.imputed_data = missing_data
        self.missing_data = missing_data
        self.original_data = original_data
        self.model = None
        self.explanations = {}

    def initial_fill(self):
        for column in self.missing_data.columns:
            if self.missing_data[column].dtype == 'object':  # Categorical column
                self.imputed_data[column].fillna(self.missing_data[column].mode()[0], inplace=True)
            else:  # Numerical column
                self.imputed_data[column].fillna(self.missing_data[column].mean(), inplace=True)

    def impute_with_ebm(self, column):
        # Splitting into training and testing sets
        train_data = self.missing_data[self.missing_data[column].notnull()]
        test_data = self.missing_data[self.missing_data[column].isnull()]

        X_train = train_data.drop(column, axis=1)
        y_train = train_data[column]
        X_test = test_data.drop(column, axis=1)

        # Determining the model based on column type
        if pd.api.types.is_numeric_dtype(self.missing_data[column]):
            self.model = ExplainableBoostingRegressor()
        else:
            self.model = ExplainableBoostingClassifier()

        # Train the model and predict missing values
        self.model.fit(X_train, y_train)
        predicted_values = self.model.predict(X_test)

        # Impute predicted values into the dataset
        self.imputed_data.loc[self.missing_data[column].isnull(), column] = predicted_values

    def impute_with_ebm_iteratively(self, max_iter=10, tolerance=0.01):
        for col in self.missing_data.columns:
            if self.missing_data[col].isnull().sum() > 0:  # Check if the column has missing values
                prev_data = self.imputed_data.copy()
                for i in range(max_iter):
                    self.impute_with_ebm(col)

                    # Calculate sum of squared differences
                    diff = (prev_data[col] - self.imputed_data[col]) ** 2
                    sum_squared_diff = diff.sum()

                    # Check for convergence
                    if sum_squared_diff <= tolerance:
                        break
                    prev_data = self.imputed_data.copy()

                # Generate global explanation for the column
                self.explanations[col] = self.model.explain_global()

    def impute(self):
        self.initial_fill()
        self.impute_with_ebm_iteratively()

    def accuracy(self):
        return np.sum(np.abs(self.imputed_data[self.missing_data.isnull().any(axis=1)] - self.original_data[
            self.missing_data.isnull().any(axis=1)]))

    def visualise(self, col):
        sns.set(style='darkgrid')
        fig, ax = plt.plot(figsize=(15, 10))
        sns.kdeplot(x=self.missing_data[col][self.missing_data.isnull().any(axis=1)], label='Original')
        sns.kdeplot(x=self.imputed_data[col][self.missing_data.isnull().any(axis=1)], label='MissForest')
        ax.legend()
        fig.show()

    def get_explanations(self):
        for col, explanation in self.explanations.items():
            print(f"Explanations for imputing {col}:")
            show(explanation)
