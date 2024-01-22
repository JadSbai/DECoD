from missforest.miss_forest import MissForest
import pandas as pd
import numpy as np
from utils import compute_imputation_metrics
import seaborn as sns
import matplotlib.pyplot as plt


class MissForestImputer:
    def __init__(self, original_data, missing_data):
        self.imputer = MissForest()
        self.imputed_data = None
        self.missing_data = missing_data
        self.original_data = original_data

    def fit(self):
        imputed_data = self.imputer.fit_transform(self.missing_data)
        self.convert(imputed_data)

    def convert(self, imputed_data):
        self.imputed_data = pd.DataFrame(imputed_data, columns=self.missing_data.columns)

    def compute_error(self):
        metrics = compute_imputation_metrics(self.imputed_data, self.original_data)
        return metrics

    def plot_categorical_comparisons(self):
        """
        Plots count comparisons for each categorical column between original and imputed data.

        :param original_data: DataFrame with original values.
        :param imputed_data: DataFrame with imputed values.
        """
        sns.set(style='darkgrid')
        n_cols = self.original_data.shape[1]
        fig, axs = plt.subplots(n_cols, 1, figsize=(15, n_cols * 5))

        if n_cols == 1:
            axs = [axs]

        for ax, col in zip(axs, self.original_data.columns):
            # Create a combined DataFrame for original and imputed data
            combined_data = self.original_data.copy()
            combined_data[col + '_imputed'] = self.imputed_data[col]
            combined_data = combined_data.melt(value_vars=[col, col + '_imputed'],
                                               var_name='Dataset', value_name='Value')

            # Plot the countplot
            sns.countplot(x='Value', hue='Dataset', data=combined_data, ax=ax)
            ax.set_title(f"Count Comparison for {col}")

        plt.tight_layout()
        plt.show()


