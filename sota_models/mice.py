import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import miceforest as mf

"""Use the MICE algorithm to impute missing values. Uses the LightGBM model under the hood."""


class MiceImputer:
    def __init__(self, original_data, missing_data):
        self.imputer = mf.ImputationKernel(
            data=missing_data,
            save_all_iterations=True,
            random_state=1343
        )
        self.imputed_data = None
        self.missing_data = missing_data
        self.original_data = original_data

    def fit(self):
        self.imputer.mice(3, verbose=True)

    def impute(self):
        self.fit()
        self.imputed_data = self.imputer.complete_data(dataset=0, inplace=False)

    def accuracy(self):
        return np.sum(np.abs(self.imputed_data[self.missing_data.isnull().any(axis=1)] - self.original_data[
            self.missing_data.isnull().any(axis=1)]))

    def visualise(self, col):
        sns.set(style='darkgrid')
        fig, ax = plt.plot(figsize=(15, 10))
        sns.kdeplot(x=self.missing_data[col][self.missing_data.isnull().any(axis=1)], label='Original')
        sns.kdeplot(x=self.imputed_data[col][self.missing_data.isnull().any(axis=1)], label='MissForest')
        ax.legend()
        plt.show()
