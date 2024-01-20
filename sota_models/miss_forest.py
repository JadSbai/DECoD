import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.neighbors._base
import sys
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest

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
        self.imputed_data = pd.DataFrame(imputed_data, columns=self.missing_data.columns).round(1)

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
