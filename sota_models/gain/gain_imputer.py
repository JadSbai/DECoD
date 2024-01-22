# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Main function for UCI letter and spam datasets.
'''

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import pandas as pd

from .gain import gain
from .gain_utils import rmse_loss


class GAINImputer:
    '''Main function for UCI letter and spam datasets.

    Args:
      - data_name: dataset name
      - batch:size: batch size
      - hint_rate: hint rate
      - alpha: hyperparameter
      - iterations: iterations

    Returns:
      - imputed_data_x: imputed data
      - rmse: Root Mean Squared Error
    '''

    def __init__(self, data_name, batch_size=128, hint_rate=0.9, alpha=100, iterations=10000):
        self.data_name = data_name
        self.gain_parameters = {'batch_size': batch_size,
                                'hint_rate': hint_rate,
                                'alpha': alpha,
                                'iterations': iterations}
        self.ori_data_x, self.miss_data_x, self.data_m = self.data_loader()
        print('original data: ', self.ori_data_x)
        print('missing data: ', self.miss_data_x)
        print('missing data mask: ', self.data_m)

    def impute(self):
        imputed_data_x = gain(self.miss_data_x, self.gain_parameters)
        print('imputed data: ', imputed_data_x)

    def compute_error(self):
        rmse = rmse_loss(self.ori_data_x, self.miss_data_x, self.data_m)
        print('RMSE: ' + str(np.round(rmse, 4)))
        return rmse

    def data_loader(self):
        # Load data
        file_name = 'datasets/new_' + self.data_name + '.csv'
        data_x = pd.read_csv(file_name)
        columns = data_x.select_dtypes(include=['object']).columns
        file_name = 'datasets/' + "missing_" + self.data_name + '.csv'
        miss_data_x = data_x.copy()
        for c in miss_data_x.columns:
            random_index = np.random.choice(miss_data_x.index, size=100)
            miss_data_x.loc[random_index, c] = np.nan
            # Load data

        for col in columns:
            # Factorize the original data
            data_x[col], uniques = pd.factorize(data_x[col])
            # Factorize the missing data using the same categories as original data
            miss_data_x[col] = pd.Categorical(miss_data_x[col], categories=uniques).codes

        # Convert dataframes to numpy arrays
        data_x_np = data_x.to_numpy()
        miss_data_x_np = miss_data_x.to_numpy()

        # Introduce missing data indicator
        data_m = (miss_data_x_np != -1).astype(int)

        return data_x_np, miss_data_x_np, data_m

