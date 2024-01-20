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

'''Data loader for UCI letter, spam and MNIST datasets.
'''

# Necessary packages
import numpy as np
import pandas as pd


def data_loader(data_name):
    # Load data
    file_name = 'datasets/' + data_name + '.csv'
    data_x = pd.read_csv(file_name)
    columns = data_x.select_dtypes(include=['object']).columns
    file_name = 'datasets/' + "missing_" + data_name + '.csv'
    miss_data_x = pd.read_csv(file_name)

    for col in columns:
        # Factorize the original data
        data_x[col], _ = pd.factorize(data_x[col])
        # Factorize the missing data using the same categories as original data
        _, unique_categories = pd.factorize(miss_data_x[col])
        miss_data_x[col] = pd.Categorical(miss_data_x[col], categories=unique_categories).codes

    # Convert dataframes to numpy arrays
    data_x_np = data_x.to_numpy()
    miss_data_x_np = miss_data_x.to_numpy()

    print(miss_data_x_np)

    # Introduce missing data indicator
    data_m = np.zeros(data_x_np.shape)
    data_m[~np.isnan(miss_data_x_np)] = 1  # Mark non-missing values

    return data_x_np, miss_data_x_np, data_m
