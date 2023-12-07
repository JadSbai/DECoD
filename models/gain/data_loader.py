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
from utils import binary_sampler
from keras.datasets import mnist


def data_loader(data_name):
    '''Loads datasets and introduce missingness.

    Args:
      - data_name: letter, spam, or mnist
      - miss_rate: the probability of missing components

    Returns:
      data_x: original data
      miss_data_x: data with missing values
      data_m: indicator matrix for missing components
    '''

    # Load data

    file_name = 'datasets/' + data_name + '.csv'
    data_x = np.loadtxt(file_name, delimiter=",", skiprows=1)
    file_name = 'datasets/' + "missing_" + data_name + '.csv'
    miss_data_x = np.loadtxt(file_name, delimiter=",", skiprows=1)

    # Introduce missing data
    data_m = np.zeros(data_x.shape)

    # Put 1s where there are no missing values in the original dataset
    data_m[np.isnan(miss_data_x) == False] = 1

    return data_x, miss_data_x, data_m
