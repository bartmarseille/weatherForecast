# encoding: UTF-8
# Copyright 2016 Bart Marseille
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import numpy as np
import pickle

pd.set_option('precision',4)
pd.set_option('expand_frame_repr', False)

# def initDB(name):
#     global __DBNAME__  # add this line!
#     if __DBNAME__ is None: # explicit test for None
#         __DBNAME__ = name
#     else:
#         raise RuntimeError("Database name has already been set.")

# normalize data with optional argument to calculate from passed dataframe
def normalize(df, filename, calculate = False):
    if calculate:
        print 'normalizing data:'
        norm = pd.DataFrame(df[:2].copy(), index=['min','max']).transpose()
        for col in df.columns.values:
            norm.loc[col,'min'] = df[col].min()
            norm.loc[col,'max'] = df[col].max()
        with open(filename, 'wb') as f:
            pickle.dump(norm, f)
        print norm.transpose(), '\n'

    result= df.copy()
    with open(filename, 'rb') as f:
        norm = pickle.load(f)
        for col in df.columns.values:
            result[col] = (df[col] - norm.loc[col,'min']) / (norm.loc[col,'max'] - norm.loc[col,'min'])
    return result

# denormalize data
def denormalize(df, filename):
    result= df.copy()
    with open(filename, 'rb') as f:
        norm = pickle.load(f)
        for col in df.columns.values:
        	result[col] = (df[col] * (norm.loc[col,'max'] - norm.loc[col,'min'])) + norm.loc[col,'min']
    return result

# circularize normalized data
def circularize(data):
    data_sin = np.sin(data * np.pi * 2)
    data_cos = np.cos(data * np.pi * 2)
    return data_sin, data_cos

# decircularize normalized data
def decircularize(data_sin, data_cos):
    return np.arctan2(data_sin, data_cos) / (np.pi * 2)

# rounds floats to whole numbers and clips them to range [0, 1]
def binarize(data):
    return np.clip(np.around(data), 0, 1)
