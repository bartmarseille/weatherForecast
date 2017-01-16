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
import shutil
import tensorflow as tf
from dfutils import normalize, denormalize, circularize, decircularize, binarize
import csv

# pandas configuration
pd.set_option('precision',2)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
# pd.set_option('display.height', 1000)
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)

# global variables
datafile = 'data/KNMI_2000_hourly.csv'
modelfile = '/tmp/KNMI_2000_hourly_model'
normalizefile = 'data/KNMI_2000_hourly_normalizer'
csvfile = 'model/KNMI_2000_hourly_model.csv'

# source_weather_data is one weather fact of 20050403 0900h put in the desired column order
# source_weather_data = pd.DataFrame.from_dict(data ={'HH': [9], 'FH': [40], 'T': [140], 'TD': [58], 'SQ': [01], 'Q': [156], 'DR': [0], 'RH': [0], 'P': [10214], 'VV': [64], 'N': [0], 'U': [58], 'M': [0], 'R': [0], 'S': [0], 'O': [0], 'Y': [0]})
# columns = ['HH', 'FH', 'T', 'TD', 'SQ', 'Q', 'DR', 'RH', 'P', 'VV', 'N', 'U', 'M', 'R', 'S', 'O', 'Y']
source_weather_data = pd.DataFrame.from_dict(data ={'FH': [40], 'T': [140], 'Q': [156], 'DR': [0], 'RH': [0], 'P': [10214], 'U': [58]})
columns = ['FH', 'T', 'Q', 'DR', 'RH', 'P', 'U']
source_weather_data = source_weather_data[columns]

print "Source weather data:\n", source_weather_data
source_weather_data = normalize(source_weather_data, normalizefile, False)
# enrich data
# source_weather_data[['HH_sin', 'HH_cos']] = source_weather_data.apply(lambda row: pd.Series(circularize(row['HH'])), axis=1)
# source_weather_data.drop('HH', axis=1, inplace=True)

columns = source_weather_data.columns.values
print "Normalized source weather data:\n", source_weather_data

# the Graph
num_source_vertices = len(source_weather_data.columns)
num_target_vertices = len(source_weather_data.columns)
num_hidden_vertices = int(num_source_vertices + num_target_vertices)
# source (input) and target (output) data
source_data = tf.placeholder("float", [None, num_source_vertices])
target_data = tf.placeholder("float", [None, num_target_vertices])
# graph weight & bias initialization
weights1 = tf.Variable(tf.truncated_normal([num_source_vertices, num_hidden_vertices], mean=0.0, stddev=0.1, name="weights1"))
weights2 = tf.Variable(tf.truncated_normal([num_hidden_vertices, num_hidden_vertices], mean=0.0, stddev=0.1, name="weights2"))
weights3 = tf.Variable(tf.truncated_normal([num_hidden_vertices, num_target_vertices], mean=0.0, stddev=0.1, name="weights3"))
biases1 = tf.Variable(tf.truncated_normal([1, num_hidden_vertices], mean=0.0, stddev=0.1, name="biases1"))
biases2 = tf.Variable(tf.truncated_normal([1, num_hidden_vertices], mean=0.0, stddev=0.1, name="biases2"))
biases3 = tf.Variable(tf.truncated_normal([1, num_target_vertices], mean=0.0, stddev=0.1, name="biases3"))
# model initialization using RELU
# data(source => hidden1 => hidden2 =>sink)
hidden1_data = tf.nn.relu(tf.matmul(source_data, weights1) + biases1)
hidden2_data = tf.nn.relu(tf.matmul(hidden1_data, weights2) + biases2)
sink_data = (tf.matmul(hidden2_data, weights3) + biases3) #* SCALE_NUM_OUTPUTS

num_dream_iterations = 24
with tf.Session() as sess:
# Restore variables from disk.
    saver = tf.train.Saver()
    saver.restore(sess, modelfile)
    print "Model restored:\n", modelfile

    # keep track of actuals to save together
    actuals = source_weather_data.copy()
    # actuals['HH'] = actuals.apply(lambda row: pd.Series(decircularize(row['HH_sin'], row['HH_cos'])), axis=1)
    # actuals.drop('HH_sin', axis=1, inplace=True)
    # actuals.drop('HH_cos', axis=1, inplace=True)
    actuals = denormalize(actuals, normalizefile)

    # Run the restored trained model for [num_dream_iterations] iterations
    print "\nScoring model by dreaming for %s iterations (row 0 is real, initial weather data)" % num_dream_iterations
    for iter in xrange(0, num_dream_iterations):
        tf_sink = sess.run(sink_data, feed_dict = {source_data : source_weather_data.values})
        sink_weather_data = pd.DataFrame(tf_sink[0], columns, columns=[str(iter+1)]).transpose()
        # binarize some columns we know are binary
        # sink_weather_data['M'] = np.clip(np.around(sink_weather_data['M']), 0.0, 1.0)
        # sink_weather_data['R'] = np.clip(np.around(sink_weather_data['R']), 0.0, 1.0)
        # sink_weather_data['S'] = np.clip(np.around(sink_weather_data['S']), 0.0, 1.0)
        # sink_weather_data['O'] = np.clip(np.around(sink_weather_data['O']), 0.0, 1.0)
        # sink_weather_data['Y'] = np.clip(np.around(sink_weather_data['Y']), 0.0, 1.0)
        source_weather_data = sink_weather_data

        # sink_weather_data.set_index(['header', str(iter)])
        actual = source_weather_data.copy()
        # actual['HH'] = source_weather_data.apply(lambda row: pd.Series(decircularize(row['HH_sin'], row['HH_cos'])), axis=1)
        # actual.drop('HH_sin', axis=1, inplace=True)
        # actual.drop('HH_cos', axis=1, inplace=True)
        actual = denormalize(actual, normalizefile)
        actuals = actuals.append(actual)
    print actuals
    actuals.to_csv(csvfile) #, sep='\t', encoding='utf-8')
