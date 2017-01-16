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
import tensorflow as tf
from dfutils import normalize, circularize
import source_dataset
import visualtf
import shutil
import datetime as dt

# pandas configuration
pd.set_option('precision',4)
pd.set_option('expand_frame_repr', False)
# pd.set_option('display.height', 1000)
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)

# global variables
# datafile = 'KNMI_2000_hourly.csv'
modelfile = '/tmp/KNMI_2000_hourly_model'
modeltxtfile = 'model/waetherForecatsModel%s.txt' % dt.datetime.today().strftime('%Y%m%dT%H%M%S')
normalizefile = 'data/KNMI_2000_hourly_normalizer'
csvfile = 'data/KNMI_2000_hourly_model.csv'

# read data from csv and strip column names
weather_data2000 = pd.read_csv('data/KNMI_2000_hourly.csv').rename(columns=lambda x: x.strip())
weather_data2001 = pd.read_csv('data/KNMI_2001_hourly.csv').rename(columns=lambda x: x.strip())
weather_data = pd.concat([weather_data2000, weather_data2001])

# get rid of the columns that are not used
weather_data.drop('STN', axis=1, inplace=True)
weather_data.drop('T10', axis=1, inplace=True)
weather_data.drop('WW', axis=1, inplace=True)
weather_data.drop('YYYYMMDD', axis=1, inplace=True)
weather_data.drop('DD', axis=1, inplace=True)
weather_data.drop('FF', axis=1, inplace=True)
weather_data.drop('FX', axis=1, inplace=True)
weather_data.drop('IX', axis=1, inplace=True)

# temporary drop all except hour
# weather_data.drop('FH', axis=1, inplace=True)
# weather_data.drop('T', axis=1, inplace=True)
weather_data.drop('TD', axis=1, inplace=True)
weather_data.drop('SQ', axis=1, inplace=True)
# weather_data.drop('Q', axis=1, inplace=True)
# weather_data.drop('DR', axis=1, inplace=True)
# weather_data.drop('RH', axis=1, inplace=True)
# weather_data.drop('P', axis=1, inplace=True)
weather_data.drop('VV', axis=1, inplace=True)
weather_data.drop('N', axis=1, inplace=True)
# weather_data.drop('U', axis=1, inplace=True)
weather_data.drop('M', axis=1, inplace=True)
weather_data.drop('R', axis=1, inplace=True)
weather_data.drop('S', axis=1, inplace=True)
weather_data.drop('O', axis=1, inplace=True)
weather_data.drop('Y', axis=1, inplace=True)

# normalize the weather data
weather_data = normalize(weather_data, normalizefile, True)

# enrich data HH (hour of day)
# weather_data[['HH_sin', 'HH_cos']] = weather_data.apply(lambda row: pd.Series(circularize(row['HH'])), axis=1)
weather_data.drop('HH', axis=1, inplace=True)

data = source_dataset.prepare_data_sets(weather_data, 0.8)

# the Graph
num_source_vertices = len(data.train.sources.columns)
num_target_vertices = len(data.train.targets.columns)
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

# cross_entropy = -tf.reduce_mean(target_data * tf.log(target_operation)) * 1000.0  # normalized for batches of 100 images,
                                                          # *10 because  "mean" included an unwanted division by 10

# accuracy of the trained model, between 0 (worst) and 1 (best)
# correct_prediction = tf.equal(tf.argmax(target_data, 1), tf.argmax(target_operation, 1))
correct_prediction = 1.0 - tf.abs(tf.sub(target_data, sink_data))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# cost = tf.reduce_mean(tf.square(tf.sub(target_data, target_operation)))
cross_entropy= tf.reduce_mean(tf.square(tf.sub(target_data, sink_data))) * 500.0  # normalized for batches of 50 fact sets,
                                                          # *10 because  "mean" included an unwanted division by 10
# cross_entropy = -tf.reduce_mean(target_data * tf.log(sink_data)) * 500.0  # normalized for batches of 50 fact sets,
                                                          # *10 because  "mean" included an unwanted division by 10

# cross_entropy = tf.nn.l2_loss(sink_data - target_data)
# train_step = tf.train.AdamOptimizer(learning_rate=0.003).minimize(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)


# shutil.rmtree(modelfile, ignore_errors=True) # so that we don't load weights from previous runs\n

# the visual tensorflow output while training
# matplotlib visualisation
allweights1 = tf.reshape(weights1, [-1]) # '[-1]' flattens tensor to [float32, float32, ..., float32]
allbiases1 = tf.reshape(biases1, [-1])
allweights2 = tf.reshape(weights2, [-1])
allbiases2 = tf.reshape(biases2, [-1])
allweights3 = tf.reshape(weights3, [-1])
allbiases3 = tf.reshape(biases3, [-1])

tf_plot = visualtf.Plot()

# init tensorflow
# init = tf.initialize_all_variables() # deprecated
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# This function is called in the tf_plot.animate(training_step, ..) loop to train the model
# in mini-batches of N fatcs at a time
def training_step(i, update_test_data, update_train_data):

    # training on batches of N facts
    batch_source, batch_target = data.train.next_batch(50)

    # compute training values for visualisation
    if update_train_data:
        a, c, w1, b1, w2, b2, w3, b3 = sess.run([accuracy, cross_entropy, allweights1, allbiases1, allweights2, allbiases2, allweights3, allbiases3],
            feed_dict = {source_data : batch_source.values, target_data : batch_target.values}) #.reshape(trainsize, num_target_vertices)})
        tf_plot.append_training_curves_data(i, a, c)
        tf_plot.append_data_histograms(i, w1, b1, w2, b2, w3, b3)
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c))

    # compute test values for visualisation
    if update_test_data:
        a, c = sess.run([accuracy, cross_entropy], feed_dict = {source_data : data.test.sources.values, target_data : data.test.targets.values})
        tf_plot.append_test_curves_data(i, a, c)
        print(str(i) + ": ********* epoch " + str(i) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))

    # the backpropagation training step
    sess.run(train_step, feed_dict = {source_data : batch_source.values, target_data : batch_target.values})

# Add ops to save and restore all the variables.
def save_model():
    print("max test accuracy: " + str(tf_plot.get_max_test_accuracy()))
    saver = tf.train.Saver()
    save_path = saver.save(sess, modelfile)
    print("Model preserved in file: %s" % modelfile)

    # Write the model as array to disk, for analysis
    with file(modeltxtfile, 'w') as outfile:
        outfile.write('#layer1\n#weights\n')
        np.savetxt(outfile, weights1.eval(session=sess))
        outfile.write('#biases\n')
        np.savetxt(outfile, biases1.eval(session=sess))
        outfile.write('#layer2\n#weights\n')
        np.savetxt(outfile, weights2.eval(session=sess))
        outfile.write('#biases\n')
        np.savetxt(outfile, biases2.eval(session=sess))
        outfile.write('#layer3\n#weights\n')
        np.savetxt(outfile, weights3.eval(session=sess))
        outfile.write('#biases\n')
        np.savetxt(outfile, biases3.eval(session=sess))
    print("Model saved in file: %s" % modeltxtfile)

    # print 'testerror={0}'.format(np.sqrt(cost.eval(feed_dict = {
    #       feature_data : predictors[trainsize:].values,
    #       target_data : targets[trainsize:].values.reshape(testsize, noutputs)
    #   }) / testsize))

tf_plot.animate(training_step, save_model, iterations=6000+1, train_data_update_freq=50, test_data_update_freq=200, more_tests_at_start=True)

# to save the animation as a movie, add save_movie=True as an argument to datavis.animate
# to disable the visualisation use the following line instead of the datavis.animate line
# for i in range(2000+1): training_step(i, i % 50 == 0, i % 10 == 0)
