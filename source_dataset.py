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

import numpy
import tensorflow as tf
tf.set_random_seed(0)

class DataSet(object):

    def __init__(self, sources, targets, label=""):
        self._num_examples = sources.shape[0]
        self._sources = sources
        self._targets = targets
        self._epochs_completed = 0
        self._index_in_epoch = 0
        print '\ndata %r:\nsources:\n%r\ntargets:\n%r' %(label, sources[:5], targets[:5])

    @property
    def sources(self):
        return self._sources

    @property
    def targets(self):
        return self._targets

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # # Shuffle the data
            # perm = numpy.arange(self._num_examples)
            # numpy.random.shuffle(perm)
            # self._images = self._images[perm]
            # self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._sources[start:end], self._targets[start:end]


def prepare_data_sets(data, train_percentage=0.0):
    class DataSets(object):
        pass

    print '*** preparing dataset ***'
    data_sets = DataSets()

    ncolumns = len(data.columns)
    # add target/output (next row) as column _t (target)
    for col in list(data.columns.values):
      data[col + '_t'] = data[col].shift(-1)
    # drop the last row as this has no target values
    data = data[:-1]
    print 'data:\n', data[:5], data.shape

    # shuffle thet data
    shuffled = data.sample(frac=1)
    print 'data shuffled:\n', shuffled[:5]
    # split in facts and related targets (next row state)
    sources = shuffled.iloc[:,0:ncolumns]
    targets = shuffled.iloc[:,ncolumns:]

    # We'll use a large % of our dataset for training (default 80%)and the rest for testing the model we have trained.
    trainsize = int(shuffled.shape[0] * train_percentage)
    train_sources = sources[:trainsize]
    train_targets = targets[:trainsize]
    print 'train-set size:', train_sources.shape[0]

    test_sources = sources[trainsize:]
    test_targets = targets[trainsize:]
    print 'test-set size:', test_sources.shape[0]

    data_sets.train = DataSet(train_sources, train_targets, 'train')
    data_sets.test = DataSet(test_sources, test_targets, 'test')

    return data_sets
