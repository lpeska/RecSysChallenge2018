# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Basic word2vec example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import sys
import argparse
import random
from tempfile import gettempdir
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import pandas as pd

from tensorflow.contrib.tensorboard.plugins import projector

vocabulary_size = 250000 #50000, 250000, #734684
#vocabulary_size = 2000000 #50000, 250000, #2000000
batch_size = 200
#embedding_size = 64  # 32, 64, 128 Dimension of the embedding vector.
skip_window = 10  # How many words to consider left and right.
num_negative = 4  # How many times to reuse an input to generate a label.
#num_sampled = 32  # Number of negative examples to sample.

num_steps = 500001

filename = "word2vec_tracks.txt"
#filename = "word2vec_albums.txt"

# Give a folder path as an argument with '--log_dir' to save
# TensorBoard summaries. Default is a log folder in current directory.
current_path = os.path.dirname(os.path.realpath(sys.argv[0]))

parser = argparse.ArgumentParser()
parser.add_argument(
    '--log_dir',
    type=str,
    default=os.path.join(current_path, 'log'),
    help='The log directory for TensorBoard summaries.')
FLAGS, unparsed = parser.parse_known_args()

# Create the directory for TensorBoard variables if there is not.
if not os.path.exists(FLAGS.log_dir):
  os.makedirs(FLAGS.log_dir)




# Read the data into a list of strings.
def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words."""
  with open(filename) as f:
    data = tf.compat.as_str(f.read()).split()
  data = [i for i in data if len(i)<30]
  return data


vocabulary = read_data(filename)
print('Data size', len(vocabulary))

# Step 2: Build the dictionary and replace rare words with UNK token.



#TODO: implement this
# load data
# create tensor from them
# use tf.nn.embedding_lookup in batches
# emploi siamese loss
def getAudioFeaturesData(): #rowOrder
    audioTrack = pd.read_csv("trackAudioStd.csv", delimiter=",", header=0, index_col=0)
    audioTrack.set_index("id", inplace=True)
    #print(audioTrack.head())
    #audioTrack.reindex(rowOrder, copy=False) #TODO: create rowOrder
    """
    noVal = np.array([0,0,0,0,0,0,0,0,0,0])
    at = []
    for v in rowOrder:
        try:
            at.append(audioTrack.xs(v).tolist())
        except:
            at.append(noVal)
    """
    #print(data[2000000], reverse_dictionary[data[2000000]], audioTrack.xs(reverse_dictionary[data[2000000]]).values)
    return audioTrack
#TODO: use audioTrack.xs(reverse_dictionary[data[2000000]]) to populate embedings in TF - hope it wont be too slow:)
#rowOrder = [k for k in sorted(dictionary, key=dictionary.get, reverse=True)]
#audioDF = getAudioFeaturesData(rowOrder).values
audioDF = getAudioFeaturesData() #rowOrder

#print(rowOrder[0:5])
#print(audioDF[0:5])


def build_dataset(words, validWords):
  """Process raw inputs into a dataset."""
  #ctr = collections.Counter(words)
  #count = []
  #count.extend(ctr.most_common())
  #print(count[5])
  dictionary = dict()
  for word in validWords.index.values:
    dictionary[word] = len(dictionary)
  data = list()
  for word in words:
    index = dictionary.get(word, -1)
    if index != -1:
        data.append(index)
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  data = np.asarray(data)
  return (data, dictionary, reversed_dictionary, len(validWords))


# Filling 4 global variables:
# data - list of codes (integers from 0 to vocabulary_size-1).
#   This is the original text but words are replaced by their codes
# count - map of words(strings) to count of occurrences
# dictionary - map of words(strings) to their codes(integers)
# reverse_dictionary - maps codes(integers) to words(strings)
data, dictionary, reverse_dictionary, vocabulary_size = build_dataset(vocabulary, audioDF)
del vocabulary  # Hint to reduce memory.
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
print("New vocabulary size: "+ str(vocabulary_size))
print("Largest data entry:" + str(np.amax(data)) +", "+ str(reverse_dictionary[np.amax(data)]))


def getIDsForChallengeTracks(dictionary):
  data = list()
  unk_count = 0
  countAll = 0
  with open("challenge_track_names.csv", "r") as chTr:
    for word in chTr:
        countAll += 1
        word = word.replace("\n", "")
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
  print("There were "+ str(unk_count) +" unknown challenge set tracks")
  return data, countAll
challengeTrackIDs, countChallengeTracks = getIDsForChallengeTracks(dictionary)
print(challengeTrackIDs[0:10])





np.random.seed(2018)
# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_negative, skip_window):
    #generate indices of left examples
    #generate indices of similar tracks:= tracks to the distance of skip_window from left examples
    #generate indices of neg examples at random

    assert batch_size % (num_negative + 1) == 0
    batchLeft = np.ndarray(shape=(batch_size), dtype=np.int32)
    batchRight = np.ndarray(shape=(batch_size), dtype=np.int32)
    batchLabel = np.ndarray(shape=(batch_size), dtype=np.int32)

    numExamples = batch_size // (num_negative + 1)
    leftPosExampleIDs = np.random.randint(skip_window, len(data)-skip_window, numExamples)
    rightPosExampleIDs = leftPosExampleIDs + np.random.randint(-skip_window, skip_window, numExamples)
    batchLeft[0:numExamples] = data[leftPosExampleIDs]
    batchRight[0:numExamples] = data[rightPosExampleIDs]
    batchLabel[0:numExamples] = 1

    leftNegExampleIDs = leftPosExampleIDs.repeat(num_negative)
    rightNegExampleIDs = np.random.randint(0, len(data), (numExamples*num_negative))
    batchLeft[numExamples:] = data[leftNegExampleIDs]
    batchRight[numExamples:] = data[rightNegExampleIDs]
    batchLabel[numExamples:] = 0

    return batchLeft, batchRight, batchLabel



def loss_with_spring(y, o1, o2):
        margin = 5.0
        labels_t = y
        labels_f = tf.subtract(1.0, y, name="1-yi")          # labels_ = !labels;
        eucd2 = tf.pow(tf.subtract(o1, o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        # yi*||CNN(p1i)-CNN(p2i)||^2 + (1-yi)*max(0, C-||CNN(p1i)-CNN(p2i)||^2)
        pos = tf.multiply(labels_t, eucd2, name="yi_x_eucd2")
        # neg = tf.multiply(labels_f, tf.subtract(0.0,eucd2), name="yi_x_eucd2")
        # neg = tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C,eucd2)), name="Nyi_x_C-eucd_xx_2")
        neg = tf.multiply(labels_f, tf.pow(tf.maximum(tf.subtract(C, eucd), 0), 2), name="Nyi_x_C-eucd_xx_2")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss

def network(x1, x2):
    y11 = tf.layers.dense(x1, 256, name='h1')
    y11Relu = tf.nn.relu(y11)
    y11a = tf.layers.dense(y11Relu, 64, name='h11')
    y11aRelu = tf.nn.relu(y11a)

    y12 = tf.layers.dense(y11aRelu, 32, name='h2')


    y21 = tf.layers.dense(x2, 256, name='h1', reuse=True)
    y21Relu = tf.nn.relu(y21)
    y21a = tf.layers.dense(y21Relu, 64, name='h11', reuse=True)
    y21aRelu = tf.nn.relu(y21a)

    y22 = tf.layers.dense(y21aRelu, 32, name='h2', reuse=True)

    return y12, y22


graph = tf.Graph()

with graph.as_default():
    # Input data.
    with tf.name_scope('inputs'):
        embed1 = tf.placeholder(tf.float32, shape=[batch_size,10])
        embed2 = tf.placeholder(tf.float32, shape=[batch_size,10])
        #train_input2 = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.float32, shape=[batch_size])
        challenge_dataset = tf.constant(challengeTrackIDs, dtype=tf.int32)

        #inputFeatures = tf.constant(audioDF, dtype=tf.float32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        #with tf.name_scope('embeddings'):

            #embed1 = tf.nn.embedding_lookup(inputFeatures, train_input1)
            #embed2 = tf.nn.embedding_lookup(inputFeatures, train_input2)

        with tf.variable_scope("siamese") as scope:
            output1, output2  = network(embed1, embed2)

        loss = loss_with_spring(train_labels,  output1, output2)

    # Add the loss value as a scalar to summary.
    tf.summary.scalar('loss', loss)

    # Construct the SGD optimizer using a learning rate of 1.0.
    with tf.name_scope('optimizer'):
        #optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

    # Merge all summaries.
    merged = tf.summary.merge_all()

    # Add variable initializer.
    init = tf.global_variables_initializer()

    # Create a saver.
    saver = tf.train.Saver()

# Step 5: Begin training.


with tf.Session(graph=graph) as session:
    #batchLeft, batchRight, labels = generate_batch(batch_size=8, num_negative=3, skip_window=6)



    #for i in range(8):
    #    print(reverse_dictionary[batchLeft[i]], batchLeft[i], batchRight[i], labels[i])
    #print(tf.nn.embedding_lookup(inputFeatures, batchLeft).eval())
    #print(tf.nn.embedding_lookup(inputFeatures, batchLeft).eval())

    # Open a writer to write summaries.
    writer = tf.summary.FileWriter(FLAGS.log_dir, session.graph)

    # We must initialize all variables before we use them.
    init.run()
    print('Initialized')

    average_loss = 0
    for step in xrange(num_steps):
        batchLeft, batchRight, labels = generate_batch(batch_size, num_negative, skip_window)
        embedLeft = audioDF.iloc[batchLeft.tolist()].values  #(list(map(reverse_dictionary.get, batchLeft.tolist()))).values
        embedRight = audioDF.iloc[batchRight.tolist()].values #(list(map(reverse_dictionary.get, batchRight.tolist()))).values

        #embedLeft = audioDF.xs(reverse_dictionary[batchLeft.tolist()]).values
        #embedRight = audioDF.xs(reverse_dictionary[batchRight.tolist()]).values

        feed_dict = {embed1: embedLeft, embed2: embedRight, train_labels:labels}

        # Define metadata variable.
        run_metadata = tf.RunMetadata()

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        # Also, evaluate the merged op to get all summaries from the returned "summary" variable.
        # Feed metadata variable to session for visualizing the graph in TensorBoard.
        _, summary, loss_val = session.run(
            [optimizer, merged, loss],
            feed_dict=feed_dict,
            run_metadata=run_metadata)
        average_loss += loss_val

        # Add returned summaries to writer in each step.
        writer.add_summary(summary, step)
        # Add metadata to visualize the graph for the last run.
        if step == (num_steps - 1):
            writer.add_run_metadata(run_metadata, 'step%d' % step)

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step ', step, ': ', average_loss)
            average_loss = 0


    with open("names_audioFeatureSimEmbed3_32.csv", 'w') as f:
        for i in xrange(vocabulary_size):
            f.write(reverse_dictionary[i] + '\n')

    with open("audioFeatureSimEmbed3_32.csv", "ab") as fPred:
            for i in xrange(vocabulary_size):
                if i % batch_size == 0:
                    if i != 0:
                        embLeft = audioDF.iloc[items.tolist()].values
                        #embLeft = audioDF.xs(list(map(reverse_dictionary.get, items))).values
                        emb = session.run(output1,feed_dict={embed1: embLeft})
                        np.savetxt(fPred, emb, fmt="%.6f", delimiter=';')

                    items = np.zeros(batch_size)
                j = i % batch_size
                items[j] = i

            embLeft = audioDF.iloc[items.tolist()].values
            #embLeft = audioDF.xs(list(map(reverse_dictionary.get, items.tolist()))).values
            emb = session.run(output1, feed_dict={embed1: embLeft})
            np.savetxt(fPred, emb, fmt="%.6f", delimiter=';')




    """ 
    # Write corresponding labels for the embeddings.
    with open(FLAGS.log_dir + '/metadata.tsv', 'w') as f:
      for i in xrange(vocabulary_size):
        f.write(reverse_dictionary[i] + '\n')
  
    # Save the model for checkpoints.
    saver.save(session, os.path.join(FLAGS.log_dir, 'model.ckpt'))
  
    # Create a configuration for visualizing embeddings with the labels in TensorBoard.
    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = embeddings.name
    embedding_conf.metadata_path = os.path.join(FLAGS.log_dir, 'metadata.tsv')
    projector.visualize_embeddings(writer, config)
    """
writer.close()
