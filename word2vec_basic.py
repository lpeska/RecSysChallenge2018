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

from tensorflow.contrib.tensorboard.plugins import projector

vocabulary_size = 2000000 #250000
#vocabulary_size = 2000000 #50000, 250000, #2000000
batch_size = 128
embedding_size = 64  # 32, 64, 128 Dimension of the embedding vector.
skip_window = 6  # How many words to consider left and right.
num_skips = 4  # How many times to reuse an input to generate a label.
num_sampled = 32  # Number of negative examples to sample.

num_steps = 1000001

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
  return data


vocabulary = read_data(filename)
print('Data size', len(vocabulary))

# Step 2: Build the dictionary and replace rare words with UNK token.

def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  ctr = collections.Counter(words)
  count = [['UNK', -1]]
  count.extend(ctr.most_common(n_words - 1))
  names = dict()
  newAppend = 0
  for i in count:
    names[i[0]] = 1
  print(list(names.keys())[0:10])
  print(count[5])
  #TODO: add known items from the challenge set - creative track only!!
  with open("challenge_track_names.csv", "r") as chTr:
    for tr in chTr:
      tr = tr.replace("\n","")
      if names.get(tr, 0) <= 0:
        count.append([tr, ctr[tr]])
        if newAppend % 1000 == 0:
          print([tr, ctr[tr]])
        newAppend += 1

  print("Appended challenge data: " +str(newAppend) )

  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    index = dictionary.get(word, 0)
    if index == 0:  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary, len(count)


# Filling 4 global variables:
# data - list of codes (integers from 0 to vocabulary_size-1).
#   This is the original text but words are replaced by their codes
# count - map of words(strings) to count of occurrences
# dictionary - map of words(strings) to their codes(integers)
# reverse_dictionary - maps codes(integers) to words(strings)
data, count, dictionary, reverse_dictionary, vocabulary_size = build_dataset(
    vocabulary, vocabulary_size)
del vocabulary  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
print("New vocabulary size: "+ str(vocabulary_size))
data_index = 0

import pandas as pd

tracks = pd.read_csv("trackCounter.csv", delimiter=";", header=0)
tracks.set_index("TrackID", inplace=True)
tracks = tracks.Count.to_dict()
trackCount = []  # TODO
for t in range(vocabulary_size):
    trackCount.append(tracks.get(reverse_dictionary[t], 1))
print(trackCount[0:30])

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
resultBatchSize = 100
extSize = len(challengeTrackIDs) % resultBatchSize
print(len(challengeTrackIDs))
challengeTrackIDs.extend([0] * (resultBatchSize-extSize))
print(extSize, resultBatchSize, len(challengeTrackIDs))
challengeTrackIDs = np.asarray(challengeTrackIDs)
print(challengeTrackIDs.shape[0], challengeTrackIDs.shape[0] // resultBatchSize)
batches = np.split(challengeTrackIDs, challengeTrackIDs.shape[0] // resultBatchSize)

# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
  if data_index + span > len(data):
    data_index = 0
  buffer.extend(data[data_index:data_index + span])
  data_index += span
  for i in range(batch_size // num_skips):
    context_words = [w for w in range(span) if w != skip_window]
    words_to_use = random.sample(context_words, num_skips)
    for j, context_word in enumerate(words_to_use):
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[context_word]
    if data_index == len(data):
      buffer.extend(data[0:span])
      data_index = span
    else:
      buffer.append(data[data_index])
      data_index += 1
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels

"""
batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0],
        reverse_dictionary[labels[i, 0]])
"""
# Step 4: Build and train a skip-gram model.



# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.
valid_size = 4  # Random set of words to evaluate similarity on.
valid_window = 10  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

graph = tf.Graph()

with graph.as_default():

  # Input data.
  with tf.name_scope('inputs'):
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    challenge_dataset = tf.placeholder(tf.int32, shape=[resultBatchSize])
    trackMultiplier = tf.constant([trackCount], dtype=tf.float32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    with tf.name_scope('embeddings'):
      embeddings = tf.Variable(
          tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
      embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    with tf.name_scope('weights'):
      nce_weights = tf.Variable(
          tf.truncated_normal(
              [vocabulary_size, embedding_size],
              stddev=1.0 / math.sqrt(embedding_size)))
    with tf.name_scope('biases'):
      nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  # Explanation of the meaning of NCE loss:
  #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
  with tf.name_scope('loss'):
    loss = tf.reduce_mean(
        tf.nn.nce_loss(
            weights=nce_weights,
            biases=nce_biases,
            labels=train_labels,
            inputs=embed,
            num_sampled=num_sampled,
            num_classes=vocabulary_size))

  # Add the loss value as a scalar to summary.
  tf.summary.scalar('loss', loss)

  # Construct the SGD optimizer using a learning rate of 1.0.
  with tf.name_scope('optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                            valid_dataset)
  challenge_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                            challenge_dataset)
  logits = tf.matmul(challenge_embeddings, tf.transpose(nce_weights))
  logits = tf.nn.bias_add(logits, nce_biases)



  challengeVals, challengeIndices = tf.nn.top_k(logits,k=1000,sorted=True,name=None)
  challengeVals = tf.nn.softmax(challengeVals)

  challenge_sim = tf.matmul(
      challenge_embeddings, normalized_embeddings, transpose_b=True)
  trackMultiplier = tf.log(trackMultiplier) + 0.1
  #challenge_sim = tf.multiply(challenge_sim, trackMultiplier)

  challenge_simVals, challenge_simIndices = tf.nn.top_k(challenge_sim,k=1000,sorted=True,name=None)
  challenge_simVals = tf.nn.softmax(challenge_simVals)

  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)



  # Merge all summaries.
  merged = tf.summary.merge_all()

  # Add variable initializer.
  init = tf.global_variables_initializer()

  # Create a saver.
  saver = tf.train.Saver()

# Step 5: Begin training.


with tf.Session(graph=graph) as session:
  # Open a writer to write summaries.
  writer = tf.summary.FileWriter(FLAGS.log_dir, session.graph)

  # We must initialize all variables before we use them.
  init.run()
  print('Initialized')

  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(batch_size, num_skips,
                                                skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

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

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    """
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = '%s %s,' % (log_str, close_word)
        print(log_str)
    """

  final_embeddings = normalized_embeddings.eval()
  embeddingsFname = "embedImprSim_word2vec_"+str(vocabulary_size)+"_"+str(embedding_size)+".csv"
  #embeddingsFname = "embed_word2vec_album_"+str(vocabulary_size)+"_"+str(embedding_size)+".csv"

  np.savetxt(embeddingsFname, final_embeddings, fmt="%.5e")
  with open("names_"+embeddingsFname, 'w') as f:
    for i in xrange(vocabulary_size):
      f.write(reverse_dictionary[i] + '\n')


  def trName(id):
    return reverse_dictionary[id]
  trackNameVectorized = np.vectorize(trName, otypes=[str])


  #with open("challenge_track_predImprSim_"+str(vocabulary_size)+"_"+str(embedding_size)+".csv","ab") as fPred:
  #  with open("challenge_track_scoreImprSim_"+str(vocabulary_size)+"_"+str(embedding_size)+".csv", "ab") as fScore:
  with open("challenge_track_predNNSimFinal_" + str(vocabulary_size) + "_" + str(embedding_size) + ".csv","ab") as knnPred:
    with open("challenge_track_scoreNNSimFinal_" + str(vocabulary_size) + "_" + str(embedding_size) + ".csv","ab") as knnScore:
              for batch in batches:

                  sim, ind, nnSim, nnInd = session.run([challengeVals, challengeIndices, challenge_simVals, challenge_simIndices], feed_dict={challenge_dataset:batch})
                  ind = trackNameVectorized(ind)
                  nnInd = trackNameVectorized(nnInd)

                  print(nnSim[0, 0:3])
                  print(nnInd[0, 0:3])

                  #np.savetxt(fPred, ind, delimiter=';', fmt="%s")
                  #np.savetxt(fScore, sim, fmt="%.6f", delimiter=';')

                  np.savetxt(knnPred, nnInd, delimiter=';', fmt="%s")
                  np.savetxt(knnScore, nnSim, fmt="%.6f", delimiter=';')

                  """
                  valid_word = reverse_dictionary[i]
                  top_k = 500  # number of nearest neighbors
                  nearest = (-sim[0, :]).argsort()[1:top_k + 1]
                  tracks = list()
                  scores = list()
                  for k in xrange(top_k):
                    tracks.append(reverse_dictionary[nearest[k]])
                    scores.append( str(round(sim[0,nearest[k]],5)) )
                  fPred.write(",".join(tracks)+"\n")
                  fScore.write(",".join(scores) + "\n")
                  """




