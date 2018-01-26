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

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
import cv2
import csv
import random
FLAGS = None


def main(_):
  # Import data
 # mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  #print(mnist.train.labels[0])
  csvTrainImages=open("csvTrainImages.csv",'r')
  csvTrainImagesReader=csv.reader(csvTrainImages)
  csvTrainLabel=open("csvTrainLabel.csv",'r')
  csvTrainLabelReader=csv.reader(csvTrainLabel)
  csvTestImages=open("csvTestImages.csv",'r')
  csvTestImagesReader=csv.reader(csvTestImages)
  csvTestLabel=open("csvTestLabel.csv",'r')
  csvTestLabelReader=csv.reader(csvTestLabel)
  trainLabel=[]
  trainImages=[]
  testLabel=[]
  testImages=[]
  for row in csvTrainLabelReader:
    v=np.zeros(10)
    v[int(row[0])]=1
    trainLabel.append(v)
  for row in csvTestLabelReader:
      v=np.zeros(10)
      v[int(row[0])]=1
      testLabel.append(v)
  for row in csvTrainImagesReader:
    trainImages.append(row)
  for row in csvTestImagesReader:
    testImages.append(row)
  print(len(trainLabel))
  print(len(trainImages))
  print(len(testLabel))
  print(len(testImages))
  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  for _ in range(1000):
    rand=random.sample(range(0,60000),300)
    batch_xs=[]
    batch_ys=[]
    for i in rand:
      #print(i)
      batch_xs.append(trainImages[i])
      batch_ys.append(trainLabel[i])
    #batch_xs, batch_ys = mnist.train.next_batch(300)
    #print(batch_xs[0][7])
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: testImages,
                                      y_: testLabel}))
  model_saver=tf.train.Saver({"b" : b ,"W" :W});
  model_saver.save(sess,"save_models/softmax.ckpt")
  
  
											

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
