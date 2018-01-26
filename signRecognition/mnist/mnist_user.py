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
import socket
import sys
import string
import os
FLAGS = None
def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.

  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.

  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob

 
def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial) 
  
def main(_):
  # Import data
	serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	serversocket.bind(("localhost", 5656))
	serversocket.listen(5)	
	(clientsocket, address) = serversocket.accept()
	print("WTF1");
	tf.reset_default_graph();
	b = tf.get_variable("b", [10])
	x = tf.placeholder(tf.float32, [None, 784])
	W = tf.get_variable("W", [784, 10])
	y = tf.matmul(x, W) + b
	  # Create the model
	#x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
	y_ = tf.placeholder(tf.float32, [None, 10])

  # Build the graph for the deep net
	#y_conv, keep_prob = deepnn(x)
	model_saver=tf.train.Saver();
	#with tf.name_scope('pred'):
	#	print("here")
	##pred = tf.argmax(y_conv, 1)
	with tf.Session() as sess:
		print("WTF2");
		print(os.getcwd());
		#model_saver.restore(sess,'save_models/softmax.ckpt')
		model_saver.restore(sess,'mnist\\save_models\\softmax.ckpt')
		while(1):
			st=clientsocket.recv(4);
			gray=cv2.imread("img.png",0)
			gray = cv2.resize(255-gray, (28, 28)).flatten()
			flatten=np.zeros(784);
			#reverse for the arabs
			for i in range(28):
			  for j in range (28):
			    flatten[i*28+j]=gray[j*28+i]
			#flatten = gray.flatten() / 255.0
			#image=np.subtract(np.ones(784),np.divide(image.flatten(),255))
			#img=np.multiply(np.subtract(np.ones(784),mnist.test.images[0]),255);
			
			#print(image)
			#print(image.size)
			prediction=sess.run(y,feed_dict={x:[flatten]})
			#print(prediction)
			#prediction=pred.eval(feed_dict={x:[flatten],keep_prob: 1.0})
			
			#str1=str(prediction.argmax());
			pred=(np.exp(prediction[0]/10000)/sum(np.exp(prediction[0]/10000)));
			print(pred)
			print(pred[5],pred[7],pred[8])
			#pred4 = [prediction[0][5],prediction[0][7],prediction[0][8]]
			#dict = {0:5, 1:7, 2:8}
			#for i in range(3):
			#	if(pred4[i] == max(pred4)):
			#		print(dict[i])
			str1=str(pred.argmax())+"\r\n"
					
			clientsocket.sendall(str1.encode())
			
			
  
											

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
