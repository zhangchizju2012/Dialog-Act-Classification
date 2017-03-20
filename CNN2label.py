#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 17:41:45 2017

@author: zhangchi
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
import os
from tensorflow.models.rnn.ptb import reader
from model import LanguageModel
from tensorflow.contrib import learn

def real_len(batches):
    return [np.ceil(np.argmin(batch + [0]) * 1.0) for batch in batches]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int(data_size / batch_size)

	for epoch in range(num_epochs):
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data

		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = (batch_num + 1) * batch_size
			yield shuffled_data[start_index:end_index]

class Config(object):
  init_scale = 0.1
  learning_rate = 0.001
  max_grad_norm = 5
  num_layers = 5
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 64
  vocab_size = 10000
  
class EvaConfig(object):
  init_scale = 0.1
  learning_rate = 0.001
  max_grad_norm = 5
  num_layers = 5
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 1
  vocab_size = 10000

class CNNModel(LanguageModel):
  """Implements a Softmax classifier with cross-entropy loss."""
  def load_data(self):
    fin = open('result.txt','r')
    dictionary = {'sd':[1,0],'sv':[0,1]}
    x_text = []
    y = []
    for line in fin:
        if line.split(':')[0] == 'sd' and np.random.randint(3) == 0:
            x_text.append(line.split(':')[1].split('\n')[0].lower())
            y.append(dictionary[line.split(':')[0]])
        elif line.split(':')[0] == 'sv':
            x_text.append(line.split(':')[1].split('\n')[0].lower())
            y.append(dictionary[line.split(':')[0]])
    self.y = np.array(y)
    del y
    self.max_document_length = max([len(x.split(" ")) for x in x_text])

    vocabulary = learn.preprocessing.CategoricalVocabulary()
    vocabulary_file = open('vocabulary.txt','r')
    for line in vocabulary_file:
        vocabulary.add(line.split('\n')[0])

    vocab_processor = learn.preprocessing.VocabularyProcessor(self.max_document_length,vocabulary=vocabulary)
    self.x = np.array(list(vocab_processor.fit_transform(x_text)))
    self.vocabularySize = vocabulary.__len__()
    print(str(self.max_document_length))
    del x_text
    
    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(self.y)))
    x_shuffled = self.x[shuffle_indices]
    y_shuffled = self.y[shuffle_indices]
    del self.x, self.y, shuffle_indices
    
    dev_sample_index = -1 * int(0.2 * float(len(y_shuffled)))
    self.x_train, self.x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    self.y_train, self.y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    del x_shuffled, y_shuffled
    
    print("Vocabulary Size: {:d}".format(self.vocabularySize))
    print("Train/Dev split: {:d}/{:d}".format(len(self.y_train), len(self.y_dev)))
    print("Load Data Finished")

  def add_placeholders(self):
    self.input_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.max_document_length))
    self.labels_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, 2))
    self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
    self.real_len = tf.placeholder(tf.int32, [None], name='real_len')

  def create_feed_dict(self, input_batch, label_batch, dropout_keep_prob, batch_real_len):
    feed_dict = {
        self.input_placeholder: input_batch,
        self.labels_placeholder: label_batch,
        self.dropout_keep_prob: dropout_keep_prob,
        self.real_len:batch_real_len}
    return feed_dict

  def add_embedding(self,is_training,sess):
    with tf.device("/cpu:0"):
        zero=np.float32((np.random.rand(1,50)-0.5)*0.2)
        last=np.float32((np.random.rand(self.vocabularySize-400001,50)-0.5)*0.2)
        W = np.concatenate((zero,np.loadtxt('embedding.txt',dtype=np.float32),last),axis=0)
        embedding = tf.Variable(W, name='W')
        del W
        inputs = tf.nn.embedding_lookup(embedding, self.input_placeholder)
        # The result of the embedding operation is a 3-dimensional tensor of shape [None, sequence_length, embedding_size].
        inputs = tf.expand_dims(inputs, -1)
        # TensorFlow’s convolutional conv2d operation expects a 4-dimensional tensor with dimensions corresponding to batch, width, height and channel. The result of our embedding doesn’t contain the channel dimension, 
        # so we add it manually, leaving us with a layer of shape [None, sequence_length, embedding_size, 1].

    if is_training and self.config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, self.config.keep_prob)
    self.inputs = inputs
    return inputs

  def add_model(self, inputs, is_training):
      filter_sizes = [3,4,5]
      num_filters = 128
      pooled_outputs = []
      for i, filter_size in enumerate(filter_sizes):
          with tf.name_scope("conv-maxpool-%s" % filter_size):
              # Convolution Layer
              filter_shape = [filter_size, 50, 1, num_filters]
              W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
              b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
              conv = tf.nn.conv2d(
                inputs,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
              # Apply nonlinearity
              h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
              # Max-pooling over the outputs
              pooled = tf.nn.max_pool(
                h,
                ksize=[1, self.max_document_length - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
              pooled_outputs.append(pooled)
         
      # Combine all the pooled features
      num_filters_total = num_filters * len(filter_sizes)
      self.h_pool = tf.concat(3, pooled_outputs)
      self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
      
      # Add dropout
      with tf.name_scope("dropout"):
          self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
    
      l2_loss = tf.constant(0.0)
      with tf.name_scope("output"):
            self.W = tf.Variable(tf.truncated_normal([num_filters_total, 2], stddev=0.1), name="W")
            self.b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")
            l2_loss += tf.nn.l2_loss(self.W)
            l2_loss += tf.nn.l2_loss(self.b)
            self.l2_loss = l2_loss
            self.scores = tf.nn.xw_plus_b(self.h_drop, self.W, self.b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            self.probablity = tf.nn.softmax(self.scores)
            self.target = tf.argmax(self.labels_placeholder, 1)

      with tf.name_scope('loss'):
		losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.labels_placeholder) 
		self.loss = tf.reduce_mean(losses) + 0.1 * l2_loss

      with tf.name_scope('accuracy'):
		correct_predictions = tf.equal(self.predictions, tf.argmax(self.labels_placeholder, 1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name='accuracy')

      with tf.name_scope('num_correct'):
		correct = tf.equal(self.predictions, tf.argmax(self.labels_placeholder, 1))
		self.num_correct = tf.reduce_sum(tf.cast(correct, 'float'))
        
  def add_loss_op(self):
    # no need anymore
    
    return None

  def add_training_op(self):
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-4)
    # optimizer = tf.train.RMSPropOptimizer(1e-3, decay=0.9)
    grads_and_vars = optimizer.compute_gradients(self.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step) 
    return train_op
    
  #def run_epoch(self, session, eval_op, saver,checkpoint_prefix, verbose=False):
  def run_epoch(self, session, eval_op, saver,model_path, verbose=False):
    batches = batch_iter(list(zip(self.x_train, self.y_train)), self.config.batch_size, 200)
    # batches_dev = batch_iter(list(zip(self.x_dev, self.y_dev)), self.config.batch_size, 200)
    costs = 0.0
    l2_costs = 0.0
    iters = 0
    step = 0
    accuracyList = []
    for batch in batches:
        x, y = zip(*batch)
        fetches = [self.b,self.inputs,self.loss,self.l2_loss, self.accuracy,self.predictions, self.scores,self.h_drop,eval_op]
        feed_dict = self.create_feed_dict(x,y,0.5,real_len(x))
        softmax_b,inputs,cost,l2_cost, accuracy,predictions, scores,out,_ = session.run(fetches, feed_dict)
        accuracyList.append(accuracy)
        costs += cost
        l2_costs += l2_cost
        iters += self.max_document_length
        step = step + 1
    
        if verbose and step % 20 == 0:
            # print("%.3f perplexity: %.3f speed: %.0f wps" %(step * 1.0 / 200, np.exp(costs / iters),iters * self.config.batch_size / (time.time() - start_time)))
            print("Train accuracy="+str(np.mean(accuracyList)))
            print(predictions)
            print('cost='+str(costs))
            print('l2_cost='+str(l2_costs))
            #print(len(predictions))
        if step % 100 == 0:
            # dev_step
            batches_dev = batch_iter(list(zip(self.x_dev, self.y_dev)), self.config.batch_size, 1)
            costs_dev = 0.0
            # step_dev = 0
            accuracyList_dev = []
            for batch_dev in batches_dev:
                x_dev, y_dev = zip(*batch_dev)
                fetches = [self.b,self.inputs,self.loss, self.accuracy,self.predictions, self.scores,self.h_drop]
                feed_dict = self.create_feed_dict(x_dev,y_dev,1,real_len(x_dev))
                softmax_b_dev,inputs_dev,cost_dev, accuracy_dev,predictions_dev, scores_dev,out_dev = session.run(fetches, feed_dict)
                accuracyList_dev.append(accuracy_dev)
                costs_dev += cost_dev
                # step_dev = step_dev + 1
            print("-----------------------------------------------")
            print("Test accuracy="+str(np.mean(accuracyList_dev)))
            print("-----------------------------------------------")
        if step % 300 == 0:
            save_path = saver.save(session, model_path)
            print("Model saved in file: %s" % save_path)
            '''
            path = saver.save(session, checkpoint_prefix, global_step=step)
            print("Saved model checkpoint to {}\n".format(path))
            '''
  '''
  def fit(self, sess,saver,checkpoint_prefix):
    self.run_epoch(sess, self.train_op, saver,checkpoint_prefix,True)
  '''
  def fit(self, sess,saver,model_path):
    self.run_epoch(sess, self.train_op, saver,model_path,True)
  def __init__(self, config, sess):
    self.config = config
    self.load_data()
    self.add_placeholders()
    inputs = self.add_embedding(True,sess)
    self.add_model(inputs, True)
    #self.cost = self.add_loss_op()
    self.train_op = self.add_training_op()
  def evaluate(self, sess):
    batches_dev = batch_iter(list(zip(self.x_dev, self.y_dev)), self.config.batch_size, 1)
    costs_dev = 0.0
    # step_dev = 0
    accuracyList_dev = []
    probablityList_dev = []
    targetList_dev = []
    predictionList_dev = []
    scoresList_dev = []
    for batch_dev in batches_dev:
        x_dev, y_dev = zip(*batch_dev)
        fetches = [self.target,self.probablity,self.b,self.inputs,self.loss, self.accuracy,self.predictions, self.scores,self.h_drop]
        feed_dict = self.create_feed_dict(x_dev,y_dev,1,real_len(x_dev))
        target_dev,probablity_dev,softmax_b_dev,inputs_dev,cost_dev, accuracy_dev,predictions_dev, scores_dev,out_dev = sess.run(fetches, feed_dict)
        accuracyList_dev.append(accuracy_dev)
        costs_dev += cost_dev
        # step_dev = step_dev + 1
        probablityList_dev.append(probablity_dev)
        targetList_dev.append(target_dev)
        predictionList_dev.append(predictions_dev)
        scoresList_dev.append(scores_dev)
    print("-----------------------------------------------")
    print("Evaluate accuracy="+str(np.mean(accuracyList_dev)))
    print("-----------------------------------------------")
    return probablityList_dev, targetList_dev, predictionList_dev, scoresList_dev
  
  def evaluateOneByOne(self, sess):
    label = True
    dictionary = {0:'Statement-opinion',
                  1:'Statement-non-opinion'}
    while label:
        data = []
        user_input = raw_input("Enter: ")
        if user_input == 'qqq':
            label = False
        data.append(user_input)
        x = np.array(list(self.vocab_processor.fit_transform(data)))
        y = np.array([0,0]).reshape(1,2)
    
        fetches = [self.b,self.inputs,self.loss, self.accuracy,self.predictions, self.scores,self.h_drop]
        feed_dict = self.create_feed_dict(x,y,1,real_len(x))
        softmax_b_dev,inputs_dev,cost_dev, accuracy_dev,predictions_dev, scores_dev,out_dev = sess.run(fetches, feed_dict)
            # step_dev = step_dev + 1
        print("prediction = "+dictionary[predictions_dev[0]])
        print(predictions_dev[0])
        print(scores_dev)
        print("-----------------------------------------------")

def test_CNNModel(startType='Restart'):
  """Train softmax model for a number of steps."""
  config = Config()
  if startType == 'Restart':
      with tf.Graph().as_default():
          sess = tf.Session()
          initializer = tf.random_uniform_initializer(-0.1,0.1)
          with tf.variable_scope("model", reuse=None, initializer=initializer):
              model = CNNModel(config,sess)
          '''
          # Output directory for models and summaries
          timestamp = str(int(time.time()))
          out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
          print("Writing to {}\n".format(out_dir))
          
          # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
          checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
          checkpoint_prefix = os.path.join(checkpoint_dir, "model")
          if not os.path.exists(checkpoint_dir):
              os.makedirs(checkpoint_dir)
          saver = tf.train.Saver(tf.all_variables(), max_to_keep=3)
          
          init = tf.initialize_all_variables()
          sess.run(init)    
          model.fit(sess,saver,checkpoint_prefix)
          '''
          model_path = "CNNmodel2.ckpt"
          saver = tf.train.Saver(tf.all_variables(), max_to_keep=3)
          init = tf.initialize_all_variables()
          sess.run(init)    
          model.fit(sess,saver,model_path)
  else:
      with tf.Graph().as_default():
          sess = tf.Session()
          initializer = tf.random_uniform_initializer(-0.1,0.1)
          with tf.variable_scope("model", reuse=None, initializer=initializer):
              model = CNNModel(config,sess)
          model_path = "CNNmodel2.ckpt"
          saver = tf.train.Saver(tf.all_variables(), max_to_keep=3)
          #init = tf.initialize_all_variables()
          #sess.run(init)
          saver.restore(sess, model_path)
          model.fit(sess,saver,model_path)

def evaluate_CNNModel():
    config = Config()
    with tf.Graph().as_default():
        sess = tf.Session()
        initializer = tf.random_uniform_initializer(-0.1,0.1)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model = CNNModel(config,sess)
        model_path = "CNNmodel2.ckpt"
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=3)
        #init = tf.initialize_all_variables()
        #sess.run(init)
        saver.restore(sess, model_path)
        probablityList_dev, targetList_dev, predictionList_dev, scoresList_dev = model.evaluate(sess)
        return probablityList_dev, targetList_dev, predictionList_dev, scoresList_dev

def evaluateOneByOne_CNNModel():
    config = EvaConfig()
    with tf.Graph().as_default():
        sess = tf.Session()
        initializer = tf.random_uniform_initializer(-0.1,0.1)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model = CNNModel(config,sess)
        model_path = "CNNmodel2.ckpt"
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=3)
        #init = tf.initialize_all_variables()
        #sess.run(init)
        saver.restore(sess, model_path)
        model.evaluateOneByOne(sess)

if __name__ == "__main__":
    test_CNNModel('Restart')
    #probablityList_dev_CNN, targetList_dev_CNN, predictionList_dev_CNN, scoresList_dev_CNN = evaluate_CNNModel()
    #evaluateOneByOne_CNNModel()