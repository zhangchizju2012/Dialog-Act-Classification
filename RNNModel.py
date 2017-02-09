#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 13:43:03 2017

@author: zhangchi
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

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

class RNNModel(LanguageModel):
  """Implements a Softmax classifier with cross-entropy loss."""
  def load_data(self):
    fin = open('result.txt','r')
    dictionary = {'sd':[0,1,0,0,0,0,0,0],'sv':[1,0,0,0,0,0,0,0],'aa':[0,0,1,0,0,0,0,0],'ny':[0,0,1,0,0,0,0,0],'ba':[0,0,0,1,0,0,0,0],'qy':[0,0,0,0,1,0,0,0],'qy^d':[0,0,0,0,1,0,0,0],'fc':[0,0,0,0,0,1,0,0],'qw':[0,0,0,0,0,0,1,0],'nn':[0,0,0,0,0,0,0,1]}
    x_text = []
    y = []
    count = {'sd':0,'sv':0,'aa':0}
    for line in fin:
        if line.split(':')[0] == 'ny':
            pass
        elif line.split(':')[0] in count and count[line.split(':')[0]] > 4000:
            pass
        elif line.split(':')[0] in count and count[line.split(':')[0]] <= 4000:
            x_text.append(line.split(':')[1].split('\n')[0].lower())
            y.append(dictionary[line.split(':')[0]])
            count[line.split(':')[0]] = count[line.split(':')[0]] + 1
        else:
            x_text.append(line.split(':')[1].split('\n')[0].lower())
            y.append(dictionary[line.split(':')[0]])
    self.y = np.array(y)
    del y
    self.max_document_length = max([len(x.split(" ")) for x in x_text])

    vocabulary = learn.preprocessing.CategoricalVocabulary()
    vocabulary_file = open('/Users/zhangchi/Desktop/cs690/glove/vocabulary.txt','r')
    for line in vocabulary_file:
        vocabulary.add(line.split('\n')[0])

    vocab_processor = learn.preprocessing.VocabularyProcessor(self.max_document_length,vocabulary=vocabulary)
    self.x = np.array(list(vocab_processor.fit_transform(x_text)))
    self.vocabularySize = vocabulary.__len__()
    print(str(self.max_document_length))
    del x_text
    
    # Randomly shuffle data
    np.random.seed(15)
    shuffle_indices = np.random.permutation(np.arange(len(self.y)))
    x_shuffled = self.x[shuffle_indices]
    y_shuffled = self.y[shuffle_indices]
    del self.x, self.y, shuffle_indices
    
    dev_sample_index = -1 * int(0.1 * float(len(y_shuffled)))
    self.x_train, self.x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    self.y_train, self.y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    del x_shuffled, y_shuffled
    
    print("Vocabulary Size: {:d}".format(self.vocabularySize))
    print("Train/Dev split: {:d}/{:d}".format(len(self.y_train), len(self.y_dev)))
    print("Load Data Finished")
  '''
  def load_data(self):
    fin = open('result.txt','r')
    dictionary = {'sd':[1,0,0,0,0,0,0,0],'sv':[0,1,0,0,0,0,0,0],'aa':[0,0,1,0,0,0,0,0],'ny':[0,0,1,0,0,0,0,0],'ba':[0,0,0,1,0,0,0,0],'qy':[0,0,0,0,1,0,0,0],'qy^d':[0,0,0,0,1,0,0,0],'fc':[0,0,0,0,0,1,0,0],'qw':[0,0,0,0,0,0,1,0],'nn':[0,0,0,0,0,0,0,1]}
    x_text = []
    y = []
    for line in fin:
        x_text.append(line.split(':')[1].split('\n')[0].lower())
        y.append(dictionary[line.split(':')[0]])
    self.y = np.array(y)
    del y
    self.max_document_length = max([len(x.split(" ")) for x in x_text])

    vocabulary = learn.preprocessing.CategoricalVocabulary()
    vocabulary_file = open('/Users/zhangchi/Desktop/cs690/glove/vocabulary.txt','r')
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
'''
  def add_placeholders(self):
    self.input_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.max_document_length))
    self.labels_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, 8))
    self.real_len = tf.placeholder(tf.int32, [None], name='real_len')

  def create_feed_dict(self, input_batch, label_batch, batch_real_len):
    feed_dict = {
        self.input_placeholder: input_batch,
        self.labels_placeholder: label_batch,
        self.real_len:batch_real_len}
    return feed_dict

  def add_embedding(self,is_training,sess):
    with tf.device("/cpu:0"):
        zero=np.float32((np.random.rand(1,50)-0.5)*0.2)
        last=np.float32((np.random.rand(self.vocabularySize-400001,50)-0.5)*0.2)
        W = np.concatenate((zero,np.loadtxt('/Users/zhangchi/Desktop/cs690/glove/embedding.txt',dtype=np.float32),last),axis=0)
        embedding = tf.Variable(W, name='W')
        #embedding = tf.Variable(tf.constant(0.0, shape=[self.vocabularySize, 50]),trainable=True, name="W")
        
        #embedding_placeholder = tf.placeholder(tf.float32, [self.vocabularySize, 50])
        #embedding_init = embedding.assign(embedding_placeholder)

        #sess.run(embedding_init, feed_dict={embedding_placeholder: W})
        #embedding = tf.convert_to_tensor(W)# not sure whether it can be trained or not
        del W
        inputs = tf.nn.embedding_lookup(embedding, self.input_placeholder)

    if is_training and self.config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, self.config.keep_prob)
    self.inputs = inputs
    return inputs

  def add_model(self, inputs, is_training):
      lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(50)
      if is_training and self.config.keep_prob < 1:lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.config.keep_prob)
      cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * 2)
      self.initial_state = cell.zero_state(self.config.batch_size,dtype=tf.float32)
      state = self.initial_state
      # shape: (batch_size, seq_length, cell.input_size) => (seq_length, batch_size, cell.input_size)
      inputs = tf.split(1, self.max_document_length, inputs)
      self.inputs = inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
      outputs, state = tf.nn.rnn(cell, inputs, initial_state=state, sequence_length=self.real_len)
      
      l2_loss = tf.constant(0.0)
      output = outputs[0]
      with tf.variable_scope('Output'):
		tf.get_variable_scope().reuse_variables()
		one = tf.ones([1, 50], tf.float32)
		for i in range(1,len(outputs)):
			ind = self.real_len < (i+1)
			ind = tf.to_float(ind)
			ind = tf.expand_dims(ind, -1)
			mat = tf.matmul(ind, one)
			self.output = output = tf.add(tf.mul(output, mat),tf.mul(outputs[i], 1.0 - mat))#batch_size * hidden_unit
			#only leave the last one
      with tf.name_scope('output'):
		self.W = tf.Variable(tf.truncated_normal([50, 8], stddev=0.1), name='W')
		self.b = tf.Variable(tf.constant(0.1, shape=[8]), name='b')
		l2_loss += tf.nn.l2_loss(self.W)
		l2_loss += tf.nn.l2_loss(self.b)
		self.l2_loss = l2_loss
		self.scores = tf.nn.xw_plus_b(output, self.W, self.b, name='scores')
		self.predictions = tf.argmax(self.scores, 1, name='predictions')

      with tf.name_scope('loss'):
		losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.labels_placeholder) 
		self.loss = tf.reduce_mean(losses) + 0.3 * l2_loss

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
      
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.RMSPropOptimizer(1e-3, decay=0.9)
    grads_and_vars = optimizer.compute_gradients(self.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    return train_op
    
  def run_epoch(self, session, eval_op, saver,model_path, verbose=False):
    batches = batch_iter(list(zip(self.x_train, self.y_train)), self.config.batch_size, 200)
    costs = 0.0
    l2_costs = 0.0
    iters = 0
    step = 0
    accuracyList = []
    for batch in batches:
        x, y = zip(*batch)
        fetches = [self.b,self.inputs,self.loss,self.l2_loss, self.accuracy,self.predictions, self.scores,self.output,eval_op]
        feed_dict = self.create_feed_dict(x,y,real_len(x))
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
        if step % 100 == 0:
            # dev_step
            batches_dev = batch_iter(list(zip(self.x_dev, self.y_dev)), self.config.batch_size, 1)
            costs_dev = 0.0
            # step_dev = 0
            accuracyList_dev = []
            for batch_dev in batches_dev:
                x_dev, y_dev = zip(*batch_dev)
                fetches = [self.b,self.inputs,self.loss, self.accuracy,self.predictions, self.scores,self.output]
                feed_dict = self.create_feed_dict(x_dev,y_dev,real_len(x_dev))
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

def test_RNNModel(startType='Restart'):
  """Train softmax model for a number of steps."""
  config = Config()
  if startType == 'Restart':
      with tf.Graph().as_default():
          sess = tf.Session()
          initializer = tf.random_uniform_initializer(-0.1,0.1)
          with tf.variable_scope("model", reuse=None, initializer=initializer):
              model = RNNModel(config,sess)
          model_path = "RNNmodel.ckpt"
          saver = tf.train.Saver(tf.all_variables(), max_to_keep=3)
          init = tf.initialize_all_variables()
          sess.run(init)    
          model.fit(sess,saver,model_path)
  else:
      with tf.Graph().as_default():
          sess = tf.Session()
          initializer = tf.random_uniform_initializer(-0.1,0.1)
          with tf.variable_scope("model", reuse=None, initializer=initializer):
              model = RNNModel(config,sess)
          model_path = "RNNmodel.ckpt"
          saver = tf.train.Saver(tf.all_variables(), max_to_keep=3)
          #init = tf.initialize_all_variables()
          #sess.run(init)
          saver.restore(sess, model_path)
          model.fit(sess,saver,model_path)

if __name__ == "__main__":
    losses = test_RNNModel('start')