#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mnist-convnet.py

import os
import argparse
import tensorflow as tf
import numpy as np


import csv
import pandas as pd
from scipy import stats

# Just import everything into current namespace
from tensorpack import *
from tensorpack.tfutils import summary
from tensorpack.dataflow import dataset

def LSTM_Network(name, _X, n_steps, n_hidden, output_num, input_dim):

    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    _X = tf.reshape(_X, [-1, input_dim])

    with tf.variable_scope(name):
        # Linear activation
        #_X = FullyConnected('fc1', _X, n_hidden, nl=tf.nn.relu, W_init=tf.truncated_normal_initializer(stddev=0.01))

        # Split data because rnn cell needs a list of inputs for the RNN inner loop
        _X = tf.split(_X, n_steps, 0)

        lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=0.5, state_is_tuple=True)
        #lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=0.5, state_is_tuple=True)
        # lstm_cell_3 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=0.5, state_is_tuple=True)
        # lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2,lstm_cell_3], state_is_tuple=True)
        lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1], state_is_tuple=True)
        # Get LSTM cell output
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

        lstm_last_output = outputs[-1]

        output = FullyConnected('fc2', lstm_last_output, output_num, nl=tf.identity, W_init=tf.truncated_normal_initializer(stddev=0.01))

    # Linear activation
    return output


class Model(ModelDesc):
    def _get_inputs(self):
        """
        Define all the inputs (with type, shape, name) that
        the graph will need.
        """

        return [InputDesc(tf.float32, (None, 25, 1), 'input1'),
                InputDesc(tf.float32, (None, 25, 17), 'input2'),
                InputDesc(tf.float32, (None, 25, 17), 'input3'),
                InputDesc(tf.float32, (None, 25, 17), 'input4'),
                InputDesc(tf.int32, (None,), 'label')]

        '''
        return [InputDesc(tf.float32,[None, 25, 52], 'input1'),
                InputDesc(tf.int32, [None, ], 'label')]
        '''
    def _build_graph(self, inputs):
        """This function should build the model which takes the input variables
        and define self.cost at the end"""

        # inputs contains a list of input variables defined above
        input1, input2, input3, input4, label = inputs

        n_hidden = 64  
        n_classes = 5
        n_steps = 25
        output_num = 4

        with tf.name_scope('sensor1'):
            output1 = LSTM_Network('sensor1', input1, n_steps, n_hidden, output_num, 1)

        with tf.name_scope('sensor2'):   
            output2 = LSTM_Network('sensor2', input2, n_steps, n_hidden, output_num, 17)

        with tf.name_scope('sensor3'):
            output3 = LSTM_Network('sensor3', input3, n_steps, n_hidden, output_num, 17)

        with tf.name_scope('sensor4'):
            output4 = LSTM_Network('sensor4', input4, n_steps, n_hidden, output_num, 17)


        logits = Connect('cloud', [output1, output2, output3, output4]) \
        .FullyConnected('fc1', 256, activation=tf.nn.relu) \
        .FullyConnected('fc2', 5, activation=tf.identity)()
    

        tf.nn.softmax(logits, name='prob')   # a Bx10 with probabilities

        # a vector of length B with loss of each sample
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')  # the average cross-entropy loss

        correct = tf.cast(tf.nn.in_top_k(logits, label, 1), tf.float32, name='correct')
        accuracy = tf.reduce_mean(correct, name='accuracy')

        # This will monitor training error (in a moving_average fashion):
        # 1. write the value to tensosrboard
        # 2. write the value to stat.json
        # 3. print the value after each epoch
        train_error = tf.reduce_mean(1 - correct, name='train_error')
        summary.add_moving_summary(train_error, accuracy)

        # Use a regex to find parameters to apply weight decay.
        # Here we apply a weight decay on all W (weight matrix) of all fc layers
        wd_cost = tf.multiply(1e-5,
                              regularize_cost('fc.*/W', tf.nn.l2_loss),
                              name='regularize_loss')

        self.cost = tf.add_n([wd_cost, cost], name='total_cost')
        summary.add_moving_summary(cost, wd_cost, self.cost)

        # monitor histogram of all weight (of conv and fc layers) in tensorboard
        summary.add_param_summary(('.*/W', ['histogram', 'rms']))

    def _get_optimizer(self):
        
        lr = tf.train.exponential_decay(
            learning_rate=1e-3,
            global_step=get_global_step_var(),
            decay_steps=1000,
            decay_rate=0.3, staircase=True, name='learning_rate')
        # This will also put the summary in tensorboard, stat.json and print in terminal
        # but this time without moving average
        
        #lr = 1e-3

        tf.summary.scalar('lr', lr)
        return tf.train.AdamOptimizer(lr)


def get_data(train_d, test_d):
    train = BatchData(train_d, 128)
    test = BatchData(test_d, 256, remainder=True)
    return train, test


class MyDataFlow(RNGDataFlow):

    def __init__(self, train_or_test, shuffle=True):

        self.train_or_test = train_or_test
        self.shuffle = shuffle
        self.files = {
                'train': ['subject101.dat', 'subject102.dat','subject103.dat','subject104.dat', 'subject107.dat', 'subject108.dat', 'subject109.dat'],
                'test': ['subject106.dat']
            }

        self.input_width = 25

        def windowz(data, size):
            start = 0
            while start < len(data):
                yield start, start + size
                start += (size / 2)

        def segment_pa2(x_train,y_train,window_size):
            segments = np.zeros(((len(x_train)//(window_size//2))-1, window_size, 52))
            labels = np.zeros(((len(y_train)//(window_size//2))-1))

            i_segment = 0
            i_label = 0

            for (start,end) in windowz(x_train,window_size):

                if(len(x_train[start:end]) == window_size):

                    m = stats.mode(y_train[start:end])
                    segments[i_segment] = x_train[start:end]
                    labels[i_label] = m[0]
                    i_label+=1
                    i_segment+=1
            return segments, labels


        def readPamap2(files, train_or_test):

            activities = ['1','2','3','4','5']
            data = readPamap2Files(files[train_or_test], activities)

            return data

        def readPamap2Files(filelist, activities):
            data = []
            labels = []
            for i, filename in enumerate(filelist):
                print('Reading file %d of %d' % (i+1, len(filelist)))
                with open('/home/henry/Desktop/Deep-Learning-for-Human-Activity-Recognition/PAMAP2_Dataset/Protocol/' + filename, 'r') as f:
                    #print "f",f
                    df = pd.read_csv(f, delimiter=' ',  header = None)
                    df = df.dropna()
                    df = df[df[1].isin(activities)]

                    df = df._get_numeric_data()

                    df = df.as_matrix()
                    data_ = df[:,2:]
                    label_ = df[:,1]

                    data = data + data_.tolist()
                    labels = labels + label_.tolist()

            return {'inputs': np.asarray(data), 'targets': np.asarray(labels, dtype=int)}

        if self.train_or_test == 'train':
            self.data = readPamap2(self.files, 'train')
            x_train = self.data['inputs']
            y_train = self.data['targets']
            y_train = y_train - 1
            x_train, y_train = segment_pa2(x_train,y_train,self.input_width)

            self.images, self.labels = x_train, y_train
        else:
            self.data = readPamap2(self.files, 'test')
            x_test = self.data['inputs']
            y_test = self.data['targets']
            y_test = y_test - 1
            x_test, y_test = segment_pa2(x_test,y_test,self.input_width)

            self.images, self.labels = x_test, y_test
        

    def size(self):
        return self.images.shape[0]

    def get_data(self):

        idxs = list(range(self.size()))

        if self.shuffle:
            self.rng.shuffle(idxs)

        for k in idxs:
            img   = self.images[k]
            heart = img[:,0].reshape((-1,1))
            hand = img[:,1:18].reshape((-1,17))
            chest = img[:,18:35].reshape((-1,17))
            ankle = img[:,35:].reshape((-1,17))
        
            label = self.labels[k]

            yield [heart, hand, chest, ankle, label]


def get_config(train_d, test_d):
    logger.auto_set_dir()

    dataset_train, dataset_test = get_data(train_d, test_d)
    # How many iterations you want in each epoch.
    # This is the default value, don't actually need to set it in the config
    steps_per_epoch = dataset_train.size()

    # get the config which contains everything necessary in a training
    return TrainConfig(
        model= Model(),
        dataflow= dataset_train,  # the DataFlow instance for training
        callbacks=[
            ModelSaver(),   # save the model after every epoch
            #MaxSaver('validation_accuracy'),  # save the model with highest accuracy (prefix 'validation_')
            #SaveSensorNetworks(["sensor1", "sensor2", "sensor3"], saving_dir = "sensors"),
            InferenceRunner(    # run inference(for validation) after every epoch
                dataset_test,   # the DataFlow instance used for validation
                ScalarStats(['cross_entropy_loss', 'accuracy'])),
        ],
        steps_per_epoch=steps_per_epoch,
        max_epoch=40,
    )


if __name__ == '__main__':

    train_d = MyDataFlow("train")
    test_d = MyDataFlow("test")

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # automatically setup the directory train_log/mnist-convnet for logging
    logger.auto_set_dir()

    config = get_config(train_d, test_d)

    if args.load:
        config.session_init = SaverRestore(args.load)
    # SimpleTrainer is slow, this is just a demo.
    # You can use QueueInputTrainer instead
    launch_train_with_config(config, SimpleTrainer())
