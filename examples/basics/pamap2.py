#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mnist-convnet.py

import os
import argparse
import tensorflow as tf
import numpy as np

# Just import everything into current namespace
from tensorpack import *
from tensorpack.tfutils import summary
from tensorpack.dataflow import dataset


def batch_flatten(x):
    """
    Flatten the tensor except the first dimension.
    """
    shape = x.get_shape().as_list()[1:]
    if None not in shape:
        return tf.reshape(x, [-1, int(np.prod(shape))])
    return tf.reshape(x, tf.stack([tf.shape(x)[0], -1]))


IMAGE_SIZE = 12

class Model(ModelDesc):
    def _get_inputs(self):
        """
        Define all the inputs (with type, shape, name) that
        the graph will need.
        """
        return [InputDesc(tf.float32, (None, IMAGE_SIZE), 'input1'),
                InputDesc(tf.float32, (None, IMAGE_SIZE), 'input2'),
                InputDesc(tf.float32, (None, IMAGE_SIZE), 'input3'),
                InputDesc(tf.int32, (None,), 'label')]

    def _build_graph(self, inputs):
        """This function should build the model which takes the input variables
        and define self.cost at the end"""

        # inputs contains a list of input variables defined above
        input_from_sensor1, input_from_sensor2, input_from_sensor3, label = inputs
 

        # In tensorflow, inputs to convolution function are assumed to be
        # NHWC. Add a single channel here.
        #image = tf.expand_dims(image, 3)

        #image = image * 2 - 1   # center the pixels values at zero
        # The context manager `argscope` sets the default option for all the layers under
        # this context. Here we use 32 channel convolution with shape 3x3



        sensor1 = Sequential(input_from_sensor1) 
              .FullyConnected('fc0', 512, activation=tf.nn.relu) 
              .FullyConnected('fc1', 10, activation=tf.identity)()


        sensor2 = Sequential(input_from_sensor2) 
              .FullyConnected('fc2', 512, activation=tf.nn.relu) 
              .FullyConnected('fc3', 10, activation=tf.identity)()


        sensor3 = Sequential(input_from_sensor3) 
              .FullyConnected('fc4', 512, activation=tf.nn.relu) 
              .FullyConnected('fc5', 10, activation=tf.identity)()


        output = Connect('cloud', [sensor1, sensors2, sensor3]) 
                  .FullyConnected('fc6', 512, activation=tf.nn.relu) 
                  .FullyConnected('fc7', 10, activation=tf.identity)()




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
            decay_steps=468 * 10,
            decay_rate=0.3, staircase=True, name='learning_rate')
        # This will also put the summary in tensorboard, stat.json and print in terminal
        # but this time without moving average
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

        def read_data(pathName, sensors):
            data_dict = {}
            labels = []

            for sensor in sensors:
                data_dict[sensor] = []
                file_ = pathName + sensor + "_data.txt"
                with open(file_) as f:
                    for line in f:
                        line = line.strip().split(' ')
                        line = [np.float32(i) for i in line]
                        data_dict[sensor].append(line)

            #file_ = pathName + "act_id_labels_data.txt"
            file_ = pathName + "basic_activity_labels_data.txt"

            with open(file_) as f:
                for line in f:
                    line = line[:-1]
                    line = int(np.float32(line)) 
                    labels.append(line)

            train_data = [] 
            hand = data_dict['hand']
            chest = data_dict['chest']
            ankle = data_dict['ankle']

            for i in range(len(ankle)):
                tmp = []
                tmp = hand[i]+chest[i]+ankle[i]
                train_data.append(tmp)

            return  np.asarray(train_data), np.asarray(labels)

        sensors = ['hand','chest','ankle']
        train_path1 = '/home/henry/Desktop/Hirarchical_ML/iot/code/PAMAP2_parser/ext_features/subject101_ext_features/'
        train_path2 = '/home/henry/Desktop/Hirarchical_ML/iot/code/PAMAP2_parser/ext_features/subject102_ext_features/'
        #train_path3 = '/Users/henry/Desktop/ucsd/seelab/Hirarchical_ML/iot/code/PAMAP2_parser/subject103_ext_features/'
        train_path4 = '/home/henry/Desktop/Hirarchical_ML/iot/code/PAMAP2_parser/ext_features/subject104_ext_features/'
        train_path5 = '/home/henry/Desktop/Hirarchical_ML/iot/code/PAMAP2_parser/ext_features/subject105_ext_features/'
        train_path6 = '/home/henry/Desktop/Hirarchical_ML/iot/code/PAMAP2_parser/ext_features/subject106_ext_features/'
        train_path7 = '/home/henry/Desktop/Hirarchical_ML/iot/code/PAMAP2_parser/ext_features/subject107_ext_features/'
        train_path8 = '/home/henry/Desktop/Hirarchical_ML/iot/code/PAMAP2_parser/ext_features/subject108_ext_features/'

        train_data1, train_labels1 = read_data(train_path1, sensors)
        train_data2, train_labels2 = read_data(train_path2, sensors)
        #train_data3, train_labels3 = read_data(train_path3, sensors)
        train_data4, train_labels4 = read_data(train_path4, sensors)
        train_data5, train_labels5 = read_data(train_path5, sensors)
        train_data6, train_labels6 = read_data(train_path6, sensors)
        train_data7, train_labels7 = read_data(train_path7, sensors)
        train_data8, train_labels8 = read_data(train_path8, sensors)

        data_list = [train_data1, train_data2, train_data4, train_data5, train_data6, train_data7, train_data8]
        label_list = [train_labels1, train_labels2, train_labels4, train_labels5, train_labels6, train_labels7, train_labels8]
        train_data = np.concatenate(data_list)
        train_labels = np.concatenate(label_list)

        idx1 = np.nonzero(train_labels != 99)
        train_data = train_data[idx1[0]]
        train_labels = train_labels[idx1[0]]

        i = 2
        test_data = data_list[i]
        test_labels = label_list[i]

        train_data = np.concatenate(data_list[0:i] + data_list[i+1:])
        train_labels = np.concatenate(label_list[0:i] + label_list[i+1:])

        #train_labels =  train_labels.reshape(train_labels.shape[0],1)
        #test_labels = test_labels.reshape(test_labels.shape[0],1)


        idx1 = np.nonzero(train_labels != 99)
        train_data = train_data[idx1[0]]
        train_labels = train_labels[idx1[0]]
        idx2 = np.nonzero(test_labels != 99)
        test_data = test_data[idx2[0]]
        test_labels = test_labels[idx2[0]]

        #one_hot_train_labels = np.eye(5)[train_labels].reshape(train_labels.shape[0], 5)
        #one_hot_test_labels = np.eye(5)[test_labels].reshape(test_labels.shape[0], 5)


        if self.train_or_test == 'train':
            self.images, self.labels = train_data, train_labels
        else:
            self.images, self.labels = test_data, test_labels


    def size(self):
        return self.images.shape[0]

    def get_data(self):
        idxs = list(range(self.size()))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            img = self.images[k].reshape((36,))

            hand = img[0:12]
            chest = img[12:24]
            ankle = img[24:]

            label = self.labels[k]

            yield [hand, chest, ankle, label]



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
            MaxSaver('validation_accuracy'),  # save the model with highest accuracy (prefix 'validation_')
            InferenceRunner(    # run inference(for validation) after every epoch
                dataset_test,   # the DataFlow instance used for validation
                ScalarStats(['cross_entropy_loss', 'accuracy'])),
        ],
        steps_per_epoch=steps_per_epoch,
        max_epoch=1,
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
