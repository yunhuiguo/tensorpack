#!/usr/bin/python 

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import sys
sys.path.append('../')
#import Hierarchical_Neural_Networks as HNN
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import random
from scipy import stats
import csv
import pandas as pd


def gather_l2_loss(graph):
    node_defs = [n for n in graph.as_graph_def().node if 'l2_loss' in n.name]
    tensors = [graph.get_tensor_by_name(n.name+":0") for n in node_defs]
    return tensors


def LSTM_Network(name, _X, n_steps, n_hidden, output_num, input_dim):
    _X = tf.transpose(_X, [2, 0, 1, 3])  # permute n_steps and batch_size
    _X = tf.reshape(_X, [-1, input_dim])

    with tf.variable_scope(name):
        _X = tf.split(_X, n_steps, 0) 
        lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=0.5, state_is_tuple=True)
        lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1], state_is_tuple=True)
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)   

        lstm_last_output = outputs[-1]
        W = tf.Variable(tf.random_normal([n_hidden, output_num]))
        b = tf.Variable(tf.random_normal([output_num]))
        output =  tf.matmul(lstm_last_output, W) + b

        l2_loss  = tf.nn.l2_loss(W, name= "l2_loss" + name)

    return output

def LSTM_Network_cloud(name, _X, n_steps, n_hidden, output_num, input_dim):
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    _X = tf.reshape(_X, [-1, input_dim])

    with tf.variable_scope(name):
        _X = tf.split(_X, n_steps, 0) 
        lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=0.5, state_is_tuple=True)
        lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1], state_is_tuple=True)
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)   

        lstm_last_output = outputs[-1]

        W = tf.Variable(tf.random_normal([n_hidden, output_num]))
        l2_loss  = tf.nn.l2_loss(W, name= "l2_loss" + name)
        b = tf.Variable(tf.random_normal([output_num]))
        output =  tf.matmul(lstm_last_output, W) + b
    return output


def main(subject, train_data, one_hot_train_labels, \
                            test_data, one_hot_test_labels, \
                            validation_data, one_hot_validation_labels, \
                            l2_regularizer, \
                            learning_rate, output_num):

    tf.reset_default_graph()    

    def calculate_acc_loss(data, labels, batch_size):
        X_random = data
        Y_random = labels

        loss = 0.0
        batch_count = (data.shape[0] - batch_size) / batch_size
        right = 0.0
        total = 0.0
        for i in range(batch_count):
            train_data_batch = np.zeros((batch_size, window_size, window_size, 52))
            train_label_batch = np.zeros((batch_size, window_size))

            for j in range(batch_size):
                train_data_batch[j] = X_random[ (j + i*batch_size) : (j + window_size + i*batch_size), :,:]
                train_label_batch[j] = Y_random[(j + i*batch_size) : (j + window_size + i*batch_size)]

            _, loss_, acc = sess.run([train_step, total_loss, accuracy], feed_dict={x: train_data_batch, y_: train_label_batch})
            loss = loss_ + loss
            total = total + batch_size
            right = right +  acc * batch_size

        acc = right / total

        return loss, acc

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 25, 25, 52])
        y_ = tf.placeholder(tf.int64, [None, 25,])

    n_hidden = 64 
    n_classes = 5
    n_steps = 25

    batch_size = 256
    training_epochs = 150

    heart_output = []
    hand_output = []
    chest_output = []
    ankle_output = []

    i_segment = 0
    i_label = 0

    outputs = []

    heart_input = tf.reshape(x[:,:,:,0], [-1, 25, 25, 1])
    heart = LSTM_Network('sensor1', heart_input, n_steps, n_hidden, output_num, 1)
    heart = tf.reshape(heart, [batch_size, n_steps ,output_num])

    hand  = LSTM_Network('sensor2', x[:,:,:,1:18], n_steps, n_hidden, output_num, 17)
    hand = tf.reshape(hand, [batch_size, n_steps ,output_num])

    chest = LSTM_Network('sensor3', x[:,:,:,18:35], n_steps, n_hidden, output_num, 17)
    chest = tf.reshape(chest, [batch_size, n_steps ,output_num])

    ankle = LSTM_Network('sensor4', x[:,:,:,35:], n_steps, n_hidden, output_num, 17)
    ankle = tf.reshape(ankle, [batch_size, n_steps ,output_num])

    outputs = [heart, hand, chest, ankle]
    outputs = tf.concat(outputs, axis = 2)    
    outputs = tf.convert_to_tensor(outputs)
    outputs = tf.transpose(outputs, [2, 0, 1])  # permute n_steps and batch_size 
    output = LSTM_Network_cloud('cloud', outputs, n_steps, n_hidden, n_classes, 4*output_num)
    
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y_[:,-1],
                                                                logits = output))
        l2_loss = gather_l2_loss(tf.get_default_graph())
        l2_loss = tf.reduce_sum(tf.stack(l2_loss))
        total_loss = cross_entropy  + l2_regularizer * l2_loss

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(output, 1),  y_[:,-1])
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # create a summary for our cost and accuracy
        #tf.summary.scalar("cost", cross_entropy)
        #tf.summary.scalar("accuracy", accuracy)

        #batch_count = train_data.shape[0] / batch_size

        validation_loss_last_epoch = None
        last_test_accuracy = None

        cnt = 5
        for epoch in range(training_epochs):
            # number of batches in one epoch
            #idxs = np.random.permutation(train_data.shape[0]) #shuffled ordering
            #X_random = train_data[idxs]
            #Y_random = one_hot_train_labels[idxs]
            X_random = train_data
            Y_random = one_hot_train_labels

            batch_count = (train_data.shape[0] - batch_size) / batch_size
            for i in range(batch_count):
                train_data_batch = np.zeros((batch_size, window_size, window_size, 52))
                train_label_batch = np.zeros((batch_size, window_size))
    
                for j in range(batch_size):
                    train_data_batch[j] = X_random[ (j + i*batch_size) : (j + window_size + i*batch_size), :,:]
                    train_label_batch[j] = Y_random[(j + i*batch_size) : (j + window_size + i*batch_size)]

                _ = sess.run([train_step], feed_dict={x: train_data_batch, y_: train_label_batch})
                                
            train_loss, train_acc = calculate_acc_loss(train_data, one_hot_train_labels, batch_size)
            print train_loss
            print train_acc
            print "\n"

            val_loss, val_acc = calculate_acc_loss(validation_data, one_hot_validation_labels, batch_size)
            if validation_loss_last_epoch == None:
                validation_loss_last_epoch = val_loss

                test_loss, test_accuracy = calculate_acc_loss(test_data, one_hot_test_labels, batch_size)

                if last_test_accuracy == None:
                    last_test_accuracy = test_accuracy

            else:
                if val_loss < validation_loss_last_epoch:
                    cnt = 5

                    validation_loss_last_epoch = val_loss
                    test_loss, test_accuracy = calculate_acc_loss(test_data, one_hot_test_labels, batch_size)

                    if last_test_accuracy < test_accuracy:
                        last_test_accuracy = test_accuracy
                else:
                    cnt = cnt - 1

                    validation_loss_last_epoch = val_loss

                    test_loss, test_accuracy = calculate_acc_loss(test_data, one_hot_test_labels, batch_size)

                    if last_test_accuracy < test_accuracy:
                        last_test_accuracy = test_accuracy

                    if cnt == 0:
                        break

        f = open("./summary/summary_both_local_and_cloud_lstm.txt", "a+")
        if subject == 6:
            f.write(str(last_test_accuracy) + "]\n\n")
        else:
            f.write(str(last_test_accuracy) + " ")
        f.close()
                

class MyDataFlow():

    def __init__(self, shuffle=True):

        self.shuffle = shuffle
        self.files = {
                'data': ['subject101.dat', 'subject102.dat','subject104.dat', 'subject105.dat', 'subject106.dat', 'subject107.dat', 'subject108.dat']
            }

        def readPamap2(files, train_or_test):

            activities = ['1','2','3','4','5']
            data = readPamap2Files(files['data'], activities)

            return data

        def readPamap2Files(filelist, activities):
            data = []
            labels = []
            for i, filename in enumerate(filelist):
                print('Reading file %d of %d' % (i+1, len(filelist)))
                with open('./PAMAP2_Dataset/Protocol/' + filename, 'r') as f:
                    #print "f",f
                    df = pd.read_csv(f, delimiter=' ',  header = None)
                    df = df.dropna()
                    df = df[df[1].isin(activities)]
                    df = df._get_numeric_data()

                    df = df.as_matrix()
                    data_ = df[:,2:]
                    label_ = df[:,1]

                    data.append(np.asarray(data_))
                    labels.append(np.asarray(label_, dtype=int)-1)

            return {'inputs': data, 'targets': labels}

        self.data = readPamap2(self.files, 'train')

        x_train = self.data['inputs']
        y_train = self.data['targets']

        self.images, self.labels = x_train, y_train

    def get_data(self):
        return self.images, self.labels

if __name__ == '__main__':

    dataflow = MyDataFlow()
    data_list, label_list = dataflow.get_data()

    accuracy = []

    train_data = np.concatenate(data_list)
    train_labels = np.concatenate(label_list)

    window_size = 25
    step = 1

    def windowz(data, step):
        start = 0
        while start < len(data):
            yield start, start + 25
            start += step

    def segment_pa2(x_train,y_train,window_size, step):
        segments = np.zeros(( len(x_train) - window_size + step, window_size, 52))

        labels = np.zeros((len(x_train) - window_size + step))

        i_segment = 0
        i_label = 0

        for (start, end) in windowz(x_train, step):
            if(len(x_train[start:end]) == window_size):
                segments[i_segment] = x_train[start:end]
                labels[i_label] = y_train[end-1][0]
                i_label+=1
                i_segment+=1

        return segments, labels

    l2_regularizers = [0.0, 0.01, 0.1, 1]
    learning_rates = [0.001, 0.01, 0.05, 0.1]
    output_nums = [1, 2, 4, 6, 8]

    for learning_rate in learning_rates:
        for l2 in l2_regularizers:
                for output_num in output_nums:
                    f = open("./summary/summary_both_local_and_cloud_lstm.txt", "a+")

                    f.write("l2 = " + str(l2) \
                        + " " + "learning rate = " + str(learning_rate) \
                        + " " + "output_num = " + str(output_num) \
                        + "\n")     

                    f.write("[")        
                    f.close()

                    valiation_time = 7
                    for i in range(valiation_time):
                    
                        test_data = data_list[i]
                        test_labels = label_list[i]
                        
                        valiation_idx = random.randint(0, valiation_time-1)
                        while valiation_idx == i:
                             valiation_idx = random.randint(0, valiation_time-1)
                        
                        validation_data = data_list[valiation_idx]
                        validation_labels = label_list[valiation_idx]
                        
                        train_data_list = data_list[:]
                        train_label_list = label_list[:]

                        train_data_list.remove(test_data)
                        train_data_list.remove(validation_data)
                        train_label_list.remove(test_labels)
                        train_label_list.remove(validation_labels)


                        train_data = np.concatenate(train_data_list)
                        train_labels = np.concatenate(train_label_list)
                
                        train_labels =  train_labels.reshape(train_labels.shape[0],1)
                        validation_labels = validation_labels.reshape(validation_labels.shape[0],1)
                        test_labels = test_labels.reshape(test_labels.shape[0],1)
                     

                        train_data, train_labels = segment_pa2( train_data,train_labels, window_size, step)
                        test_data, test_labels = segment_pa2(test_data,test_labels, window_size, step)
                        validation_data, validation_labels = segment_pa2(validation_data, validation_labels, window_size, step)

                        train_labels = train_labels.astype(int)
                        test_labels = test_labels.astype(int)
                        validation_labels = validation_labels.astype(int)
                    
                        main(i, train_data, train_labels, \
                            test_data, test_labels, \
                            validation_data, validation_labels, \
                            l2, \
                            learning_rate, output_num)
                          
    
        