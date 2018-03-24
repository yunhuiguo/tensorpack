#!/usr/bin/python 

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import sys
sys.path.append('../')
import Hierarchical_Neural_Networks as HNN
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import random
from scipy import stats
import csv
import pandas as pd

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

        W = tf.Variable(tf.random_normal([n_hidden, output_num]))
        b = tf.Variable(tf.random_normal([output_num]))

        output = tf.matmul(lstm_last_output, W) + b

    # Linear activation
    return output


def main(subject, train_data, one_hot_train_labels, \
                            test_data, one_hot_test_labels, \
                            validation_data, one_hot_validation_labels, \
                            l2_regularizer, \
                            learning_rate, output_num):

    tf.reset_default_graph()	
    train_accuracy_batches = []
    test_accuracy_batches = []

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 25, 52])
        y_ = tf.placeholder(tf.float32, [None, 5])


    n_hidden = 64  
    n_classes = 5
    n_steps = 25
    

    '''
    heart = img[:,0].reshape((-1,1))
    hand = img[:,1:18].reshape((-1,17))
    chest = img[:,18:35].reshape((-1,17))
    ankle = img[:,35:].reshape((-1,17))
    '''
    
    heart_input = tf.reshape(x[:,:,0], [-1,25,1])
    heart = LSTM_Network('sensor1', heart_input, n_steps, n_hidden, output_num, 1)
    hand = LSTM_Network('sensor2', x[:,:,1:18], n_steps, n_hidden, output_num, 17)
    chest = LSTM_Network('sensor3', x[:,:,18:35], n_steps, n_hidden, output_num, 17)
    ankle = LSTM_Network('sensor4', x[:,:,35:], n_steps, n_hidden, output_num, 17)

    cloud = HNN.CloudNetwork("cloud", [256, 5], l2_regularizer=l2_regularizer)
    output = cloud.connect([heart,hand, chest, ankle])

    training_epochs = 150
    batch_size = 256
    
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                logits=output))   
        
        l2_loss = HNN.gather_l2_loss(tf.get_default_graph())
        l2_loss = tf.reduce_sum(tf.stack(l2_loss))
        total_loss = cross_entropy + l2_regularizer * l2_loss


    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # create a summary for our cost and accuracy
        tf.summary.scalar("cost", cross_entropy)
        tf.summary.scalar("accuracy", accuracy)

        # merge raw summaries into a single "operation" which we can execute in a session 
        #summary_op = tf.summary.merge_raw()
        #writer = tf.summary.FileWriter('./tmp/tensorflow_logs', graph=sess.graph)
        batch_count = train_data.shape[0] / batch_size

        validation_loss_last_epoch = None
        last_test_accuracy = None

        cnt = 5

        for epoch in range(training_epochs):
            # number of batches in one epoch
            idxs = np.random.permutation(train_data.shape[0]) #shuffled ordering
            X_random = train_data[idxs]
            Y_random = one_hot_train_labels[idxs]

            for i in range(batch_count):
                train_data_batch = X_random[i * batch_size: (i+1) * batch_size,:]
                train_label_batch = Y_random[i * batch_size: (i+1) * batch_size,:]

                _ = sess.run([train_step], feed_dict={x: train_data_batch, y_: train_label_batch})
                # write log
                #writer.add_summary(summary, epoch * batch_count + i)

            val_loss = sess.run((total_loss),
                    feed_dict={x: validation_data, y_: one_hot_validation_labels})

            if validation_loss_last_epoch == None:
                validation_loss_last_epoch = val_loss

                test_accuracy, test_loss = sess.run((accuracy, total_loss),
                    feed_dict={x: test_data, y_: one_hot_test_labels})

                if last_test_accuracy == None:
                    last_test_accuracy = test_accuracy

            else:
                if val_loss < validation_loss_last_epoch:
                    cnt = 5

                    validation_loss_last_epoch = val_loss
                    test_accuracy, test_loss = sess.run((accuracy, total_loss),
                        feed_dict={x: test_data, y_: one_hot_test_labels})

                    if last_test_accuracy < test_accuracy:
                        last_test_accuracy = test_accuracy
                else:
                    cnt = cnt - 1

                    validation_loss_last_epoch = val_loss

                    test_accuracy, test_loss = sess.run((accuracy, total_loss),
                        feed_dict={x: test_data, y_: one_hot_test_labels})

                    if last_test_accuracy < test_accuracy:
                        last_test_accuracy = test_accuracy

                    if cnt == 0:
                        break

        f = open("./summary/summary_local_lstm_cloud_mlp.txt", "a+")
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

        self.input_width = 25

        def readPamap2(files, train_or_test):

            activities = ['1','2','3','4','5']
            data = readPamap2Files(files['data'], activities)

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

                    data.append(np.asarray(data_))
                    labels.append(np.asarray(label_, dtype=int)-1)

            return {'inputs': data, 'targets': labels}

        self.data = readPamap2(self.files, 'train')

        x_train = self.data['inputs']
        y_train = self.data['targets']

        #x_train, y_train = segment_pa2(x_train,y_train, 25)

        self.images, self.labels = x_train, y_train

    def get_data(self):
        return self.images, self.labels


if __name__ == '__main__':

    dataflow = MyDataFlow()
    data_list, label_list = dataflow.get_data()

    accuracy = []

    train_data = np.concatenate(data_list)
    train_labels = np.concatenate(label_list)

    input_width = 25
    window_size = 25

    idx1 = np.nonzero(train_labels != 99)
    train_data = train_data[idx1[0]]
    train_labels = train_labels[idx1[0]]

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

    l2_regularizers = [0.0, 0.01, 0.1, 1]
    learning_rates = [0.001, 0.01, 0.05, 0.1]
    output_nums = [1, 2, 4, 6, 8]


    for learning_rate in learning_rates:
        for l2 in l2_regularizers:
                for output_num in output_nums:
                    f = open("./summary/summary_local_lstm_cloud_mlp.txt", "a+")

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


                        idx1 = np.nonzero(train_labels != 99)
                        train_data = train_data[idx1[0]]
                        train_labels = train_labels[idx1[0]]

                        idx2 = np.nonzero(test_labels != 99)
                        test_data = test_data[idx2[0]]
                        test_labels = test_labels[idx2[0]]


                        idx3 = np.nonzero(validation_labels != 99)
                        validation_data = validation_data[idx3[0]]
                        validation_labels = validation_labels[idx3[0]]


                        train_data, train_labels = segment_pa2( train_data,train_labels, input_width)
                        test_data, test_labels = segment_pa2(test_data,test_labels, input_width)
                        validation_data, validation_labels = segment_pa2(validation_data, validation_labels, input_width)

                        train_labels = train_labels.astype(int)
                        test_labels = test_labels.astype(int)
                        validation_labels = validation_labels.astype(int)


                        one_hot_train_labels = np.eye(5)[train_labels].reshape(train_labels.shape[0], 5)
                        one_hot_test_labels = np.eye(5)[test_labels].reshape(test_labels.shape[0], 5)
                        one_hot_validation_labels = np.eye(5)[validation_labels].reshape(validation_labels.shape[0], 5)

        
                        main(i, train_data, one_hot_train_labels, \
                            test_data, one_hot_test_labels, \
                            validation_data, one_hot_validation_labels, \
                            l2, \
                            learning_rate, output_num)
    

        