#!/usr/bin/python 

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

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
        _X = tf.split(_X, n_steps, 0)
        lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=0.5, state_is_tuple=True)
        lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1], state_is_tuple=True)
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)   

        lstm_last_output = outputs[-1]

        W = tf.Variable(tf.random_normal([n_hidden, output_num]))
        b = tf.Variable(tf.random_normal([output_num]))

    return tf.matmul(lstm_last_output, W) + b

def main(subject, train_data, one_hot_train_labels, \
                            test_data, one_hot_test_labels, \
                            validation_data, one_hot_validation_labels, \
                            l2_regularizer, \
                            keepprob, \
                            learning_rate, output_num):

    tf.reset_default_graph()    


    def windowz(data, step):
        start = 0
        while start < len(data):
            yield start, start + step
            start += step

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

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 25, 52])
        y_ = tf.placeholder(tf.int64, [None, 25, 1])
        keep_prob = tf.placeholder(tf.float32)

    n_hidden = 64 
    n_classes = 5
    n_steps = 25

    step = 1

    heart_output = []
    hand_output = []
    chest_output = []
    ankle_output = []







    heart_input = tf.reshape(x[:,:,0], [-1,25,1])
    heart = LSTM_Network('sensor1', heart_input, n_steps, n_hidden, output_num, 1)
    hand  = LSTM_Network('sensor2', x[:,:,1:18], n_steps, n_hidden, output_num, 17)
    chest = LSTM_Network('sensor3', x[:,:,18:35], n_steps, n_hidden, output_num, 17)
    ankle = LSTM_Network('sensor4', x[:,:,35:], n_steps, n_hidden, output_num, 17)

    outputs = []    
    for step in range(n_steps):
        local_output = []
        local_output.append(heart[step])
        local_output.append(hand[step])
        local_output.append(chest[step])
        local_output.append(ankle[step])
        outputs.append(tf.concat(local_output, axis = 1))

    outputs = tf.convert_to_tensor(outputs)
    print outputs

    outputs = tf.transpose(outputs, [1, 0, 2])  # permute n_steps and batch_size 

    all_output = LSTM_Network('cloud', outputs, n_steps, n_hidden, n_classes, 4*output_num)

    all_output = tf.convert_to_tensor(all_output)

    batch_size = 256
    training_epochs = 150
    output = tf.reshape(all_output, [-1, 5])
    
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(y_, [-1,]),
                                                                logits = output))   
        #l2_loss = HNN.gather_l2_loss(tf.get_default_graph())
        #l2_loss = tf.reduce_sum(tf.stack(l2_loss))
        total_loss = cross_entropy  #+ l2_regularizer * l2_loss

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.reshape(y_, [-1,]))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # create a summary for our cost and accuracy
        tf.summary.scalar("cost", cross_entropy)
        tf.summary.scalar("accuracy", accuracy)
        batch_count = train_data.shape[0] / batch_size

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

            loss = 0.0
            mean_acc = 0.0
            
            acc, train_loss = sess.run((accuracy, total_loss),
                    feed_dict={x: X_random, y_: Y_random, keep_prob: 1.0})
            print train_loss
            

            for i in range(batch_count):
                train_data_batch = X_random[i * batch_size: (i+1) * batch_size, :,:]
                train_label_batch = Y_random[i * batch_size: (i+1) * batch_size, :,:]
                _ = sess.run([train_step], feed_dict={x: train_data_batch, y_: train_label_batch, keep_prob: keepprob})
    
                acc, train_loss = sess.run((accuracy, total_loss),
                        feed_dict={x: train_data_batch, y_: train_label_batch, keep_prob: 1.0})
                

                '''
                print one_hot_validation_labels.shape
                one_hot_validation_labels = one_hot_validation_labels.reshape((-1, ))
                val_loss = sess.run((total_loss),
                        feed_dict={x: validation_data, y_: one_hot_validation_labels, keep_prob: 1.0})

                if validation_loss_last_epoch == None:
                    validation_loss_last_epoch = val_loss

                one_hot_test_labels = one_hot_test_labels.reshape((-1,))
                    test_accuracy, test_loss = sess.run((accuracy, total_loss),
                        feed_dict={x: test_data, y_: one_hot_test_labels, keep_prob: 1.0})

                    if last_test_accuracy == None:
                        last_test_accuracy = test_accuracy

                else:
                    if val_loss < validation_loss_last_epoch:
                        cnt = 5

                        validation_loss_last_epoch = val_loss
                        test_accuracy, test_loss = sess.run((accuracy, total_loss),
                            feed_dict={x: test_data, y_: one_hot_test_labels, keep_prob: 1.0})

                        if last_test_accuracy < test_accuracy:
                            last_test_accuracy = test_accuracy
                    else:
                        cnt = cnt - 1

                        validation_loss_last_epoch = val_loss

                        test_accuracy, test_loss = sess.run((accuracy, total_loss),
                            feed_dict={x: test_data, y_: one_hot_test_labels, keep_prob: 1.0})

                        if last_test_accuracy < test_accuracy:
                            last_test_accuracy = test_accuracy

                        if cnt == 0:
                            break

                f = open("./summary_both_local_and_cloud_lstm.txt", "a+")
                if subject == 6:
                    f.write(str(last_test_accuracy) + "]\n\n")
                else:
                    f.write(str(last_test_accuracy) + " ")
                f.close()
                '''

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

    input_width = 25
    window_size = 25
    step = 1

    def windowz(data, step):
        start = 0
        while start < len(data):
            yield start, start + step
            start += step

    def windowz_one_step(data, step):
        start = 0
        while start < data.shape[0]:

            yield start, start + step
            start += step

    def segment_pa2(x_train,y_train,window_size):
        segments = np.zeros(( len(x_train) - window_size + step, window_size, 52))

        labels = np.zeros((len(x_train) - window_size + step))

        print segments.shape
        print labels.shape

        i_segment = 0
        i_label = 0

        for (start, end) in windowz(x_train, 1):
            if(len(x_train[start:end]) == window_size):

                segments[i_segment] = x_train[start:end]
                labels[i_label] = y_train[end]
                i_label+=1
                i_segment+=1

        segments_one_step  = np.zeros(( segments.shape[0]-1, window_size, window_size, 52))
        labels_one_step =  np.zeros(( labels.shape[0]-1))

        i_segment = 0
        i_label = 0

        for (start, end) in windowz_one_step(segments, 1):
            if(len(segments[start:end]) == window_size):
                segments_one_step[i_segment] = segments[start:end]
                labels_one_step[i_label] = labels[start]

                i_segment += 1
                i_label += 1

        print segments_one_step.shape
        print labels_one_step.shape

        return segments_one_step, labels_one_step


    l2_regularizers = [0.01, 0.1, 1]
    keep_probs = [0.5]
    learning_rates = [0.001, 0.01, 0.1]
    output_nums = [10]

    train_data = np.concatenate(data_list)
    train_labels = np.concatenate(label_list)
    train_labels =  train_labels.reshape(train_labels.shape[0],1)
    train_data, train_labels = segment_pa2( train_data, train_labels, input_width)
    
    '''
    for learning_rate in learning_rates:
        for l2 in l2_regularizers:
            for keep_prob in keep_probs:
                for output_num in output_nums:

                    f = open("./summary_both_local_and_cloud_lstm.txt", "a+")

                    f.write("l2 = " + str(l2) \
                        + " " + "keep_prob = " + str(keep_prob) \
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

                        train_data, train_labels = segment_pa2( train_data,train_labels, input_width)
                        test_data, test_labels = segment_pa2(test_data,test_labels, input_width)
                        validation_data, validation_labels = segment_pa2(validation_data, validation_labels, input_width)

              
                        train_labels = train_labels.astype(int)
                        test_labels = test_labels.astype(int)
                        validation_labels = validation_labels.astype(int)
                 
                        main(i, train_data, train_labels, \
                            test_data, test_labels, \
                            validation_data, validation_labels, \
                            l2, \
                            keep_prob, \
                            learning_rate, output_num)
                          
    '''
        
