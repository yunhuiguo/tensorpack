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
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

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

    for i in range(len(hand)):
        tmp = []
        tmp = hand[i]+chest[i]+ankle[i]
        train_data.append(tmp)

    return  np.asarray(train_data), np.asarray(labels)

def main(subject, train_data, one_hot_train_labels, \
                            test_data, one_hot_test_labels, \
                            validation_data, one_hot_validation_labels, \
                            l2_regularizer, \
                            keepprob, \
                            learning_rate):

    tf.reset_default_graph()    
    train_accuracy_batches = []
    test_accuracy_batches = []

    with tf.name_scope('input'):
        #x = tf.placeholder(tf.float32, [None, 76])
        x = tf.placeholder(tf.float32, [None, 666])
        y_ = tf.placeholder(tf.float32, [None, 5])
        keep_prob = tf.placeholder(tf.float32)


    model = HNN.LocalSensorNetwork("fully_connected_model", x, [5], l2_regularizer=l2_regularizer, keep_prob=keepprob)
    output =  model.build_layers()

    training_epochs = 20
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

        # merge ext summaries into a single "operation" which we can execute in a session 
        #summary_op = tf.summary.merge_ext()
        #writer = tf.summary.FileWriter('./tmp/tensorflow_logs', graph=sess.graph)
        batch_count = train_data.shape[0] / batch_size

        validation_loss_last_epoch = None
        last_test_accuracy = None

        for epoch in range(training_epochs):
            # number of batches in one epoch
            idxs = np.random.permutation(train_data.shape[0]) #shuffled ordering
            X_random = train_data[idxs]
            Y_random = one_hot_train_labels[idxs]

            for i in range(batch_count):
                train_data_batch = X_random[i * batch_size: (i+1) * batch_size,:]
                train_label_batch = Y_random[i * batch_size: (i+1) * batch_size,:]

                _ = sess.run([train_step], feed_dict={x: train_data_batch, y_: train_label_batch, keep_prob: keepprob})
                # write log
                #writer.add_summary(summary, epoch * batch_count + i)

            val_loss = sess.run((total_loss),
                    feed_dict={x: validation_data, y_: one_hot_validation_labels, keep_prob: 1.0})

            if validation_loss_last_epoch == None:
                validation_loss_last_epoch = val_loss

                test_accuracy, test_loss = sess.run((accuracy, total_loss),
                    feed_dict={x: test_data, y_: one_hot_test_labels, keep_prob: 1.0})
                if last_test_accuracy == None:
                    last_test_accuracy = test_accuracy

            else:
                if val_loss < validation_loss_last_epoch:
                    validation_loss_last_epoch = val_loss
                    test_accuracy, test_loss = sess.run((accuracy, total_loss),
                        feed_dict={x: test_data, y_: one_hot_test_labels, keep_prob: 1.0})

                    last_test_accuracy = test_accuracy
                else:
                    break

        f = open("./summary_interaction_elastic_net.txt", "a+")
        if i == 6:
            f.write(str(test_accuracy) + "]\n\n")
        else:
            f.write(str(test_accuracy) + " ")
        f.close()

if __name__ == '__main__':
    sensors = ['hand', 'chest', 'ankle']

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

    accuracy = []
    data_list = [train_data1, train_data2, train_data4, train_data5, train_data6, train_data7, train_data8]
    label_list = [train_labels1, train_labels2, train_labels4, train_labels5, train_labels6, train_labels7, train_labels8]
    train_data = np.concatenate(data_list)
    train_labels = np.concatenate(label_list)

    idx1 = np.nonzero(train_labels != 99)
    #train_data = train_data[idx1[0]]
    train_labels = train_labels[idx1[0]]
    print train_data.shape



    #l2_regularizers = [0.001, 0.005 , 0.01, 0.05, 0.1, 0.5, 1]
    #connection_nums = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    #keep_probs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    #learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]


    '''
    for learning_rate in learning_rates:
        for l2 in l2_regularizers:
            for keep_prob in keep_probs:
            
                f = open("./summary_elastic_net.txt", "a+")

                f.write("l2 = " + str(l2) \
                    + " " + "keep_prob = " + str(keep_prob) \
                    + " " + "learning rate = " + str(learning_rate) \
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


                  
                    train_labels[train_labels == 5] = 8
                    train_labels[train_labels == 13] = 9
                    train_labels[train_labels == 16] = 10
                    train_labels[train_labels == 17] = 11
                    train_labels[train_labels == 24] = 5

                    test_labels[test_labels == 5] = 8
                    test_labels[test_labels == 13] = 9
                    test_labels[test_labels == 16] = 10
                    test_labels[test_labels == 17] = 11
                    test_labels[test_labels == 24] = 5

                    train_labels = train_labels - 1
                    test_labels = test_labels  - 1
                   

                    idx1 = np.nonzero(train_labels != 99)
                    train_data = train_data[idx1[0]]
                    train_labels = train_labels[idx1[0]]

                    idx2 = np.nonzero(test_labels != 99)
                    test_data = test_data[idx2[0]]
                    test_labels = test_labels[idx2[0]]


                    idx3 = np.nonzero(validation_labels != 99)
                    validation_data = validation_data[idx3[0]]
                    validation_labels = validation_labels[idx3[0]]


                    one_hot_train_labels = np.eye(5)[train_labels].reshape(train_labels.shape[0], 5)
                    one_hot_test_labels = np.eye(5)[test_labels].reshape(test_labels.shape[0], 5)
                    one_hot_validation_labels = np.eye(5)[validation_labels].reshape(validation_labels.shape[0], 5)

                        regr = ElasticNet(random_state=0)
                        regr.fit(X, y)


                    main(i, train_data, one_hot_train_labels, \
                        test_data, one_hot_test_labels, \
                        validation_data, one_hot_validation_labels, \
                        l2, \
                        keep_prob, \
                        learning_rate)
            

                    #accuracy.append(test_accuracy)

                #print sum(accuracy)/(len(accuracy) + 0.0)
    '''

    l2_regularizers = [0.0, 0.005, 0.05, 0.1, 0.5, 1]
    l1_regularizers = [0.0, 0.005, 0.05, 0.1, 0.5, 1]

    for l1 in l1_regularizers:
        for l2 in l2_regularizers:
            
            f = open("./summary_interaction_elastic_net.txt", "a+")

            f.write("l1 = " + str(l1) \
                + " " + "l2 = " + str(l2) \
                + "\n")     

            f.write("[")        
            f.close()
            
            valiation_time = 7
            for i in range(valiation_time):
                
                test_data = data_list[i]
                test_labels = label_list[i]

                #valiation_idx = random.randint(0, valiation_time-1)
                #while valiation_idx == i:
                #     valiation_idx = random.randint(0, valiation_time-1)

                #validation_data = data_list[valiation_idx]
                #validation_labels = label_list[valiation_idx]

                train_data_list = data_list[:]
                train_label_list = label_list[:]

                train_data_list.remove(test_data)
                #train_data_list.remove(validation_data)
                train_label_list.remove(test_labels)
                #train_label_list.remove(validation_labels)


                train_data = np.concatenate(train_data_list)
                train_labels = np.concatenate(train_label_list)

                train_labels =  train_labels.reshape(train_labels.shape[0],1)
                #validation_labels = validation_labels.reshape(validation_labels.shape[0],1)
                test_labels = test_labels.reshape(test_labels.shape[0],1)

                '''
                train_labels[train_labels == 5] = 8
                train_labels[train_labels == 13] = 9
                train_labels[train_labels == 16] = 10
                train_labels[train_labels == 17] = 11
                train_labels[train_labels == 24] = 5

                test_labels[test_labels == 5] = 8
                test_labels[test_labels == 13] = 9
                test_labels[test_labels == 16] = 10
                test_labels[test_labels == 17] = 11
                test_labels[test_labels == 24] = 5
                '''
                #train_labels = train_labels - 1
                #test_labels = test_labels  - 1
               

                idx1 = np.nonzero(train_labels != 99)
                train_data = train_data[idx1[0]]
                train_labels = train_labels[idx1[0]]

                idx2 = np.nonzero(test_labels != 99)
                test_data = test_data[idx2[0]]
                test_labels = test_labels[idx2[0]]

                '''
                idx3 = np.nonzero(validation_labels != 99)
                validation_data = validation_data[idx3[0]]
                validation_labels = validation_labels[idx3[0]]
                '''
                #one_hot_train_labels = np.eye(5)[train_labels].reshape(train_labels.shape[0], 5)
                #one_hot_test_labels = np.eye(5)[test_labels].reshape(test_labels.shape[0], 5)
                #one_hot_validation_labels = np.eye(5)[validation_labels].reshape(validation_labels.shape[0], 5)
                train_labels = train_labels.ravel()
                test_labels = test_labels.ravel()

                if l2 == 0.0 and l2 == 0.0:
                    clf = linear_model.LogisticRegression(penalty='l2', C = 1e42, solver='sag', max_iter =100, multi_class= 'multinomial')

                elif l1 == 0.0 and l2 != 0.0:
                    clf = linear_model.LogisticRegression(penalty='l2', C = 1/l2, solver='sag', max_iter =100, multi_class= 'multinomial')

                elif l1 != 0.0 and l2 == 0.0:
                    clf = linear_model.LogisticRegression(penalty='l1', C = 1/l1, solver='sag', max_iter =100, multi_class= 'multinomial')
                else:
                    clf = linear_model.SGDClassifier(loss='log', penalty='elasticnet', alpha= l1+l2 , l1_ratio=l1/(l1+l2), max_iter=100)
                

                poly = PolynomialFeatures(2, interaction_only = True, include_bias=False)             
                train_data = poly.fit_transform(train_data)
                print train_data.shape
            
                clf.fit(train_data, train_labels)

                test_data = poly.fit_transform(test_data)
                test_accuracy = clf.score(test_data, test_labels)

                f = open("./summary_interaction_elastic_net.txt", "a+")
                if i == 6:
                    f.write(str(test_accuracy) + "]\n\n")
                else:
                    f.write(str(test_accuracy) + " ")
                f.close()