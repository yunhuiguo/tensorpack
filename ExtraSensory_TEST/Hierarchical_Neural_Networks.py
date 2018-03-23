"""
Hirarchical Neural Networks
"""
# Author: Yunhui Guo <yug185@eng.ucsd.edu>

import tensorflow as tf
import numpy as np

class  Network(object):
    """
        The Network class 

    """
    def __init__(self, name, hidden_layers, activation_fn=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer):
        """
            Initilize network

            Args:
                name: name of the network
                hidden_layers: a 1-D tensor indicates the number of hidden neurons in each layer, the last element is the number of output
                activation_fn: activation function 
                initializer: initialization methods

            Returns:
        """

        self.name = name
        self.hidden_layers = hidden_layers    
        self.activation_fn = activation_fn
        self.initializer = initializer

    def one_fully_connected_layer(self, x, n_hidden, layer_idx):
        """
            Builds one_fully_connected_layer

            Args:
                x: an input tensor with the dimensions (N_examples, N_features)
                n_hidden: number of hidden units
                layer_idx: the layer index of the fully connected layer that is currently being built

            Returns:
                an output tensor with the dimensions (N_examples, n_hidden)
        """

        with tf.variable_scope(self.name):
            n_input = int(x.shape[1])
            w = tf.get_variable("w_"+str(layer_idx), [n_input, n_hidden],
                    initializer=self.initializer())

            b = tf.get_variable("b_"+str(layer_idx), [n_hidden,],
                    initializer=tf.constant_initializer(0.))
            return self.activation_fn(tf.matmul(x, w) + b)

    def build_layers(self):
        """
            Builds a stack of fully connected layers

            Args:

            Returns:
                an output tensor with the dimensions (N_examples, hidden_layers[-1])
        """

        output = self.x
        for layer_idx, n_hidden in enumerate(self.hidden_layers[:-1]):
            output = self.one_fully_connected_layer(output, n_hidden, layer_idx)

        with tf.variable_scope(self.name):
            n_input = int(output.shape[1])

            w = tf.get_variable("w_output", [n_input, self.hidden_layers[-1]],
                    initializer=self.initializer())

            b = tf.get_variable("b_output"+str(layer_idx), [self.hidden_layers[-1],],
                    initializer=tf.constant_initializer(0.))

            return tf.nn.sigmoid(tf.matmul(output, w) + b)


class LocalSensorNetwork(Network):
    """
        The LocalSensorNetwork class 
    """

    def __init__(self, name, x, hidden_layers, activation_fn=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer):
        """
            Initilize network

            Args:
                name: name of the network
                x: an input tensor with the dimensions (N_examples, N_features)
                hidden_layers: a 1-D tensor indicates the number of hidden neurons in each layer, the last element is the number of output
                activation_fn: activation function 
                initializer: initialization methods

            Returns:
        """
        super(LocalSensorNetwork, self).__init__(name, hidden_layers)

        self.x = x

class CloudNetwork(Network):
    """
        The CloudNetwork class

        Examples
        --------
        sensor1 = LocalSensorNetwork("sensor1", x, [128,8])
        sensor2 = LocalSensorNetwork("sensor2", x, [256,16])

        cloud = CloudNetwork("cloud", [256,10])
        model = cloud.connect([sensor1, sensor2])
        --------

    """

    def __init__(self, name, hidden_layers, activation_fn=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer):
        super(CloudNetwork, self).__init__(name, hidden_layers)

    def connect(self, sensors = [], method = "inner_product"):
        """
            Connect the output of LocalSensorNetworks to CloudNetwork

            Args:
                sensors: a list of LocalSensorNetworks instance

                method : {'inner_product'}
                    Specifies how to connect LocalSensorNetworks with the CloudNetwork

            Returns:
                an output tensor with the dimensions (N_examples, hidden_layers[-1])
        """

        outputs = []
        for sensor_idx, sensor in enumerate(sensors):
        
            with tf.variable_scope("connect_sensor_" + str(sensor_idx)):
                sensor_output = sensor.build_layers()
                n_input = int(sensor_output.shape[1])

                if method == "inner_product":
                    w = tf.get_variable("w_"+str(sensor_idx), [n_input, 1],
                        initializer=self.initializer())
                    output = tf.matmul(sensor_output, w)
                    outputs.append(output)

                elif method == "concat":
                    outputs.append(sensor_output)

        self.x = tf.concat(outputs, axis=1)

        return self.build_layers()



