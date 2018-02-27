#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: Connect.py

import tensorflow as tf

import six
from types import ModuleType
from .Sequential import Sequential
from .fc import FullyConnected

__all__ = ['Connect']

class Connect(object):
    """ Connect a list of sensors
    """
    def __init__(self, name, sensors_list, connected_method = "inner_product"):
        """
        Args:
            name: name of the network
            tensors_list: a list of sensors to connect
        """
        self._sensors_list = sensors_list
        self._name = name
        self._connected_method = connected_method
        self._output = self.connect_sensors()

    def connect_sensors(self):
        outputs = []
        for sensor_idx, sensor_output in enumerate(self._sensors_list):
            with tf.variable_scope(self._name + "_connect_sensor_" + str(sensor_idx)):
                n_input = int(sensor_output.shape[1])
                if self._connected_method == "inner_product":
                    w = tf.get_variable("w_"+str(sensor_idx), [n_input, 1],
                        initializer=tf.contrib.layers.variance_scaling_initializer(2.0))
                    output = tf.matmul(sensor_output, w)
                    outputs.append(output)
                elif self._connected_method == "concat":
                    outputs = self._sensors_list

        outputs = tf.concat(outputs, axis=1)
        return Sequential(self._name, outputs)

    def __getattr__(self, layer_name):

        def layer_func(name, *args, **kwargs):
            func = self._output.__getattr__(layer_name)
            return func(name, *args, **kwargs)
        return layer_func

    def __call__(self):
        """
        Returns:
            tf.Tensor: .
        """
        return self._output

    def print_sensor_list(self):
        """
        Print the underlying tensor and return self. Can be useful to get the
        name of tensors inside :class:`Sequential`.

        :return: self
        """
        print(self._sensor_list)
        return self