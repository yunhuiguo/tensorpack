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
    """ A simple wrapper to easily create "linear" graph,
        consisting of layers / symbolic functions with only one input & output.
    """

    def __init__(self, name, sensors_list):
        """
        Args:
            tensor (tf.Tensor): the tensor to wrap
        """
        self._sensors_list = sensors_list
        print(type(self._sensors_list))
        self.name = name
        self._output = self.connect_sensors()

    def connect_sensors(self, method = "inner_product"):
        outputs = []
        for sensor_idx, sensor_output in enumerate(self._sensors_list):
            with tf.variable_scope("connect_sensor_" + str(sensor_idx)):
                n_input = int(sensor_output.shape[1])

                if method == "inner_product":
                    w = tf.get_variable("w_"+str(sensor_idx), [n_input, 1],
                        initializer=tf.contrib.layers.variance_scaling_initializer(2.0))

                    output = tf.matmul(sensor_output, w)
                    outputs.append(output)

        outputs = tf.concat(outputs, axis=1)
        return Sequential(outputs)

    def __getattr__(self, layer_name):

        def layer_funcs(name, *args, **kwargs):
            print type(self._output)
            obj = self._output.__getattr__(layer_name)
            return obj(name, *args, **kwargs)

        return layers_func

    def __call__(self):
        """
        Returns:
            tf.Tensor: the underlying wrapped tensor.
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