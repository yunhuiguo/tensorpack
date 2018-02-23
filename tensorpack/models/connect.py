#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: fc.py

import tensorflow as tf

from .common import layer_register, VariableHolder
from .tflayer import convert_to_tflayer_args, rename_get_variable
from ..tfutils import symbolic_functions as symbf
from . import Sequential, fc


__all__ = ['Connect']


class Connect(object):
    """ A simple wrapper to easily create "linear" graph,
        consisting of layers / symbolic functions with only one input & output.
    """

    def __init__(self, name, sensors_list = None):
        """
        Args:
            tensor (tf.Tensor): the tensor to wrap
        """
        self._sensor_list = sensors_list
        self.name = name
        self._output = self.connect_sensors()

    def connect_sensors(self, method = "inner_product"):
        outputs = []
        print "\n\n"
        print "success"
        print "\n\n"
     
        for sensor_idx, sensor_output in enumerate(self.sensors_list):
        	output = FullyConnected(sensor_output, sensor_output.shape[1], activation=tf.identity)
        outputs = tf.concat(outputs, axis=1)
 
        return outputs


    def __getattr__(self, layer_name):
        layer = get_registered_layer(layer_name)

        if layer is not None:
            # this is a registered tensorpack layer
            # parse arguments by tensorpack model convention
            if layer.use_scope:
                def layer_func(name, *args, **kwargs):
                    if self._t != None:
                        ret = layer(name, self._output, *args, **kwargs)
                        return Sequential(ret)
            else:
                def layer_func(*args, **kwargs):
                    if len(args) and isinstance(args[0], six.string_types):
                        name, args = args[0], args[1:]
                        ret = layer(name, self._output, *args, **kwargs)
                    else:
                        ret = layer(self._output, *args, **kwargs)
                    return Sequential(ret)
            return layer_func
        else:
            assert layer_name == 'tf', \
                "Calling Sequential.{}:" \
                " neither a layer nor 'tf'! " \
                "Did you forget to extract tensor from Sequential?".format(layer_name)
            import tensorflow as layer  # noqa
            assert isinstance(layer, ModuleType), layer
            return Sequential._TFModuleFunc(layer, self._output)  

    def __call__(self):
        """
        Returns:
            tf.Tensor: the underlying wrapped tensor.
        """
        return self._output

    def sensors_list(self):
        """
        Equivalent to ``self.__call__()``.

        Returns:
            tf.Tensor: the underlying wrapped tensor.
        """
        return self._sensor_list

    def print_sensor_list(self):
        """
        Print the underlying tensor and return self. Can be useful to get the
        name of tensors inside :class:`Sequential`.

        :return: self
        """
        print(self._sensor_list)
        return self













@layer_register(log_shape=True)
def Connect(sensors = [], method = "inner_product"):
    def model(x = [], is_training = True):
        outputs = []

        for sensor_idx, sensor in enumerate(sensors):
            with tf.variable_scope("connect_sensor_" + str(sensor_idx)):
                sensor_output = sensor(x[sensor_idx])
                n_input = int(sensor_output.shape[1])
                if method == "inner_product":
                    w = tf.get_variable("weight_"+str(sensor_idx), [n_input, 1],
                        initializer=tf.contrib.layers.xavier_initializer())

                    output = tf.matmul(sensor_output, w)
                    outputs.append(output)
                elif method == "concat":
                    outputs.append(sensor_output)
        outputs = tf.concat(outputs, axis=1)
        return outputs
    return model