#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: fc.py

import tensorflow as tf

from .common import layer_register, VariableHolder
from .tflayer import convert_to_tflayer_args, rename_get_variable
from ..tfutils import symbolic_functions as symbf

__all__ = ['Connect']

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