# -*- coding: UTF-8 -*-
# File: save_sensor_networks.py

import tensorflow as tf
from datetime import datetime
import os

from .base import Callback
from ..utils import logger
from ..tfutils.common import get_tf_version_number

__all__ = ['SaveSensorNetworks']

def freeze_graph(sess, var_list):
    # convert_variables_to_constants(sess, input_graph_def, output_node_names, variable_names_whitelist=None)
    #with gfile.FastGFile("./tmp/" + "graph.pb", 'rb') as f:
    #    graph_def = tf.GraphDef()
    #    graph_def.ParseFromString(f.read())

    for var in var_list:
        frozen_graph_def = convert_variables_to_constants(sess, sess.graph_def, ["connect_sensor_0/hand_1/output"])
        with tf.gfile.GFile("./tmp/" + var + "frozen.pb", "wb") as f:
            f.write(frozen_graph_def.SerializeToString())

class SaveSensorNetworks(Callback):
    """
    Save the model once triggered.
    """

    def __init__(self, SensorsToSave = [],
                max_to_keep=10,
                keep_checkpoint_every_n_hours=0.5,
                saving_dir=None,
                ):
        """
        Args:
            max_to_keep (int): the same as in ``tf.train.Saver``.
            keep_checkpoint_every_n_hours (float): the same as in ``tf.train.Saver``.
            saving_dir (str): Defaults to ``logger.get_logger_dir()``.
            var_collections (str or list of str): collection of the variables (or list of collections) to save.
        """
        self._sess = tf.get_default_session()
        self._graph = tf.get_default_graph()
        self._prefix = "InferenceTower/"
        self._var_list = []

        if saving_dir is None:
            saving_dir = logger.get_logger_dir()

        if saving_dir is not None:
            if not tf.gfile.IsDirectory(saving_dir):
                tf.gfile.MakeDirs(saving_dir)

        self.saving_dir = saving_dir

    def _setup_graph(self):
        assert self.saving_dir is not None, \
            "ModelSaver() doesn't have a valid checkpoint directory."

        for sensor in SensorsToSave:
            self._var_list.append(self._prefix + sensor + "_output")

        self.path = os.path.join(self.saving_dir, 'model')

    def _before_train(self):
        # graph is finalized, OK to write it now.
        #time = datetime.now().strftime('%m%d-%H%M%S')
        pass 

    def _trigger(self):
        try:   
            freeze_graph(self._sess, self._var_list)
            logger.info("Model saved to %s." % tf.train.get_checkpoint_state(self.saving_dir).model_checkpoint_path)
        except (OSError, IOError, tf.errors.PermissionDeniedError,
                tf.errors.ResourceExhaustedError):   # disk error sometimes.. just ignore it
            logger.exception("Exception in ModelSaver!")
