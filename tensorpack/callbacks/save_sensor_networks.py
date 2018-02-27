# -*- coding: UTF-8 -*-
# File: save_sensor_networks.py

import tensorflow as tf
from datetime import datetime
import os

from .base import Callback
from ..utils import logger
from ..tfutils.common import get_tf_version_number

__all__ = ['SaveSensorNetworks']

class SaveSensorNetworks(Callback):
    """
    Save the model once triggered.
    """

    def __init__(self, max_to_keep=10,
                 keep_checkpoint_every_n_hours=0.5,
                 checkpoint_dir=None,
                 var_collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.MODEL_VARIABLES]):
        """
        Args:
            max_to_keep (int): the same as in ``tf.train.Saver``.
            keep_checkpoint_every_n_hours (float): the same as in ``tf.train.Saver``.
            checkpoint_dir (str): Defaults to ``logger.get_logger_dir()``.
            var_collections (str or list of str): collection of the variables (or list of collections) to save.
        """

        
        self._max_to_keep = max_to_keep
        self._keep_every_n_hours = keep_checkpoint_every_n_hours

        if not isinstance(var_collections, list):
            var_collections = [var_collections]
        self.var_collections = var_collections
        if checkpoint_dir is None:
            checkpoint_dir = logger.get_logger_dir()
        if checkpoint_dir is not None:
            if not tf.gfile.IsDirectory(checkpoint_dir):
                tf.gfile.MakeDirs(checkpoint_dir)
        self.checkpoint_dir = checkpoint_dir

    def _setup_graph(self):
        assert self.checkpoint_dir is not None, \
            "ModelSaver() doesn't have a valid checkpoint directory."
        vars = []
        for key in self.var_collections:
            vars.extend(tf.get_collection(key))
        vars = list(set(vars))
        self.path = os.path.join(self.checkpoint_dir, 'model')
        if get_tf_version_number() <= 1.1:
            self.saver = tf.train.Saver(
                var_list=vars,
                max_to_keep=self._max_to_keep,
                keep_checkpoint_every_n_hours=self._keep_every_n_hours,
                write_version=tf.train.SaverDef.V2)
        else:
            self.saver = tf.train.Saver(
                var_list=vars,
                max_to_keep=self._max_to_keep,
                keep_checkpoint_every_n_hours=self._keep_every_n_hours,
                write_version=tf.train.SaverDef.V2,
                save_relative_paths=True)
        # Scaffold will call saver.build from this collection
        tf.add_to_collection(tf.GraphKeys.SAVERS, self.saver)

    def _before_train(self):
        # graph is finalized, OK to write it now.
        time = datetime.now().strftime('%m%d-%H%M%S')
        self.saver.export_meta_graph(
            os.path.join(self.checkpoint_dir,
                         'graph-{}.meta'.format(time)),
            collection_list=self.graph.get_all_collection_keys())

    def _trigger(self):
        try:
            self.saver.save(
                tf.get_default_session(),
                self.path,
                global_step=tf.train.get_global_step(),
                write_meta_graph=False)
            logger.info("Model saved to %s." % tf.train.get_checkpoint_state(self.checkpoint_dir).model_checkpoint_path)
        except (OSError, IOError, tf.errors.PermissionDeniedError,
                tf.errors.ResourceExhaustedError):   # disk error sometimes.. just ignore it
            logger.exception("Exception in ModelSaver!")
