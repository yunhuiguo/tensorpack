#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: Sequential.py

import tensorflow as tf
import six
from types import ModuleType
from .registry import get_registered_layer

__all__ = ['Sequential']

class Sequential(object):
    """ A simple wrapper to easily create "linear" graph,
        consisting of layers / symbolic functions with only one input & output.
    """

    class _TFModuleFunc(object):
        def __init__(self, name, mod, tensor):
            self._mod = mod
            self._t = tensor
            self._name = name

        def __getattr__(self, name):

            ret = getattr(self._mod, name)
            if isinstance(ret, ModuleType):
                return Sequential._TFModuleFunc(ret, self._t)
            else:
                # assume to be a tf function
                def f(*args, **kwargs):
                    o = ret(self._t, *args, **kwargs)
                    return Sequential(self._name, o)
                return f

    def __init__(self, name, tensor):
        """
        Args:
            tensor (tf.Tensor): the tensor to wrap
        """
        self._t = tensor
        self._name = name
 
    def __getattr__(self, layer_name):
        
        layer = get_registered_layer(layer_name)

        if layer is not None:
            # this is a registered tensorpack layer
            # parse arguments by tensorpack model convention
            if layer.use_scope:
                def layer_func(name, *args, **kwargs):
                    if self._t != None:
                        ret = layer(self._name + "_" + name, self._t, *args, **kwargs)
                        return Sequential(self._name, ret)
            else:
                def layer_func(*args, **kwargs):
                    if len(args) and isinstance(args[0], six.string_types):
                        name, args = args[0], args[1:]
                        ret = layer(self._name + "_" + name, self._t, *args, **kwargs)
                    else:
                        ret = layer(self._t, *args, **kwargs)
                    return Sequential(self._name, ret)
            return layer_func
        else:
            assert layer_name == 'tf', \
                "Calling Sequential.{}:" \
                " neither a layer nor 'tf'! " \
                "Did you forget to extract tensor from Sequential?".format(layer_name)
            import tensorflow as layer  # noqa
            assert isinstance(layer, ModuleType), layer
            return Sequential._TFModuleFunc(self._name, layer, self._t)
    

    def apply(self, func, *args, **kwargs):
        """
        Apply a function on the wrapped tensor.

        Returns:
            Sequential: ``Sequential(func(self.tensor(), *args, **kwargs))``.
        """
        ret = func(self._t, *args, **kwargs)
        return Sequential(ret)

    def apply2(self, func, *args, **kwargs):
        """
        Apply a function on the wrapped tensor. The tensor
        will be the second argument of func.

        Returns:
            Sequential: ``Sequential(func(args[0], self.tensor(), *args[1:], **kwargs))``.
        """
        ret = func(args[0], self._t, *(args[1:]), **kwargs)
        return Sequential(ret)

    def __call__(self):
        """
        Returns:
            tf.Tensor: the underlying wrapped tensor.
        """
        return self._t

    def tensor(self):
        """
        Equivalent to ``self.__call__()``.

        Returns:
            tf.Tensor: the underlying wrapped tensor.
        """
        return self._t

    def print_tensor(self):
        """
        Print the underlying tensor and return self. Can be useful to get the
        name of tensors inside :class:`Sequential`.

        :return: self
        """
        print(self._t)
        return self
