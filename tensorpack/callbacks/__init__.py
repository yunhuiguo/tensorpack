#  -*- coding: UTF-8 -*-
#  File: __init__.py


if False:
    from .base import *
    from .concurrency import *
    from .graph import *
    from .group import *
    from .hooks import *
    from .inference import *
    from .inference_runner import *
    from .monitor import *
    from .param import *
    from .prof import *
    from .saver import *
    from .stats import *
    from .steps import *
    from .summary import *
    from .trigger import *


from pkgutil import iter_modules
import os


__all__ = []


def _global_import(name):
    p = __import__(name, globals(), locals(), level=1)
    lst = p.__all__ if '__all__' in dir(p) else dir(p)
    if lst:
        del globals()[name]
        for k in lst:
            if not k.startswith('__'):
                globals()[k] = p.__dict__[k]
                __all__.append(k)


_CURR_DIR = os.path.dirname(__file__)
for _, module_name, _ in iter_modules(
       [_CURR_DIR]):
    srcpath = os.path.join(_CURR_DIR, module_name + '.py')
    if not os.path.isfile(srcpath):
        continue
    if not module_name.startswith('_'):
        _global_import(module_name)
