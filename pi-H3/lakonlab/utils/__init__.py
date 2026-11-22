import logging
from functools import reduce


def get_root_logger():
    return logging.getLogger('lakonlab')


def rgetattr(obj, attr):
    return reduce(getattr, attr.split('.'), obj)


def reduce_mean(tensor):
    return tensor


def is_warmup():
    return False


def onload(method):
    return method
