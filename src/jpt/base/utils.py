import math
from functools import reduce

import numpy as np
from dnutils import ifnone

from .intervals import ContinuousSet


def mapstr(seq, format=None):
    '''Convert the sequence ``seq`` into a list of strings by applying ``str`` to each of its elements.'''
    return [format(e) for e in seq] if callable(format) else [ifnone(format, '%s') % (e,) for e in seq]


def prod(iterable):
    return reduce(lambda x, y: x * y, iterable)


def tojson(obj):
    """Recursively generate a JSON representation of the object ``obj``."""
    if hasattr(obj, 'tojson'):
        return obj.tojson()
    if type(obj) in (list, tuple):
        return [tojson(e) for e in obj]
    elif isinstance(obj, dict):
        return {str(k): tojson(v) for k, v in obj.items()}
    return obj


def format_path(path):
    '''
    Returns a readible string representation of a conjunction of variable assignments,
    given by the dictionary ``path``.
    '''
    return ' ^ '.join([var.str(val, fmt='logic') for var, val in path.items()])


# ----------------------------------------------------------------------------------------------------------------------
# Entropy calculation

def entropy(p):
    '''Compute the entropy of the multinomial probability distribution ``p``.
    :param p:   the probabilities
    :type p:    [float] or {str:float}
    :return:
    '''
    if isinstance(p, dict):
        p = list(p.values())
    return abs(-sum([0 if p_i == 0 else math.log(p_i, 2) * p_i for p_i in p]))


def max_entropy(n):
    '''Compute the maximal entropy that a multinomial random variable with ``n`` states can have,
    i.e. the entropy value assuming a uniform distribution over the values.
    :param p:
    :return:
    '''
    return entropy([1 / n for _ in range(n)])


def rel_entropy(p):
    '''Compute the entropy of the multinomial probability distribution ``p`` normalized
    by the maximal entropy that a multinomial distribution of the dimensionality of ``p``
    can have.
    :type p: distribution'''
    if len(p) == 1:
        return 0
    return entropy(p) / max_entropy(len(p))


# ----------------------------------------------------------------------------------------------------------------------
# Gini index


def gini(p):
    '''Compute the Gini impurity for the distribution ``p``.'''
    if isinstance(p, dict):
        p = list(p.values())
    return np.mean([p_i * (1 - p_i) for p_i in p])


# ----------------------------------------------------------------------------------------------------------------------


class ClassPropertyDescriptor(object):

    def __init__(self, fget, fset=None):
        self.fget = fget
        self.fset = fset

    def __get__(self, obj, klass=None):
        if klass is None:
            klass = type(obj)
        return self.fget.__get__(obj, klass)()

    def __set__(self, obj, value):
        if not self.fset:
            raise AttributeError("can't set attribute")
        type_ = type(obj)
        return self.fset.__get__(obj, type_)(value)

    def setter(self, func):
        if not isinstance(func, (classmethod, staticmethod)):
            func = classmethod(func)
        self.fset = func
        return self


def classproperty(func):
    '''
    This decorator allows to define class properties in the same way as normal object properties.

    https://stackoverflow.com/questions/5189699/how-to-make-a-class-property
    '''
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)

    return ClassPropertyDescriptor(func)


# ----------------------------------------------------------------------------------------------------------------------


def list2interval(l):
    '''
    Converts a list representation of an interval to an instance of type
    '''
    lower, upper = l
    return ContinuousSet(np.NINF if lower in (np.NINF, -float('inf'), None) else lower,
                         np.PINF if upper in (np.PINF, float('inf'), None) else upper)