'''© Copyright 2021, Mareike Picklum, Daniel Nyga.
'''

from jpt.distributions.univariate.integer import Integer
from jpt.distributions.univariate.multinomial import Multinomial
from jpt.distributions.univariate.numeric import Numeric, ScaledNumeric

try:
    from ..base.intervals import __module__
    from .quantile.quantiles import __module__
    from ..base.functions import __module__
except ModuleNotFoundError:
    import pyximport

    pyximport.install()
finally:
    from ..base.intervals import R, ContinuousSet, RealSet, NumberSet
    from ..base.functions import LinearFunction, ConstantFunction, PiecewiseFunction
    from .quantile.quantiles import QuantileDistribution

# ----------------------------------------------------------------------------------------------------------------------
# Constant symbols

SYMBOLIC = 'symbolic'
NUMERIC = 'numeric'
CONTINUOUS = 'continuous'
DISCRETE = 'discrete'


# ----------------------------------------------------------------------------------------------------------------------
# Gaussian distribution. This is somewhat deprecated as we use model-free
# quantile distributions, but this code is used in testing to sample
# from Gaussian distributions.
# TODO: In order to keep the code consistent, this class should inherit from 'Distribution'


# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------

# noinspection DuplicatedCode


# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------

# noinspection DuplicatedCode


# ----------------------------------------------------------------------------------------------------------------------

# noinspection PyPep8Naming


# noinspection PyPep8Naming


# noinspection PyPep8Naming


# ----------------------------------------------------------------------------------------------------------------------

_DISTRIBUTION_TYPES = {
    'numeric': Numeric,
    'scaled-numeric': ScaledNumeric,
    'symbolic': Multinomial,
    'integer': Integer
}

_DISTRIBUTIONS = {
    'Numeric': Numeric,
    'ScaledNumeric': ScaledNumeric,
    'Multinomial': Multinomial,
    'Integer': Integer
}
