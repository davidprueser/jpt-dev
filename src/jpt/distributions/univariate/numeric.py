import copy
import numbers
import os
from _operator import itemgetter
from collections import deque
from typing import Union, Optional, List, Tuple, Iterable, Type

import numpy as np
from dnutils import ifnone, pairwise, ifnot
from jpt.distributions.quantile.quantiles import QuantileDistribution
from matplotlib import pyplot as plt

from jpt.base.intervals import NumberSet, ContinuousSet, RealSet

from jpt.base.functions import ConstantFunction, LinearFunction, PiecewiseFunction
from jpt.base.utils import normalized, save_plot, none2nan
from jpt.distributions.univariate.distribution import Distribution
from jpt.distributions.utils import Identity, DataScaler, DataScalerProxy


class Numeric(Distribution):
    '''
    Wrapper class for numeric domains and distributions.
    '''

    PRECISION = 'precision'

    values = Identity()
    labels = Identity()

    SETTINGS = {
        PRECISION: .01
    }

    def __init__(self, **settings):
        super().__init__(**settings)
        self._quantile: QuantileDistribution = None
        self.to_json = self.inst_to_json

    def __str__(self):
        return self.cdf.pfmt()

    def __getitem__(self, value):
        return self.p(value)

    def __eq__(self, o: 'Numeric'):
        if not issubclass(type(o), Numeric):
            raise TypeError('Cannot compare object of type %s with other object of type %s' % (type(self),
                                                                                               type(o)))
        return type(o).equiv(type(self)) and self._quantile == o._quantile

    # noinspection DuplicatedCode
    @classmethod
    def value2label(cls, value: Union[numbers.Real, NumberSet]) -> Union[numbers.Real, NumberSet]:
        if isinstance(value, ContinuousSet):
            return ContinuousSet(cls.labels[value.lower], cls.labels[value.upper], value.left, value.right)
        elif isinstance(value, RealSet):
            return RealSet([cls.value2label(i) for i in value.intervals])
        elif isinstance(value, numbers.Real):
            return cls.labels[value]
        else:
            raise TypeError('Expected float or NumberSet type, got %s.' % type(value).__name__)

    # noinspection DuplicatedCode
    @classmethod
    def label2value(cls, label: Union[numbers.Real, NumberSet]) -> Union[numbers.Real, NumberSet]:
        if isinstance(label, ContinuousSet):
            return ContinuousSet(cls.values[label.lower], cls.values[label.upper], label.left, label.right)
        elif isinstance(label, RealSet):
            return RealSet([cls.label2value(i) for i in label.intervals])
        elif isinstance(label, numbers.Real):
            return cls.values[label]
        else:
            raise TypeError('Expected float or NumberSet type, got %s.' % type(label).__name__)

    @classmethod
    def equiv(cls, other):
        return (issubclass(other, Numeric) and
                cls.__name__ == other.__name__ and
                cls.values == other.values and
                cls.labels == other.labels)

    @property
    def cdf(self):
        return self._quantile.cdf

    @property
    def pdf(self):
        return self._quantile.pdf

    @property
    def ppf(self):
        return self._quantile.ppf

    def _sample(self, n):
        return self._quantile.sample(n)

    def _sample_one(self):
        return self._quantile.sample(1)[0]

    def number_of_parameters(self) -> int:
        """
        :return: The number of relevant parameters in this decision node.
                 1 if this is a dirac impulse, number of intervals times two else
        """
        if self.is_dirac_impulse():
            return 1
        else:
            return len(self.cdf.intervals)

    def _expectation(self) -> numbers.Real:
        e = 0
        singular = True  # In case the CDF is jump fct the expectation is where the jump happens
        for i, f in zip(self.cdf.intervals, self.cdf.functions):
            if i.lower == np.NINF or i.upper == np.PINF:
                continue
            e += (self.cdf.eval(i.upper) - self.cdf.eval(i.lower)) * (i.upper + i.lower) / 2
            singular = False
        return e if not singular else i.lower

    def expectation(self) -> numbers.Real:
        return self.moment(1)

    def variance(self) -> numbers.Real:
        return self.moment(2)

    def quantile(self, gamma: numbers.Real) -> numbers.Real:
        return self.ppf.eval(gamma)

    def create_dirac_impulse(self, value):
        """Create a dirac impulse at the given value aus quantile distribution."""
        self._quantile = QuantileDistribution()
        self._quantile.fit(
            np.asarray([[value]], dtype=np.float64),
            rows=np.asarray([0], dtype=np.int64),
            col=0
        )
        return self

    def is_dirac_impulse(self) -> bool:
        """Checks if this distribution is a dirac impulse."""
        return len(self._quantile.cdf.intervals) == 2

    def mpe(self) -> (float, RealSet):
        """
        Calculate the most probable configuration of this quantile distribution.
        :return: The likelihood of the mpe as float and the mpe itself as RealSet
        """
        _max = max(f.value for f in self.pdf.functions)
        return _max, self.value2label(
            RealSet([
                interval
                for interval, function in zip(self.pdf.intervals, self.pdf.functions)
                if function.value == _max
            ])
        )

    def k_mpe(self, k: Optional[int] = None) -> List[Tuple[float, RealSet]]:
        """
        Calculate the ``k`` most probable explanation states.
        :param k: The number of solutions to generate, defaults to the maximum possible number.
        :return: A list containing a tuple containing the likelihood and state in descending order.
        """
        if k is None:
            k = len(self.pdf.functions[1:-1])

        sorted_likelihood = sorted(set([f.value for f in self.pdf.functions[1:-1]]), reverse=True)[:k]
        result = []

        for likelihood in sorted_likelihood:
            result.append((likelihood, RealSet([
                interval
                for interval, function in zip(self.pdf.intervals, self.pdf.functions)
                if function.value == likelihood
            ])))

        return result

    def _fit(
            self,
            data: np.ndarray,
            rows: np.ndarray = None,
            col: numbers.Integral = None
    ) -> 'Numeric':
        self._quantile = QuantileDistribution(epsilon=self.precision)
        self._quantile.fit(
            data,
            rows=ifnone(
                rows,
                np.array(list(range(data.shape[0])), dtype=np.int64)
            ),
            col=ifnone(col, 0)
        )
        return self

    fit = _fit

    def set(self, params: QuantileDistribution) -> 'Numeric':
        self._quantile = params
        return self

    def _p(self, value: Union[numbers.Number, NumberSet]) -> numbers.Real:
        if isinstance(value, numbers.Number):
            value = ContinuousSet(value, value)
        elif isinstance(value, RealSet):
            return sum(self._p(i) for i in value.intervals)
        probspace = self.pdf.gt(0)
        if probspace.isdisjoint(value):
            return 0
        probmass = ((self.cdf.eval(value.upper) if value.upper != np.PINF else 1.) -
                    (self.cdf.eval(value.lower) if value.lower != np.NINF else 0.))
        if not probmass:
            return probspace in value
        return probmass

    def p(self, labels: Union[numbers.Number, NumberSet]) -> numbers.Real:
        if not isinstance(labels, (NumberSet, numbers.Number)):
            raise TypeError('Argument must be numbers.Number or '
                            'jpt.base.intervals.NumberSet (got %s).' % type(labels))
        if isinstance(labels, ContinuousSet):
            return self._p(self.label2value(labels))
        elif isinstance(labels, RealSet):
            self._p(RealSet([ContinuousSet(self.values[i.lower],
                                           self.values[i.upper],
                                           i.left,
                                           i.right) for i in labels.intervals]))
        else:
            return self._p(self.values[labels])

    def kl_divergence(self, other: 'Numeric') -> numbers.Real:
        if type(other) is not type(self):
            raise TypeError('Can only compute KL divergence between '
                            'distributions of the same type, got %s' % type(other))
        self_ = [(i.lower, f.value, None) for i, f in self.pdf.iter()]
        other_ = [(i.lower, None, f.value) for i, f in other.pdf.iter()]
        all_ = deque(sorted(self_ + other_, key=itemgetter(0)))
        queue = deque()
        while all_:
            v, p, q = all_.popleft()
            if queue and v == queue[-1][0]:
                if p is not None:
                    queue[-1][1] = p
                if q is not None:
                    queue[-1][2] = q
            else:
                queue.append([v, p, q])
        result = 0
        p, q = 0, 0
        for (x0, p_, q_), (x1, _, _) in pairwise(queue):
            p = ifnone(p_, p)
            q = ifnone(q_, q)
            i = ContinuousSet(x0, x1)
            result += self._p(i) * abs(self._p(i) - other._p(i))
        return result

    def copy(self):
        dist = type(self)(**self.settings).set(params=self._quantile.copy())
        dist.values = copy.copy(self.values)
        dist.labels = copy.copy(self.labels)
        return dist

    @staticmethod
    def merge(distributions: List['Numeric'], weights: Iterable[numbers.Real]) -> 'Numeric':
        if not all(distributions[0].__class__ == d.__class__ for d in distributions):
            raise TypeError('Only distributions of the same type can be merged.')
        return type(distributions[0])().set(QuantileDistribution.merge(distributions, weights))

    def update(self, dist: 'Numeric', weight: numbers.Real) -> 'Numeric':
        if not 0 <= weight <= 1:
            raise ValueError('Weight must be in [0, 1]')
        if type(dist) is not type(self):
            raise TypeError('Can only update with distribution of the same type, got %s' % type(dist))
        tmp = Numeric.merge([self, dist], normalized([1, weight]))
        self.values = tmp.values
        self.labels = tmp.labels
        self._quantile = tmp._quantile
        return self

    def _crop(self, interval):
        dist = self.copy()
        dist._quantile = self._quantile.crop(interval)
        return dist

    def crop(self, restriction: RealSet or ContinuousSet or numbers.Number):
        """Apply a restriction to this distribution. The restricted distrubtion will only assign mass
        to the given range and will preserve the relativity of the pdf.

        :param restriction: The range to limit this distribution (or singular value)
        :type restriction: float or int or ContinuousSet
        """

        # for real sets the result is a merge of the single ContinuousSet crops
        if isinstance(restriction, RealSet):

            distributions = []

            for idx, continuous_set in enumerate(restriction.intervals):
                distributions.append(self.crop(continuous_set))

            weights = np.full((len(distributions)), 1 / len(distributions))

            return self.merge(distributions, weights)

        elif isinstance(restriction, ContinuousSet):
            if restriction.size() == 1:
                return self.crop(restriction.lower)
            else:
                return self._crop(restriction)

        elif isinstance(restriction, numbers.Number):
            return self.create_dirac_impulse(restriction)

        else:
            raise ValueError("Unknown Datatype for cropping a numeric distribution, type is %s" % type(restriction))

    @classmethod
    def type_to_json(cls):
        return {
            'type': 'numeric',
            'class': cls.__name__
        }

    def inst_to_json(self):
        return {
            'class': type(self).__name__,
            'settings': self.settings,
            'quantile': self._quantile.to_json() if self._quantile is not None else None
        }

    to_json = type_to_json

    @staticmethod
    def from_json(data):
        return Numeric(**data['settings']).set(QuantileDistribution.from_json(data['quantile']))

    @classmethod
    def type_from_json(cls, data):
        return cls

    def insert_convex_fragments(self, left: ContinuousSet or None, right: ContinuousSet or None,
                                number_of_samples: int):
        """Insert fragments of distributions on the right and left part of this distribution. This should only be used
        to create a convex hull around the JPTs domain which density is never 0.

        :param right: The right (lower) interval to add on if needed and None else
        :param left: The left (upper) interval to add on if needed and None else
        :param number_of_samples: The number of samples to use as basis for the weight
        """

        # create intervals used in the new distribution
        points = [-float("inf")]

        if left:
            points.extend([left.lower, left.upper])

        if right:
            points.extend([right.lower, right.upper])

        points.append(float("inf"))

        intervals = [ContinuousSet(a, b) for a, b in zip(points[:-1], points[1:])]

        valid_arguments = [e for e in [left, right] if e is not None]
        number_of_intervals = len(valid_arguments)
        functions = [ConstantFunction(0.)]

        for idx, interval in enumerate(intervals[1:]):
            prev_value = functions[idx].eval(interval.lower)

            if interval in valid_arguments:
                functions.append(
                    LinearFunction.from_points(
                        (interval.lower, prev_value),
                        (interval.upper, prev_value + 1 / number_of_intervals)
                    )
                )
            else:
                functions.append(ConstantFunction(prev_value))

        cdf = PiecewiseFunction()
        cdf.intervals = intervals
        cdf.functions = functions
        quantile = QuantileDistribution.from_cdf(cdf)
        self._quantile = QuantileDistribution.merge(
            [self._quantile, quantile],
            [
                1 - (1 / (2 * number_of_samples)),
                1 / (2 * number_of_samples)
            ]
        )

    def moment(self, order=1, center=0):
        r"""Calculate the central moment of the r-th order almost everywhere.

        .. math:: \int (x-c)^{r} p(x)

        :param order: The order of the moment to calculate
        :param center: The constant (c) to subtract in the basis of the exponent
        """
        # We have to catch the special case in which the
        # PDF is an impulse function
        if self.pdf.is_impulse():
            if order == 1:
                return self.pdf.gt(0).min
            elif order >= 2:
                return 0
        result = 0
        for interval, function in zip(self.pdf.intervals[1:-1], self.pdf.functions[1:-1]):
            interval_ = self.value2label(interval)

            function_value = function.value * interval.range() / interval_.range()
            result += (
                          (pow(interval_.upper - center, order + 1) - pow(interval_.lower - center, order + 1))
                      ) * function_value / (order + 1)
        return result

    def plot(self, title=None, fname=None, xlabel='value', directory='/tmp', pdf=False, view=False, **kwargs):
        '''
        Generates a plot of the piecewise linear function representing
        the variable's cumulative distribution function

        :param title:       the name of the variable this distribution represents
        :type title:        str
        :param fname:       the name of the file to be stored
        :type fname:        str
        :param xlabel:      the label of the x-axis
        :type xlabel:       str
        :param directory:   the directory to store the generated plot files
        :type directory:    str
        :param pdf:         whether to store files as PDF. If false, a png is generated by default
        :type pdf:          bool
        :param view:        whether to display generated plots, default False (only stores files)
        :type view:         bool
        :return:            None
        '''
        if not os.path.exists(directory):
            os.makedirs(directory)

        if not view:
            plt.ioff()

        fig, ax = plt.subplots()
        ax.set_title(f'{title or f"CDF of {self._cl}"}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('%')
        ax.set_ylim(-.1, 1.1)

        if len(self.cdf.intervals) == 2:
            std = abs(self.cdf.intervals[0].upper) * .1
        else:
            std = ifnot(np.std([i.upper - i.lower for i in self.cdf.intervals[1:-1]]),
                        self.cdf.intervals[1].upper - self.cdf.intervals[1].lower) * 2

        # add horizontal line before first interval of distribution
        X = np.array([self.cdf.intervals[0].upper - std])

        for i, f in zip(self.cdf.intervals[:-1], self.cdf.functions[:-1]):
            if isinstance(f, ConstantFunction):
                X = np.append(X, [np.nextafter(i.upper, i.upper - 1), i.upper])
            else:
                X = np.append(X, i.upper)

        # add horizontal line after last interval of distribution
        X = np.append(X, self.cdf.intervals[-1].lower + std)
        X_ = np.array([self.labels[x] for x in X])
        Y = np.array(self.cdf.multi_eval(X))
        ax.plot(X_,
                Y,
                color='cornflowerblue',
                linestyle='dashed',
                label='Piecewise linear CDF from bounds',
                linewidth=2,
                markersize=12)

        bounds = np.array([i.upper for i in self.cdf.intervals[:-1]])
        bounds_ = np.array([self.labels[b] for b in bounds])
        ax.scatter(bounds_,
                   np.asarray(self.cdf.multi_eval(bounds)),
                   color='orange',
                   marker='o',
                   label='Piecewise Function limits')

        ax.legend(loc='upper left', prop={'size': 8})  # do we need a legend with only one plotted line?
        fig.tight_layout()

        save_plot(fig, directory, fname or self.__class__.__name__, fmt='pdf' if pdf else 'svg')

        if view:
            plt.show()


class ScaledNumeric(Numeric):
    '''
    Scaled numeric distribution represented by mean and variance.
    '''

    scaler = None

    def __init__(self, **settings):
        super().__init__(**settings)

    @classmethod
    def type_to_json(cls):
        return {
            'type': 'scaled-numeric',
            'class': cls.__name__,
            'scaler': cls.scaler.to_json()
        }

    to_json = type_to_json

    @staticmethod
    def type_from_json(data):
        clazz = NumericType(data['class'], None)
        clazz.scaler = DataScaler.from_json(data['scaler'])
        clazz.values = DataScalerProxy(clazz.scaler)
        clazz.labels = DataScalerProxy(clazz.scaler, True)
        return clazz

    @classmethod
    def from_json(cls, data):
        return cls(**data['settings']).set(QuantileDistribution.from_json(data['quantile']))


def NumericType(name: str, values: Iterable[float]) -> Type[ScaledNumeric]:
    t = type(name, (ScaledNumeric,), {})
    if values is not None:
        values = np.array(list(none2nan(values)))
        if (~np.isfinite(values)).any():
            raise ValueError('Values contain nan or inf.')
        t.scaler = DataScaler(values)
        t.values = DataScalerProxy(t.scaler, inverse=False)
        t.labels = DataScalerProxy(t.scaler, inverse=True)
    return t
