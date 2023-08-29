import numbers
import re
from _operator import itemgetter
from itertools import tee
from types import FunctionType
from typing import Type, List, Set, Union, Any, Iterable, Tuple, Optional

import numpy as np
from dnutils import edict, ifnone, project
from matplotlib import pyplot as plt

from jpt.base.errors import Unsatisfiability
from jpt.base.sampling import wsample, wchoice
from jpt.base.utils import classproperty, setstr, normalized, save_plot
from jpt.distributions.univariate.distribution import Distribution
from jpt.distributions.utils import OrderedDictProxy


class Integer(Distribution):
    lmin = None
    lmax = None
    vmin = None
    vmax = None
    values = None
    labels = None

    OPEN_DOMAIN = 'open_domain'
    AUTO_DOMAIN = 'auto_domain'

    SETTINGS = edict(Distribution.SETTINGS) + {
        OPEN_DOMAIN: False,
        AUTO_DOMAIN: False
    }

    def __init__(self, **settings):
        super().__init__(**settings)
        if not issubclass(type(self), Integer) or type(self) is Integer:
            raise Exception(f'Instantiation of abstract class {type(self)} is not allowed!')
        self._params: np.ndarray = None
        self.to_json: FunctionType = self.inst_to_json

    @classmethod
    def equiv(cls, other: Type):
        if not issubclass(other, Integer):
            return False
        return all((
            cls.__name__ == other.__name__,
            cls.labels == other.labels,
            cls.values == other.values,
            cls.lmin == other.lmin,
            cls.lmax == other.lmax,
            cls.vmin == other.vmin,
            cls.vmax == other.vmax
        ))

    @classmethod
    def type_to_json(cls):
        return {
            'type': 'integer',
            'class': cls.__qualname__,
            'labels': list(cls.labels.values()),
            'vmin': int(cls.vmin),
            'vmax': int(cls.vmax),
            'lmin': int(cls.lmin),
            'lmax': int(cls.lmax)
        }

    to_json = type_to_json

    def inst_to_json(self):
        return {
            'class': type(self).__qualname__,
            'params': list(self._params),
            'settings': self.settings
        }

    @classmethod
    def list2set(cls, bounds: List[int]) -> Set[int]:
        '''
        Convert a 2-element list specifying a lower and an upper bound into a
        integer set containing the admissible values of the corresponding interval
        '''
        if not len(bounds) == 2:
            raise ValueError('Argument list must have length 2, got length %d.' % len(bounds))
        if bounds[0] < cls.lmin or bounds[1] > cls.lmax:
            raise ValueError(f'Argument must be in [%d, %d].' % (cls.lmin, cls.lmax))
        return set(range(bounds[0], bounds[1] + 1))

    @staticmethod
    def type_from_json(data):
        return IntegerType(data['class'], data['lmin'], data['lmax'])

    @classmethod
    def from_json(cls, data):
        return cls(**data['settings']).set(data['params'])

    def copy(self):
        result = type(self)(**self.settings)
        result._params = np.array(self._params)
        return result

    @property
    def probabilities(self):
        return self._params

    @classproperty
    def n_values(cls):
        return cls.lmax - cls.lmin + 1

    # noinspection DuplicatedCode
    @classmethod
    def value2label(cls, value: Union[Any, Set]) -> Union[Any, Set]:
        if type(value) is set:
            # if cls.open_domain:
            #     return {cls.labels[v] for v in value if v in cls.labels}
            # else:
            return {cls.labels[v] for v in value}
        else:
            return cls.labels[value]

    # noinspection DuplicatedCode
    @classmethod
    def label2value(cls, label: Union[Any, Set]) -> Union[Any, Set]:
        if type(label) is set:
            return {cls.values[l_] for l_ in label}
        else:
            return cls.values[label]

    def _sample(self, n) -> Iterable:
        return wsample(list(self.values.values()), weights=self.probabilities, k=n)

    def _sample_one(self) -> int:
        return wchoice(list(self.values.values()), weights=self.probabilities)

    def p(self, labels: Union[int, Iterable[int]]):
        if not isinstance(labels, Iterable):
            labels = {labels}
        i1, i2 = tee(labels, 2)
        if not all(isinstance(v, numbers.Integral) and self.lmin <= v <= self.lmax for v in i1):
            if self.open_domain:
                return 0
            raise ValueError('Arguments must be in %s' % setstr(self.labels.values(), limit=5))
        return self._p(self.values[l] for l in labels)

    def _p(self, values: Union[int, Iterable[int]]):
        if not isinstance(values, Iterable):
            values = {values}
        i1, i2 = tee(values, 2)
        if not all(isinstance(v, numbers.Integral) and self.vmin <= v <= self.vmax for v in i1):
            raise ValueError('Arguments must be in %s' % setstr(self.values.values(), limit=5))
        return sum(self._params[v] for v in i2)

    def expectation(self) -> numbers.Real:
        return sum(p * v for p, v in zip(self.probabilities, self.values))

    def _expectation(self) -> numbers.Real:
        return sum(p * v for p, v in zip(self.probabilities, self.labels))

    def variance(self) -> numbers.Real:
        e = self.expectation()
        return sum((l - e) ** 2 * p for l, p in zip(self.labels.values(), self.probabilities))

    def _variance(self) -> numbers.Real:
        e = self._expectation()
        return sum((v - e) ** 2 * p for v, p in zip(self.values.values(), self.probabilities))

    def mpe(self) -> Tuple[float, Set[int]]:
        p_max = max(self.probabilities)
        return p_max, {l for l, p in zip(self.labels.values(), self.probabilities) if p == p_max}

    def _mpe(self) -> Tuple[float, Set[int]]:
        p_max = max(self.probabilities)
        return p_max, {l for l, p in zip(self.values.values(), self.probabilities) if p == p_max}

    def k_mpe(self, k: Optional[int] = None) -> List[Tuple[float, Set[int]]]:
        """
        Calculate the ``k`` most probable explanation states.
        :param k: The number of solutions to generate, defaults to the maximum possible number.
        :return: A list containing a tuple containing the likelihood and state in descending order.
        """

        if k is None:
            k = len(self.probabilities)

        sorted_likelihood = sorted(set(self.probabilities), reverse=True)[:k]
        result = []

        for likelihood in sorted_likelihood:
            result.append((likelihood, set([label for label, p in zip(self.labels.values(), self.probabilities) if
                                            p == likelihood])))

        return result

    def _k_mpe(self, k: int) -> List[Tuple[float, Set[int]]]:
        """
        Calculate the ``k`` most probable explanation states.
        :param k: The number of solutions to generate
        :return: An list containing a tuple containing the likelihood and state in descending order.
        """
        sorted_likelihood = sorted(set(self.probabilities), reverse=True)[:k]
        result = []

        for likelihood in sorted_likelihood:
            result.append((likelihood, set([value for value, p in zip(self.values.values(), self.probabilities) if
                                            p == likelihood])))

        return result

    def crop(self, restriction: Union[Iterable, int]) -> 'Distribution':
        if isinstance(restriction, numbers.Integral):
            restriction = {restriction}
        return self._crop([self.label2value(l) for l in restriction])

    def _crop(self, restriction: Union[Iterable, int]) -> 'Distribution':
        if isinstance(restriction, numbers.Integral):
            restriction = {restriction}
        result = self.copy()
        try:
            params = normalized([
                p if v in restriction else 0
                for p, v in zip(self.probabilities, self.values.values())
            ])
        except ValueError:
            raise Unsatisfiability('Restriction unsatisfiable: probabilities must sum to 1.')
        else:
            return result.set(params=params)

    @staticmethod
    def merge(distributions: Iterable['Integer'], weights: Iterable[numbers.Real]) -> 'Integer':
        if not all(type(distributions[0]).equiv(type(d)) for d in distributions):
            raise TypeError('Only distributions of the same type can be merged.')
        if abs(1 - sum(weights)) > 1e-10:
            raise ValueError('Weights must sum to 1 (but is %s).' % sum(weights))
        params = np.zeros(distributions[0].n_values)
        for d, w in zip(distributions, weights):
            params += d.probabilities * w
        if abs(sum(params)) < 1e-10:
            raise Unsatisfiability('Sum of weights must not be zero.')
        return type(distributions[0])().set(params)

    def update(self, dist: 'Integer', weight: numbers.Real) -> 'Integer':
        if not 0 <= weight <= 1:
            raise ValueError('Weight must be in [0, 1]')
        if self._params is None:
            self._params = np.zeros(self.n_values)
        self._params *= 1 - weight
        self._params += dist._params * weight
        return self

    def fit(self,
            data: np.ndarray,
            rows: np.ndarray = None,
            col: numbers.Integral = None) -> 'Integer':
        if rows is None:
            rows = range(data.shape[0])
        data_ = np.array([self.label2value(int(data[row][col])) for row in rows], dtype=data.dtype)
        return self._fit(data_.reshape(-1, 1), None, 0)

    def _fit(self,
             data: np.ndarray,
             rows: np.ndarray = None,
             col: numbers.Integral = None) -> 'Integer':
        self._params = np.zeros(shape=self.n_values, dtype=np.float64)
        n_samples = ifnone(rows, len(data), len)
        col = ifnone(col, 0)
        for row in ifnone(rows, range(data.shape[0])):
            self._params[int(data[row, col])] += 1 / n_samples
        return self

    def set(self, params: Iterable[numbers.Real]) -> 'Integer':
        if len(self.values) != len(params):
            raise ValueError('Number of values and probabilities must coincide.')
        self._params = np.array(params, dtype=np.float64)
        return self

    def __eq__(self, other):
        return type(self).equiv(type(other)) and (self.probabilities == other.probabilities).all()

    def __str__(self):
        if self._p is None:
            return f'<{type(self).__qualname__} p=n/a>'
        return f'<{self._cl} p=[{"; ".join([f"{v}: {p:.3f}" for v, p in zip(self.values, self.probabilities)])}]>'

    def __repr__(self):
        return str(self)

    def sorted(self):
        return sorted([(p, l) for p, l in zip(self._params, self.labels.values())],
                      key=itemgetter(0), reverse=True)

    def _items(self):
        '''Return a list of (probability, value) pairs representing this distribution.'''
        return [(p, v) for p, v in zip(self._params, self.values.values())]

    def items(self):
        '''Return a list of (probability, value) pairs representing this distribution.'''
        return [(p, self.label2value(l)) for p, l in self._items()]

    def kl_divergence(self, other: 'Distribution'):
        if type(other) is not type(self):
            raise TypeError('Can only compute KL divergence between '
                            'distributions of the same type, got %s' % type(other))
        result = 0
        for v in range(self.n_values):
            result += self._params[v] * abs(self._params[v] - other._params[v])
        return result

    def number_of_parameters(self) -> int:
        return self._params.shape[0]

    def moment(self, order=1, c=0):
        r"""Calculate the central moment of the r-th order almost everywhere.

        .. math:: \int (x-c)^{r} p(x)

        :param order: The order of the moment to calculate
        :param c: The constant to subtract in the basis of the exponent
        """
        result = 0
        for value, probability in zip(self.labels.values(), self._params):
            result += pow(value - c, order) * probability
        return result

    def plot(self, title=None, fname=None, directory='/tmp', pdf=False, view=False, horizontal=False, max_values=None):
        '''Generates a ``horizontal`` (if set) otherwise `vertical` bar plot representing the variable's distribution.

        :param title:       the name of the variable this distribution represents
        :type title:        str
        :param fname:       the name of the file to be stored
        :type fname:        str
        :param directory:   the directory to store the generated plot files
        :type directory:    str
        :param pdf:         whether to store files as PDF. If false, a png is generated by default
        :type pdf:          bool
        :param view:        whether to display generated plots, default False (only stores files)
        :type view:         bool
        :param horizontal:  whether to plot the bars horizontally, default is False, i.e. vertical bars
        :type horizontal:   bool
        :param max_values:  maximum number of values to plot
        :type max_values:   int
        :return:            None
        '''
        # Only save figures, do not show
        if not view:
            plt.ioff()

        max_values = min(ifnone(max_values, len(self.labels)), len(self.labels))

        labels = list(sorted(list(enumerate(self.labels.values())),
                             key=lambda x: self._params[x[0]],
                             reverse=True))[:max_values]
        labels = project(labels, 1)
        probs = list(sorted(self._params, reverse=True))[:max_values]

        vals = [re.escape(str(x)) for x in labels]

        x = np.arange(max_values)  # the label locations
        # width = .35  # the width of the bars
        err = [.015] * max_values

        fig, ax = plt.subplots()
        ax.set_title(f'{title or f"Distribution of {self._cl}"}')
        if horizontal:
            ax.barh(x, probs, xerr=err, color='cornflowerblue', label='P', align='center')
            ax.set_xlabel('P')
            ax.set_yticks(x)
            ax.set_yticklabels(vals)
            ax.invert_yaxis()
            ax.set_xlim(left=0., right=1.)

            for p in ax.patches:
                h = p.get_width() - .09 if p.get_width() >= .9 else p.get_width() + .03
                plt.text(h, p.get_y() + p.get_height() / 2,
                         f'{p.get_width():.2f}',
                         fontsize=10, color='black', verticalalignment='center')
        else:
            ax.bar(x, probs, yerr=err, color='cornflowerblue', label='P')
            ax.set_ylabel('P')
            ax.set_xticks(x)
            ax.set_xticklabels(vals)
            ax.set_ylim(bottom=0., top=1.)

            # print precise value labels on bars
            for p in ax.patches:
                h = p.get_height() - .09 if p.get_height() >= .9 else p.get_height() + .03
                plt.text(p.get_x() + p.get_width() / 2, h,
                         f'{p.get_height():.2f}',
                         rotation=90, fontsize=10, color='black', horizontalalignment='center')

        fig.tight_layout()

        save_plot(fig, directory, fname or self.__class__.__name__, fmt='pdf' if pdf else 'svg')

        if view:
            plt.show()


def IntegerType(name: str, lmin: int, lmax: int) -> Type[Integer]:
    if lmin > lmax:
        raise ValueError('Min label is greater tham max value: %s > %s' % (lmin, lmax))
    t = type(name, (Integer,), {})
    t.values = OrderedDictProxy([(l, v) for l, v in zip(range(lmin, lmax + 1), range(lmax - lmin + 1))])
    t.labels = OrderedDictProxy([(v, l) for l, v in zip(range(lmin, lmax + 1), range(lmax - lmin + 1))])
    t.lmin = lmin
    t.lmax = lmax
    t.vmin = 0
    t.vmax = lmax - lmin
    return t
