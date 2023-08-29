import numbers
import re
from _operator import itemgetter
from collections import Counter
from itertools import tee
from types import FunctionType
from typing import Union, Any, Set, Iterable, Optional, List, Tuple, Type

import numpy as np
from dnutils import ifnone, project
from matplotlib import pyplot as plt
from numpy import iterable

from jpt.base.constants import sepcomma
from jpt.base.errors import Unsatisfiability
from jpt.base.sampling import wsample, wchoice
from jpt.base.utils import mapstr, classproperty, normalized, save_plot
from jpt.distributions.univariate.distribution import Distribution
from jpt.distributions.utils import OrderedDictProxy


class Multinomial(Distribution):
    '''
    Abstract supertype of all symbolic domains and distributions.
    '''

    values: OrderedDictProxy = None
    labels: OrderedDictProxy = None

    def __init__(self, **settings):
        super().__init__(**settings)
        if not issubclass(type(self), Multinomial) or type(self) is Multinomial:
            raise Exception(f'Instantiation of abstract class {type(self)} is not allowed!')
        self._params: np.ndarray = None
        self.to_json: FunctionType = self.inst_to_json

    # noinspection DuplicatedCode
    @classmethod
    def value2label(cls, value: Union[Any, Set]) -> Union[Any, Set]:
        if type(value) is set:
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

    @classmethod
    def pfmt(cls, max_values=10, labels_or_values='labels') -> str:
        '''
        Returns a pretty-formatted string representation of this class.

        By default, a set notation with value labels is used. By setting
        ``labels_or_values`` to ``"values"``, the internal value representation
        is used. If the domain comprises more than ``max_values`` values,
        the middle part of the list of values is abbreviated by "...".
        '''
        if labels_or_values not in ('labels', 'values'):
            raise ValueError('Illegal Value for "labels_or_values": Expected one out of '
                             '{"labels", "values"}, got "%s"' % labels_or_values)
        return '%s = {%s}' % (cls.__name__, ', '.join(mapstr(cls.values.values()
                                                             if labels_or_values == 'values'
                                                             else cls.labels.values(), limit=max_values)))

    @property
    def probabilities(self):
        return self._params

    @classproperty
    def n_values(cls):
        return len(cls.values)

    def __contains__(self, item):
        return item in self.values

    @classmethod
    def equiv(cls, other):
        if not issubclass(other, Multinomial):
            return False
        return cls.__name__ == other.__name__ and cls.labels == other.labels and cls.values == other.values

    def __getitem__(self, value):
        return self.p([value])

    def __setitem__(self, label, p):
        self._params[self.values[label]] = p

    def __eq__(self, other):
        return type(self).equiv(type(other)) and (self.probabilities == other.probabilities).all()

    def __str__(self):
        if self._p is None:
            return f'{self._cl}<p=n/a>'
        return f'{self._cl}<p=[{";".join([f"{v}={p:.3f}" for v, p in zip(self.labels, self.probabilities)])}]>'

    def __repr__(self):
        if self._p is None:
            return f'{self._cl}<p=n/a>'
        return f'\n{self._cl}<p=[\n{sepcomma.join([f" {v}={p:.3}" for v, p in zip(self.labels, self.probabilities)])}]>;'

    def sorted(self):
        return sorted([
            (p, l) for p, l in zip(self._params, self.labels.values())],
            key=itemgetter(0),
            reverse=True
        )

    def items(self):
        '''Return a list of (probability, label) pairs representing this distribution.'''
        return [(p, l) for p, l in zip(self._params, self.labels.values())]

    def copy(self):
        return type(self)(**self.settings).set(params=self._params)

    def p(self, labels):
        if not isinstance(labels, Iterable):
            values = {labels}
        if not isinstance(labels, (set, list, tuple, np.ndarray)):
            raise TypeError('Argument must be iterable (got %s).' % type(labels))
        return self._p(self.values[label] for label in labels)

    def _p(self, values):
        if not isinstance(values, Iterable):
            values = {values}
        i1, i2 = tee(values, 2)
        if not all(isinstance(v, numbers.Integral) for v in i1):
            raise TypeError('All arguments must be integers.')
        return sum(self._params[v] for v in i2)

    def create_dirac_impulse(self, value):
        result = self.copy()
        result._params = np.zeros(shape=result.n_values, dtype=np.float64)
        result._params[result.values[result.labels[value]]] = 1
        return result

    def _sample(self, n: int) -> Iterable[Any]:
        '''Returns ``n`` sample `values` according to their respective probability'''
        return wsample(list(self.values.values()), self._params, n)

    def _sample_one(self) -> Any:
        '''Returns one sample `value` according to its probability'''
        return wchoice(list(self.values.values()), self._params)

    def _expectation(self):
        '''Returns the value with the highest probability for this variable'''
        return max([(v, p) for v, p in zip(self.values.values(), self._params)], key=itemgetter(1))[0]

    def expectation(self) -> set:
        """
        For symbolic variables the expectation is equal to the mpe.
        :return: The set of all most likely values
        """
        return self.mpe()[1]

    def mpe(self) -> (float, set):
        """
        Calculate the most probable configuration of this distribution.
        :return: The likelihood of the mpe as float and the mpe itself as Set
        """
        _max = max(self.probabilities)
        return _max, set([label for label, p in zip(self.labels.values(), self.probabilities) if p == _max])

    def _mpe(self) -> (float, set):
        """
        Calculate the most probable configuration of this distribution.
        :return: The likelihood of the mpe as float and the mpe itself as Set
        """
        _max = max(self.probabilities)
        return _max, set([value for value, p in zip(self.value.values(), self.probabilities) if p == _max])

    def k_mpe(self, k: Optional[int] = None) -> List[Tuple[float, set]]:
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

    def _k_mpe(self, k: int) -> List[Tuple[float, set]]:
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


    def kl_divergence(self, other):
        if type(other) is not type(self):
            raise TypeError('Can only compute KL divergence between '
                            'distributions of the same type, got %s' % type(other))
        result = 0
        for v in range(self.n_values):
            result += self._params[v] * abs(self._params[v] - other._params[v])
        return result

    def _crop(self, incl_values=None, excl_values=None):
        if incl_values and excl_values:
            raise Unsatisfiability("Admissible and inadmissible values must be disjoint.")
        posterior = self.copy()
        if incl_values:
            posterior._params[...] = 0
            for i in incl_values:
                posterior._params[int(i)] = self._params[int(i)]
        if excl_values:
            for i in excl_values:
                posterior._params[int(i)] = 0
        try:
            params = normalized(posterior._params)
        except ValueError:
            raise Unsatisfiability('All values have zero probability [%s].' % type(self).__name__)
        else:
            posterior._params = np.array(params)
        return posterior

    def crop(self, restriction: set or List or str):
        """
        Apply a restriction to this distribution such that all values are in the given set.

        :param restriction: The values to remain
        :return: Copy of self that is consistent with the restriction
        """
        if not isinstance(restriction, set) and not isinstance(restriction, list):
            return self.create_dirac_impulse(restriction)

        result = self.copy()
        for idx, value in enumerate(result.labels.keys()):
            if value not in restriction:
                result._params[idx] = 0

        if sum(result._params) == 0:
            raise Unsatisfiability('All values have zero probability [%s].' % type(result).__name__)
        else:
            result._params = result._params / sum(result._params)
        return result

    def _fit(self,
             data: np.ndarray,
             rows: np.ndarray = None,
             col: numbers.Integral = None) -> 'Multinomial':
        self._params = np.zeros(shape=self.n_values, dtype=np.float64)
        n_samples = ifnone(rows, len(data), len)
        col = ifnone(col, 0)
        for row in ifnone(rows, range(len(data))):
            self._params[int(data[row, col])] += 1 / n_samples
        return self

    def set(self, params: Iterable[numbers.Real]) -> 'Multinomial':
        if len(self.values) != len(params):
            raise ValueError('Number of values and probabilities must coincide.')
        self._params = np.array(params, dtype=np.float64)
        return self

    def update(self, dist: 'Multinomial', weight: numbers.Real) -> 'Multinomial':
        if not 0 <= weight <= 1:
            raise ValueError('Weight must be in [0, 1]')
        if self._params is None:
            self._params = np.zeros(self.n_values)
        self._params *= 1 - weight
        self._params += dist._params * weight
        return self

    @staticmethod
    def merge(distributions: Iterable['Multivariate'], weights: Iterable[numbers.Real]) -> 'Multinomial':
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

    @classmethod
    def type_to_json(cls):
        return {
            'type': 'symbolic',
            'class': cls.__qualname__,
            'labels': list(cls.labels.values())
        }

    def inst_to_json(self):
        return {
            'class': type(self).__qualname__,
            'params': list(self._params),
            'settings': self.settings
        }

    to_json = type_to_json

    @staticmethod
    def type_from_json(data):
        return SymbolicType(data['class'], data['labels'])

    @classmethod
    def from_json(cls, data):
        return cls(**data['settings']).set(data['params'])

    def is_dirac_impulse(self):
        for p in self._params:
            if p == 1:
                return True
        return False

    def number_of_parameters(self) -> int:
        """
        :return: The number of relevant parameters in this decision node.
                 1 if this is a dirac impulse, number of parameters else
        """
        if self.is_dirac_impulse():
            return 1
        return len(self._params)

    @classmethod
    def list2set(cls, values: List[str]) -> Set[str]:
        """
        Convert a list to a set.
        """
        return set(values)

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
            ax.set_xlabel('%')
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
            ax.set_ylabel('%')
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


class Bool(Multinomial):
    '''
    Wrapper class for Boolean domains and distributions.
    '''

    values = OrderedDictProxy([(False, 0), (True, 1)])
    labels = OrderedDictProxy([(0, False), (1, True)])

    def __init__(self, **settings):
        super().__init__(**settings)

    def set(self, params: Union[np.ndarray, numbers.Real]) -> 'Bool':
        if params is not None and not iterable(params):
            params = [1 - params, params]
        super().set(params)
        return self

    def __str__(self):
        if self.p is None:
            return f'{self._cl}<p=n/a>'
        return f'{self._cl}<p=[{",".join([f"{v}={p:.3f}" for v, p in zip(self.labels, self._params)])}]>'

    def __setitem__(self, v, p):
        if not iterable(p):
            p = np.array([p, 1 - p])
        super().__setitem__(v, p)


def SymbolicType(name: str, labels: List[Any]) -> Type[Multinomial]:
    if len(labels) < 1:
        raise ValueError('At least one value is needed for a symbolic type.')
    if len(set(labels)) != len(labels):
        duplicates = [item for item, count in Counter(labels).items() if count > 1]
        raise ValueError('List of labels  contains duplicates: %s' % duplicates)
    t = type(name, (Multinomial,), {})
    t.values = OrderedDictProxy([(lbl, int(val)) for val, lbl in zip(range(len(labels)), labels)])
    t.labels = OrderedDictProxy([(int(val), lbl) for val, lbl in zip(range(len(labels)), labels)])
    return t
