import math
from collections import deque
from typing import Iterable, Optional, Tuple, Union, Any, List

import numpy as np
import pandas as pd
import portion
from probabilistic_model.probabilistic_circuit.units import DeterministicSumUnit, DecomposableProductUnit
from probabilistic_model.probabilistic_model import ProbabilisticModel
from probabilistic_model.probabilistic_circuit.distributions import SymbolicDistribution, IntegerDistribution, \
    DiracDeltaDistribution
from probabilistic_model.learning.nyga_distribution import NygaDistribution
from random_events.events import VariableMap
from random_events.variables import Variable, Continuous as REContinuous, Integer as REInteger, Symbolic

try:
    from .learning.impurity import Impurity
except ModuleNotFoundError:
    import pyximport

    pyximport.install()
finally:
    from .learning.impurity import Impurity


def infer_variables_from_dataframe(data: pd.DataFrame, scale_continuous_types: bool = True) -> List[Variable]:
    """
    Infer the variables from a dataframe.
    The variables are inferred by the column names and types of the dataframe.

    :param data: The dataframe to infer the variables from.
    :param scale_continuous_types: Whether to scale numeric types.
    :return: The inferred variables.
    """
    result = []

    for column, datatype in zip(data.columns, data.dtypes):

        # handle continuous variables
        if datatype in [float]:

            minimal_distance_between_values = np.diff(np.sort(data[column].unique())).min()
            mean = data[column].mean()
            std = data[column].std()

            if scale_continuous_types:
                variable = ScaledContinuous(column, mean, std, minimal_distance_between_values)
            else:
                variable = Continuous(column, mean, std, minimal_distance_between_values)

        # handle discrete variables
        elif datatype in [object, int]:

            unique_values = data[column].unique()

            if datatype == int:
                mean = data[column].mean()
                std = data[column].std()
                variable = Integer(column, unique_values, mean, std)
            elif datatype == object:
                variable = Symbolic(column, unique_values)
            else:
                raise ValueError(f"Datatype {datatype} of column {column} is not supported.")

        else:
            raise ValueError(f"Datatype {datatype} of column {column} is not supported.")

        result.append(variable)

    return result


class Integer(REInteger):
    mean: float
    """
    Mean of the random variable.
    """

    std: float
    """
    Standard Deviation of the random variable.
    """

    def __init__(self, name: str, domain: Iterable, mean, std):
        super().__init__(name, domain)
        self.mean = mean
        self.std = std


class Continuous(REContinuous):
    """
    Base class for continuous variables in JPTs. This class does not standardize the data,
    but needs to know mean and std anyway.
    """

    minimal_distance: float
    """
    The minimal distance between two values of the variable.
    """

    mean: float
    """
    Mean of the random variable.
    """

    std: float
    """
    Standard Deviation of the random variable.
    """

    min_likelihood_improvement: float
    """
    The minimum likelihood improvement passed to the Nyga Distributions.
    """

    min_samples_per_quantile: int
    """
    The minimum number of samples per quantile passed to the Nyga Distributions.
    """

    def __init__(self, name: str, mean: float, std: float, minimal_distance: float = 1.,
                 min_likelihood_improvement: float = 0.1, min_samples_per_quantile: int = 10):
        super().__init__(name)
        self.mean = mean
        self.std = std
        self.minimal_distance = minimal_distance
        self.min_likelihood_improvement = min_likelihood_improvement
        self.min_samples_per_quantile = min_samples_per_quantile


class ScaledContinuous(Continuous):
    """
    A continuous variable that is standardized.
    """

    def __init__(self, name: str, mean: float, std: float, minimal_distance: float = 1.):
        super().__init__(name, mean, std, minimal_distance)

    def encode(self, value: Any):
        return (value - self.mean) / self.std

    def decode(self, value: float) -> float:
        return value * self.std + self.mean

    def __str__(self):
        return f"{self.__class__.__name__}({self.name}, {self.mean}, {self.std}, {self.minimal_distance})"


class Criterion:
    """
    A criterion that is used to decide which branch to take in a decision node.
    """

    variable: Variable
    value: Union[portion.Interval, Tuple]

    def __init__(self, variable: Variable, value: Union[portion.Interval, Tuple]):
        self.variable = variable
        self.value = value


class DecisionNode(DeterministicSumUnit):
    criterion: Criterion
    """
    The criterion that is used to decide which branch to take.
    """

    def __init__(self, variables: Iterable[Variable], weights: Iterable, criterion: Criterion):
        super().__init__(variables, weights)
        self.criterion = criterion


class JPT(DeterministicSumUnit):

    targets: Tuple[Variable]
    """
    The variables to optimize for.
    """

    features: Tuple[Variable]
    """
    The variables that are used to craft criteria.
    """

    _min_samples_leaf: Union[int, float]
    """
    The minimum number of samples to create another sum node. If this is smaller than one, it will be reinterpreted
    as fraction w. r. t. the number of samples total.
    """

    min_impurity_improvement: float
    """
    The minimum impurity improvement to create another sum node.
    """

    max_leaves: Union[int, float]
    """
    The maximum number of leaves.
    """

    max_depth: Union[int, float]
    """
    The maximum depth of the tree.
    """

    dependencies: VariableMap
    """
    The dependencies between the variables.
    """

    total_samples: int = 1
    """
    The total amount of samples that were used to fit the model.
    """

    indices: Optional[np.ndarray] = None
    impurity: Optional[Impurity] = None
    c45queue: deque = deque()
    weights: List[float]

    def __init__(self, variables: Iterable[Variable], targets: Optional[Iterable[Variable]] = None,
                 features: Optional[Iterable[Variable]] = None, min_samples_leaf: Union[int, float] = 1,
                 min_impurity_improvement: float = 0.0, max_leaves: Union[int, float] = float("inf"),
                 max_depth: Union[int, float] = float("inf"), dependencies: Optional[VariableMap] = None, ):

        super().__init__(variables, weights=[])
        self.set_targets_and_features(targets, features)
        self._min_samples_leaf = min_samples_leaf
        self.min_impurity_improvement = min_impurity_improvement
        self.max_leaves = max_leaves
        self.max_depth = max_depth

        if dependencies is None:
            self.dependencies = VariableMap({var: list(self.targets) for var in self.features})
        else:
            self.dependencies = dependencies

    def set_targets_and_features(self, targets: Optional[Iterable[Variable]],
                                 features: Optional[Iterable[Variable]]) -> None:
        """
        Set the targets and features of the model.
        If only one of them is provided, the other is set as the complement of the provided one.
        If none are provided, both of them are set as the variables of the model.
        If both are provided, they are taken as given.

        :param targets: The targets of the model.
        :param features: The features of the model.
        :return: None
        """
        # if targets are not specified
        if targets is None:

            # and features are not specified
            if features is None:
                self.targets = self.variables
                self.features = self.variables

            # and features are specified
            else:
                self.targets = tuple(sorted(set(self.variables) - set(features)))
                self.features = tuple(sorted(features))

        # if targets are specified
        else:
            # and features are not specified
            if features is None:
                self.targets = tuple(sorted(set(targets)))
                self.features = tuple(sorted(set(self.variables) - set(targets)))

            # and features are specified
            else:
                self.targets = tuple(sorted(set(targets)))
                self.features = tuple(sorted(set(features)))

    def __eq__(self, other):
        return (isinstance(other, JPT) and
                self.variables == other.variables and
                self.targets == other.targets and
                self.features == other.features and
                self.children == other.children and
                self.min_impurity_improvement == other.min_impurity_improvement and
                self.max_depth == other.max_depth)

    @property
    def min_samples_leaf(self):
        """
        The minimum number of samples to create another sum node.
        """
        if self._min_samples_leaf < 1.:
            return math.ceil(self._min_samples_leaf * self.total_samples)
        else:
            return self._min_samples_leaf

    @property
    def numeric_variables(self):
        return [variable for variable in self.variables if isinstance(variable, (Continuous, Integer))]

    @property
    def numeric_targets(self):
        return [variable for variable in self.targets if isinstance(variable, (Continuous, Integer))]

    @property
    def numeric_features(self):
        return [variable for variable in self.features if isinstance(variable, (Continuous, Integer))]

    @property
    def symbolic_variables(self):
        return [variable for variable in self.variables if isinstance(variable, Symbolic)]

    @property
    def symbolic_targets(self):
        return [variable for variable in self.targets if isinstance(variable, Symbolic)]

    @property
    def symbolic_features(self):
        return [variable for variable in self.features if isinstance(variable, Symbolic)]

    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess the data to be used in the model.

        :param data: The data to preprocess.
        :return: The preprocessed data.
        """

        result = np.zeros(data.shape)

        for variable_index, variable in enumerate(self.variables):
            column = data[variable.name]
            column = variable.encode_many(column)
            result[:, variable_index] = column

        return result

    def fit(self, data: pd.DataFrame) -> 'JPT':
        """
        Fit the model to the data.

        :param data: The data to fit the model to.
        :return: The fitted model.
        """

        preprocessed_data = self.preprocess_data(data)

        self.total_samples = len(preprocessed_data)

        self.indices = np.ascontiguousarray(np.arange(preprocessed_data.shape[0], dtype=np.int64))
        self.impurity = self.construct_impurity()
        self.impurity.setup(preprocessed_data, self.indices)

        self.c45queue.append((preprocessed_data, 0, len(preprocessed_data), 0))

        while self.c45queue:
            self.c45(*self.c45queue.popleft())

        return self

    def c45(self, data: np.ndarray, start: int, end: int, depth: int):
        """
        Construct a DecisionNode or DecomposableProductNode from the data.

        :param data: The data to calculate the impurity from.
        :param start: Starting index in the data.
        :param end: Ending index in the data.
        :param depth: The current depth of the induction
        :return: The constructed decision tree node
        """
        number_of_samples = end - start

        # if the inducing in this step would result in inadmissible nodes, skip the impurity calculation
        if depth >= self.max_depth or number_of_samples < self.min_samples_leaf:
            max_gain = -float("inf")
        else:
            max_gain = self.impurity.compute_best_split(start, end)

        # if the max gain is insufficient
        if max_gain <= self.min_impurity_improvement:

            # create decomposable product node
            leaf_node = self.create_leaf_node(data[self.indices[start:end]])
            self.weights.append(number_of_samples/len(data))
            leaf_node.parent = self

            # terminate the induction
            return

        # if the max gain is sufficient
        split_pos = self.impurity.best_split_pos

        # increase the depth
        new_depth = depth + 1

        # append the new induction steps
        self.c45queue.append((data, start, start + split_pos + 1, new_depth))
        self.c45queue.append((data, start + split_pos + 1, end, new_depth))

    def create_leaf_node(self, data: np.ndarray) -> DecomposableProductUnit:
        result = DecomposableProductUnit(self.variables)

        for index, variable in enumerate(self.variables):
            if isinstance(variable, Continuous):
                distribution = NygaDistribution(variable,
                                                min_likelihood_improvement=variable.min_likelihood_improvement,
                                                min_samples_per_quantile=variable.min_samples_per_quantile)
                distribution._fit(data[:, index].tolist())

                if isinstance(distribution.children[0], DiracDeltaDistribution):
                    distribution.children[0].density_cap = 1/variable.minimal_distance

            elif isinstance(variable, Symbolic):
                distribution = SymbolicDistribution(variable, weights=[1/len(variable.domain)]*len(variable.domain))
                distribution._fit(data[:, index].tolist())

            elif isinstance(variable, Integer):
                distribution = IntegerDistribution(variable, weights=[1/len(variable.domain)]*len(variable.domain))
                distribution._fit(data[:, index].tolist())
            else:
                raise ValueError(f"Variable {variable} is not supported.")

            distribution.parent = result

        return result

    def construct_impurity(self) -> Impurity:
        min_samples_leaf = self.min_samples_leaf

        numeric_vars = (
            np.array([index for index, variable in enumerate(self.variables) if variable in self.numeric_targets]))
        symbolic_vars = np.array(
            [index for index, variable in enumerate(self.variables) if variable in self.symbolic_targets])

        invert_impurity = np.array([0] * len(self.symbolic_targets))

        n_sym_vars_total = len(self.symbolic_variables)
        n_num_vars_total = len(self.numeric_variables)

        numeric_features = np.array(
            [index for index, variable in enumerate(self.variables) if variable in self.numeric_features])
        symbolic_features = np.array(
            [index for index, variable in enumerate(self.variables) if variable in self.symbolic_features])

        symbols = np.array([len(variable.domain) for variable in self.symbolic_variables])
        max_variances = np.array([variable.std ** 2 for variable in self.numeric_variables])

        dependency_indices = dict()

        for variable, dep_vars in self.dependencies.items():
            # get the index version of the dependent variables and store them
            idx_var = self.variables.index(variable)
            idc_dep = [self.variables.index(var) for var in dep_vars]
            dependency_indices[idx_var] = idc_dep

        return Impurity(min_samples_leaf, numeric_vars, symbolic_vars, invert_impurity, n_sym_vars_total,
                        n_num_vars_total, numeric_features, symbolic_features, symbols, max_variances,
                        dependency_indices)
