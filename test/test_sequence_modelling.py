import unittest

#raise unittest.SkipTest('Skip the sequential JPT tests until the errors in this module have been fixed.')

import numpy as np
import jpt.variables
from jpt.learning.distributions import SymbolicType
from jpt.sequential_jpt import SequentialJPT


class UniformSeries:

    def __init__(self, basis_function=np.sin, epsilon=0.05):
        self.epsilon = 0.05
        self.basis_function = basis_function

    def sample(self, samples) -> np.array:
        samples = self.basis_function(samples)
        samples = samples + np.random.uniform(-self.epsilon, self.epsilon, samples.shape)
        return samples


class SequenceTest(unittest.TestCase):

    def setUp(self) -> None:
        self.g = UniformSeries()
        self.data = np.expand_dims(self.g.sample(np.arange(np.pi / 2, 10000, np.pi)), -1)
        self.variables = [jpt.variables.NumericVariable("X", precision=0.1)]

    def test_learning(self):
        tree = SequentialJPT(self.variables, min_samples_leaf=1500)
        tree.learn([self.data, self.data])

    def test_integral(self):
        tree = SequentialJPT(self.variables, min_samples_leaf=1500)
        tree.learn([self.data])
        # tree.plot(plotvars=tree.variables)
        self.assertAlmostEqual(tree.probability_mass_, 0.5)

    def test_likelihood(self):
        tree = SequentialJPT(self.variables, min_samples_leaf=500)
        tree.learn([self.data])
        samples = np.expand_dims(np.array([[1., -1., 1.], [-1., 1., -1.]]), 2)
        l = tree.likelihood(samples)
        self.assertAlmostEqual(l[0], l[1])

    def test_infer(self):
        tree = SequentialJPT(self.variables, min_samples_leaf=1500)
        tree.learn([self.data])
        # tree.plot(plotvars=tree.variables, directory="/tmp/mcjpt")
        q_0 = {self.variables[0]: [0.95, 1.05]}
        q_1 = {self.variables[0]: [-1.05, -0.95]}

        p = tree.infer(queries=[q_0,q_1,q_0, q_1], evidences=[dict(), dict(), dict(),dict()])

        #for leaf_combo, distributions in tree.shared_dimensions_integral.items():
         #   for variable, distribution in distributions.items():
          #      print(leaf_combo, distribution.cdf.intervals, distribution.cdf.functions)
           #     distribution.plot(title=str(leaf_combo))
            #    plt.show()

        self.assertAlmostEqual(p, 0.5, places=2)


class DiscreteSequenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.g = UniformSeries()
        self.data = np.around(np.expand_dims(self.g.sample(np.arange(np.pi / 2, 10000, np.pi)), -1)).astype(str)
        x = SymbolicType('DigitType', [-1, 1])
        self.variables = [jpt.variables.SymbolicVariable("X", domain=x)]

    def test_learning(self):
        tree = SequentialJPT(self.variables, min_samples_leaf=1500)
        tree.learn([self.data, self.data])

    def test_integral(self):
        tree = SequentialJPT(self.variables, min_samples_leaf=1500)
        tree.learn([self.data])
        self.assertAlmostEqual(tree.probability_mass_, 0.5)

    def test_infer(self):
        tree = SequentialJPT(self.variables, min_samples_leaf=1500)
        tree.learn([self.data])
        q_0 = {self.variables[0]: 1}
        q_1 = {self.variables[0]: -1}

        p = tree.infer(queries=[q_1, q_0, q_1, ], evidences=[dict(), dict(), dict()])
        self.assertAlmostEqual(p, 0.5)

if __name__ == '__main__':
    unittest.main()