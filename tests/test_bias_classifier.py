import unittest

import numpy as np
from scipy.sparse import random as sparse_random

from src.bias.bias_classifier import BiasClassifier, ClassifierResult


def _make_test_data(n_samples=20, n_features=10, n_classes=3):
    rng = np.random.RandomState(42)
    matrix = sparse_random(n_samples, n_features, density=0.5, random_state=rng).tocsr()
    labels = [f"outlet_{i % n_classes}" for i in range(n_samples)]
    names = [f"feature_{i}" for i in range(n_features)]
    return matrix, labels, names


class BiasClassifierTests(unittest.TestCase):
    def test_run_returns_classifier_result(self):
        matrix, labels, names = _make_test_data()
        classifier = BiasClassifier()

        result = classifier.run(matrix, labels, names)

        self.assertIsInstance(result, ClassifierResult)

    def test_coefficients_shape_matches_features_and_classes(self):
        matrix, labels, names = _make_test_data()
        classifier = BiasClassifier()

        result = classifier.run(matrix, labels, names)

        self.assertEqual(result.coefficients.shape, (10, 3))

    def test_coefficients_index_matches_feature_names(self):
        matrix, labels, names = _make_test_data()
        classifier = BiasClassifier()

        result = classifier.run(matrix, labels, names)

        self.assertEqual(list(result.coefficients.index), names)

    def test_accuracy_is_between_zero_and_one(self):
        matrix, labels, names = _make_test_data()
        classifier = BiasClassifier()

        result = classifier.run(matrix, labels, names)

        self.assertGreaterEqual(result.accuracy, 0.0)
        self.assertLessEqual(result.accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
