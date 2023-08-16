import pandas as pd
from typing import List, Any
from math import log
import random

from classifier import DecisionTree


class AdaBoost:

    def __init__(self, n_classifier: int = 5, seed: int = 42) -> None:
        random.seed(seed)

        self.n_classifier: int = n_classifier

        # Function fit will later populate these variables
        self.target_attribute: str = ""  # target attribute of training dataset
        self.classes: List[
            Any
        ] = []  # list of class labels, used later in function _classify_single_tuple()
        self.weights: List[List[float]] = []  # weights of each trained classifier
        self.betas: List[float] = []  # beta values of each trained classifier
        self.classifiers: List[DecisionTree] = []  # list of trained classifiers

    def fit(self, dataset: pd.DataFrame, target_attribute: str) -> None:
        """Fit AdaBoost by training multiple weak learners to the given dataset."""
        # Assign target_attribute to object variable
        self.target_attribute = target_attribute

        # Initialize weights
        self._initialize_weights(dataset=dataset)

        # Get and store number of classes to object variable
        self.classes = dataset[target_attribute].unique().tolist()

        # Get number of tuples in dataset
        n_tuples = dataset.shape[0]

        while True:
            # Stop while-loop when there are enough classifiers
            if len(self.classifiers) == self.n_classifier:
                break

            # Calculate “normalized” weights
            normalized_weights = self._normalize_weights(weights=self.weights[-1])

            # Sample dataset with replacement according to p^k to form training set D_k
            sampled_set = dataset.sample(n=n_tuples, replace=True, weights=normalized_weights)

            # Train a weak learner on the sampled dataset
            weak_classifier = self._weak_learn(sampled_set)

            # Make predictions using the weak classifier
            predictions = weak_classifier.predict(sampled_set)

            # Calculate misclassification errors
            misclassification_errors = []
            for true, predicted in zip(sampled_set[self.target_attribute], predictions):
                misclassification_errors.append(self._misclassification_error(true, predicted))

            # Calculate overall weighted error
            error = 0
            for normalized_weight, misclassification_error in zip(normalized_weights, misclassification_errors):
                error += normalized_weight * misclassification_error

            # If error > 0.5: Abandon this classifier and go back to step 1 (go to next iteration)
            if error > 0.5:
                continue

            # Calculate beta
            beta = error / (1 - error)
            self.betas.append(beta)

            # Update weights for the next iteration
            updated_weights = self._calculate_new_weights(weights=self.weights[-1], beta=beta,
                                                          error=misclassification_errors)
            self.weights.append(updated_weights)

            # Store weak classifier
            self.classifiers.append(weak_classifier)

    def _initialize_weights(self, dataset: pd.DataFrame) -> None:
        """Initialize weights if they have not been initialized before.
        Formula: weights = 1 / number of tuples in data"""
        if not self.weights:
            n_tuples = dataset.shape[0]
            self.weights.append([1 / n_tuples for _ in range(n_tuples)])

    def _weak_learn(self, dataset: pd.DataFrame) -> Any:
        """Fit a weak learner and return this classifier."""
        # Instantiate weak classifier
        classifier = DecisionTree()

        # Train weak classifier
        classifier.fit(dataset=dataset, target_attribute=self.target_attribute)

        return classifier

    @staticmethod
    def _misclassification_error(true: List[float], predicted: List[float]) -> int:
        """Calculate the misclassification error.
        Returns 1 if misclassification (true != predicted), 0 if correct (true == predicted).
        """

        if true != predicted:
            return 1
        else:
            return 0

    @staticmethod
    def _calculate_new_weights(
            weights: List[float], beta: float, error: List[float]
    ) -> List[float]:
        """Update weights by multiplying weights with beta to the power of 1 - error."""

        # return [w * (beta ** (1 - error)) for w in weights]
        new_weights = []
        for weight, error in zip(weights, error):
            new_weights.append(weight * beta ** (1 - error))
        return new_weights

    @staticmethod
    def _normalize_weights(weights: List[float]) -> List[float]:
        """Normalize weights. Formula: weights = weights / sum(weights)"""

        normalized_weights = []
        weights_sum = sum(weights)

        if weights_sum == 0:
            for _ in weights:
                normalized_weights.append(0)
        else:
            for weight in weights:
                normalized_weights.append(weight / weights_sum)

        return normalized_weights

    def predict(self, dataset: pd.DataFrame) -> List[Any]:
        """Return prediction for a given dataset."""
        return [
            self._classify_single_tuple(dataset_tuple.to_frame().T)
            for _, dataset_tuple in dataset.iterrows()
        ]

    def _classify_single_tuple(self, dataset_tuple: pd.DataFrame) -> Any:
        """Classifies a single tuple."""

        weights = {c: 0 for c in self.classes}

        for beta, classifier in zip(self.betas, self.classifiers):
            predictions = classifier.predict(dataset_tuple)
            for prediction in predictions:
                weights[prediction] += log(1 / beta)
        return max(weights, key=weights.get)
