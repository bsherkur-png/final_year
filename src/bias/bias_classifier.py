from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score


@dataclass
class ClassifierResult:
    """Output of the bias classifier."""

    coefficients: pd.DataFrame
    classification_report: str
    accuracy: float


class BiasClassifier:
    """Multinomial logistic regression using outlet as proxy label."""

    def __init__(self):
        self._model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        )

    def run(
            self,
            feature_matrix: csr_matrix,
            labels: list[str],
            feature_names: list[str],
    ) -> ClassifierResult:
        """Train and evaluate the classifier, then return coefficients and metrics."""
        label_array = np.array(labels)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            self._model,
            feature_matrix,
            label_array,
            cv=cv,
            scoring="accuracy",
        )
        accuracy = float(cv_scores.mean())

        self._model.fit(feature_matrix, label_array)

        predictions = self._model.predict(feature_matrix)
        report = classification_report(label_array, predictions)

        coef_df = pd.DataFrame(
            self._model.coef_.T,
            index=feature_names,
            columns=self._model.classes_,
        )

        return ClassifierResult(
            coefficients=coef_df,
            classification_report=report,
            accuracy=accuracy,
        )