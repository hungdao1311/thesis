from sklearn import svm
import numpy as np


def run_weak_classifier(x: np.ndarray, c: svm.SVC) -> int:
    """This is where we execute the weak classifier (could be changed depends on how we use scikit-learn)"""
    x = x.reshape((1, 36))
    return 1 if c.predict(x)[0] == 1 else 0


class Layer:
    def __init__(self, threshold=float(), weak_clf_ensemble=None):
        self.threshold = threshold
        self.weak_clf_ensemble = weak_clf_ensemble

    def predict(self, x: np.array):
        sum_hypotheses = 0.
        sum_alphas = 0.

        for weak_clf in self.weak_clf_ensemble:
            sum_hypotheses += weak_clf['alpha'] * run_weak_classifier(x[weak_clf['feature']], weak_clf['clf'])
            sum_alphas += weak_clf['alpha']

        return 1 if (sum_hypotheses >= self.threshold) else 0
