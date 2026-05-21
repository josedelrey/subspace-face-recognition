import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def error_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return 1.0 - accuracy_score(y_true, y_pred)