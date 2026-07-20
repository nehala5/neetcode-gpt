import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        n = y_true.size
        ans = []
        tot =0
        epsilon = 1e-7
        for i in range(n):
            loss = y_true[i] * np.log(y_pred[i] + epsilon)
            loss += (1 - y_true[i]) * np.log(1 - y_pred[i] + epsilon)
            tot += loss
        ans = -tot / n
        return round(ans,4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        epsilon = 1e-7
        n = y_true.shape[0]
        total = 0
        for i in range(n):
            for j in range(y_true.shape[1]):
                total += y_true[i][j] * np.log(y_pred[i][j] + epsilon)

        return round(-total / n, 4)
