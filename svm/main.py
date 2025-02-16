import numpy as np
from typing import Union
from sklearn.datasets import load_iris


class SVM:
    """SVM model"""

    def __init__(
        self,
        learning_rate: float = 1e-3,
        lambda_param: float = 1e-2,
        n_iters: int = 1000,
    ) -> None:
        """
        initialize the SVM model.
        Args:
            learning_rate (float): the step size
            lambda_param (float): regularization parameter
            n_iters (int): number of iterations

        Raises:
            ValueError: if learning_rate or n_iters is less than or equal to 0
            ValueError: if n_iters is less than or equal to 0
        """

        if learning_rate <= 0:
            raise ValueError("learning rate must be greater than 0")

        if n_iters <= 0:
            raise ValueError("number of iterations must be greater than 0")

        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights: Union[np.ndarray, None] = None
        self.bias: Union[float, None] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        train the SVM model
        Args:
            X (np.ndarray): training features
            y (np.ndarray): target values
        """
        _, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)

        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):

            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (
                        2 * self.lambda_param * self.weights
                    )
                else:
                    self.weights -= self.learning_rate * (
                        2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx])
                    )
                    self.bias -= self.learning_rate * y_[idx]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        predict the target values
        Args:
            X (np.ndarray): features
        Returns:
            np.ndarray: predicted values
        Raises:
            ValueError: if model is not trained
        """

        if self.weights is None or self.bias is None:
            raise ValueError("model is not trained")

        return np.sign(np.dot(X, self.weights) - self.bias)


if __name__ == "__main__":

    data = load_iris()
    X, y = data.data[:, :2], data.target
    y = np.where(y == 0, -1, 1)

    svm = SVM()
    svm.fit(X, y)

    test_data = np.array([[0, 0], [7.5, 3.5]])
    predictions = svm.predict(test_data)
    print(predictions)
