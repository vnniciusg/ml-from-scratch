import numpy as np
from typing import Union


class LogisticRegression:
    """logistic regression using gradient descent"""

    def __init__(self, learning_rate: float = 0.01, n_iters: int = 1000) -> None:
        """
        initialize the linear regression model with gradient descent.

        Args:
            learning_rate (float): the step size
            n_iters (int): number of iterations
        """
        if learning_rate <= 0:
            raise ValueError("learning rate must be greater than 0")
        if n_iters <= 0:
            raise ValueError("number of iterations must be greater than 0")

        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weigths: Union[np.ndarray, None] = None
        self.bias: Union[float, None] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        fit the linear regression model

        Args:
            X (np.ndarray): input features of shape (n_samples, n_features)
            y (np.ndarray): target values of shape (n_samples, )
        """

        n_samples, n_features = X.shape

        self.weigths = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):

            y_pred: np.ndarray = self._forward(X)

            error: np.ndarray = y_pred - y

            grad_weights: np.ndarray = (2 / n_samples) * np.dot(X.T, error)
            grad_bias: float = (2 / n_samples) * np.sum(error)

            self.weigths -= self.learning_rate * grad_weights
            self.bias -= self.learning_rate * grad_bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        predict target values for given input features

        Args:
            X (np.ndarray): input features of shape (n_samples, n_features)

        Returns:
            np.ndarray: Predicted values of shape (n_samples, )
        """
        if self.weigths is None or self.bias is None:
            raise ValueError("model must be fitted before making predictions.")

        return self._forward(X)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """compute forward pass"""
        return np.dot(X, self.weigths) + self.bias


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    calculate the mean squared error (mse) between true and predicted values.
    Args:
        y_true (np.ndarray): actual target values
        y_pred (np.ndarray): predicted target values
    Returns:
        float: the MSE value
    """
    return np.mean((y_true - y_pred) ** 2)


if __name__ == "__main__":

    np.random.seed(42)

    X: np.ndarray = 2 * np.random.rand(100, 1)
    y: np.ndarray = 4 + 3 * X.squeeze() + np.random.randn(100)

    model = LogisticRegression(learning_rate=0.01, n_iters=1000)
    model.fit(X, y)

    predictions: np.ndarray = model.predict(X)

    mse = mean_squared_error(y, predictions)

    print("mean squared error:", mse)
