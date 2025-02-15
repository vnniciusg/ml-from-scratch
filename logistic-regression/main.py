from typing import Union
import numpy as np


def sigmoid(X: np.ndarray) -> float:
    """
    sigmoid function
    Args:
        X (np.ndarray): input values

    Returns:
        np.ndarray: sigmoid of input values
    """
    return 1 / (1 + np.exp(-X))


class LogisticRegression:
    """logistic regression model"""

    def __init__(self, learning_rate: float = 0.01, n_iters: int = 1000) -> None:
        """
        initialize the logistic regression model.

        Args:
            learning_rate (float): the step size
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
        self.n_iters = n_iters
        self.weights: Union[np.ndarray, None] = None
        self.bias: Union[float, None] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        train the logistic regression model

        Args:
            X (np.ndarray): training features
            y (np.ndarray): target values
        """

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_pred = self._forward(X)
            predictions = sigmoid(linear_pred)

            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def _forward(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias

    def predict(self, X: np.ndarray) -> float:
        """
        make predictions for input data

        Args:
            X (np.ndarray): input features

        Returns:
            float: predicted values

        Raises:
            ValueError: if model is not trained
        """

        if self.bias is None or self.weights is None:
            raise ValueError("model must be trained before predict new values")

        linear_pred = self._forward(X)
        predictions = sigmoid(linear_pred)

        class_prediction = [0 if y <= 0.5 else 1 for y in predictions]

        return class_prediction


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    calculate the accuracy of the model.

    Args:
        y_pred (np.ndarray): the predicted values from the model
        y_true (np.ndarray): the actual values from the target variable

    Returns:
        float: the accuracy percentage of the model
    """
    return np.sum(y_pred == y_true) / len(y_true)


if __name__ == "__main__":

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    bc_dataset = load_breast_cancer()

    X, y = bc_dataset.data, bc_dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

    y_pred = log_reg.predict(X_test)
    print("accuracy:", accuracy(y_pred, y_test))
