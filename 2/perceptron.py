import numpy as np
from itertools import product
from sklearn.metrics import accuracy_score


class NormalPerceptron:
    """
    Perceptron for classification elements
    of multivariate normal distribution.
    """

    location: np.ndarray
    covariance: np.ndarray
    weights: np.ndarray
    dims: int

    def __init__(self, learning_rate: float = 1) -> None:
        self.learning_rate = learning_rate

    def train(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.dims = X.shape[1]

        # Initialize distribution params
        self.covariance = np.eye(self.dims)
        self.location = np.zeros(self.dims)
        self.weights = self.__get_weights()

        # Data preprocessing
        preprocessed_x = self.__preprocess_data(X)

        # Get initial accuracy score
        predictions: np.ndarray = self.predict(X)
        accuracy: float = accuracy_score(Y, predictions)
        while accuracy < 1:
            eig_vectors: np.ndarray = self.__preprocess_eig_vectors()
            new_x = np.vstack([preprocessed_x, eig_vectors])
            new_y = np.concatenate([Y, np.ones(len(eig_vectors))])

            for i in np.random.choice(np.arange(len(new_x)), len(new_x), False):
                x = new_x[i]
                y = new_y[i]

                if y * x @ self.weights < 0:
                    self.weights = self.weights + y * self.learning_rate * x

            self.location, self.covariance = self.__get_distribution_params()
            predictions = self.predict(X)
            accuracy = accuracy_score(Y, predictions)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self.__preprocess_data(X)

        predictions: np.ndarray = X @ self.weights
        predictions[predictions > 0] = 1
        predictions[predictions < 0] = -1

        return predictions

    def __preprocess_eig_vectors(self) -> np.ndarray:
        eig_vals, eig_vects = np.linalg.eig(np.linalg.inv(self.covariance))
        eig_vects = eig_vects.T
        eig_vects = eig_vects[eig_vals <= 0]
        eig_vects = self.__preprocess_data(eig_vects)
        eig_vects[:, :self.dims+1] = 0

        return eig_vects

    def __get_weights(self) -> np.ndarray:
        return np.concatenate([
            [self.c],
            -2 * np.linalg.inv(self.covariance) @ self.location,
            np.concatenate(np.linalg.inv(self.covariance))
        ])

    def __get_distribution_params(self) -> tuple[np.ndarray, np.ndarray]:
        covariance: np.ndarray = np.linalg.inv(self.weights[self.dims+1:].reshape((self.dims, self.dims)))
        location: np.ndarray = covariance @ self.weights[1:self.dims+1] / (-2)

        return location, covariance

    @staticmethod
    def __preprocess_data(X: np.ndarray) -> np.ndarray:
        new_x: np.ndarray = np.concatenate([np.ones([X.shape[0], 1]), X], axis=1)

        for i1, i2 in product(range(X.shape[1]), range(X.shape[1])):
            new_x = np.concatenate([new_x, (X[:, i1] * X[:, i2]).reshape([X.shape[0], 1])], axis=1)

        return new_x
