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
        self.learning_rate: float = learning_rate

    def train(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.dims = X.shape[1]

        # Initialize distribution params
        self.covariance = np.eye(self.dims)
        self.location = np.zeros(self.dims)
        self.weights = self.__get_weights()

        # Data preprocessing
        preprocessed_x = self.__preprocess_data(X)

        # Get initial accuracy score
        stop_train: bool = False
        while not stop_train:
            stop_train = True

            # Add eigenvectors constraint
            eig_vectors: np.ndarray = self.__preprocess_eig_vectors()
            new_x = np.vstack([preprocessed_x, eig_vectors])
            new_y = np.concatenate([Y, np.ones(len(eig_vectors))])

            for x, y in zip(new_x, new_y):
                if y * x @ self.weights <= 0:
                    self.weights = self.weights + y * self.learning_rate * x
                    stop_train = False
                    break

            # Calculate new distribution params
            self.location, self.covariance = self.__get_distribution_params()

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self.__preprocess_data(X)

        return self.__predict(X)

    def __predict(self, X: np.ndarray) -> np.ndarray:
        predictions: np.ndarray = X @ self.weights
        predictions[predictions > 0] = 1
        predictions[predictions < 0] = -1

        return predictions

    def __preprocess_eig_vectors(self) -> np.ndarray:
        eig_vals, eig_vects = np.linalg.eig(np.linalg.inv(self.covariance))
        eig_vects = eig_vects.T
        eig_vects = self.__preprocess_data(eig_vects)
        eig_vects[:, :self.dims+1] = 0

        return eig_vects

    def __get_weights(self) -> np.ndarray:
        return np.concatenate([
            [np.log(2 * np.pi) * self.dims],
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
