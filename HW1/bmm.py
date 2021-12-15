import numpy as np
from typing import Tuple, List


class BMM:
    """
    Expectation maximization algorithm for
    Bernoulli mixture models
    """

    def __init__(self, clusters_num: int) -> None:
        self.clusters_num: int = clusters_num
        self.mixing_coeffs: np.ndarray = None
        self.params: np.ndarray = None

    def train(self, samples: np.ndarray, iters_num: int) -> None:
        # Random initialization of mixing coefficients
        self.mixing_coeffs = np.random.random(self.clusters_num)
        self.mixing_coeffs = self.mixing_coeffs / np.sum(self.mixing_coeffs)

        # Random initialization of parameters
        params_num: int = samples.shape[1]
        self.params: np.ndarray = np.random.uniform(.1, .9, (self.clusters_num, params_num))

        for i in range(iters_num):
            # E-step
            predictions: np.ndarray = self._e_step(samples)

            # M-step
            self.mixing_coeffs, self.params = self._m_step(samples, predictions)

    def predict(self, samples: np.ndarray) -> np.ndarray:
        exps: np.ndarray = self._e_step(samples)

        return np.argmax(exps, axis=1)

    def _e_step(self, samples: np.ndarray) -> np.ndarray:
        # Implementation of formula (1) from README
        def prob(x: np.ndarray, k: int) -> float:
            mu = self.params[k]
            p = self.mixing_coeffs[k]

            return p * np.prod(mu ** x * (1 - mu) ** (1 - x))

        predictions: np.ndarray = np.zeros((samples.shape[0], self.clusters_num))

        for i in range(samples.shape[0]):
            x: np.ndarray = samples[i]
            probs: List[float] = list(map(lambda k: prob(x, k), range(self.clusters_num)))

            for k in range(self.clusters_num):
                predictions[i, k] = probs[k] / np.sum(probs)

        return predictions

    def _m_step(self, samples: np.ndarray, predictions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Implementation of formulae (3) and (5) from README
        new_mixture_coeffs: np.ndarray = np.sum(predictions, axis=0) / predictions.shape[0]

        # Implementation of formulae (2) and (4) from README
        new_params: np.ndarray = samples.T @ predictions / np.sum(predictions, axis=0)

        return new_mixture_coeffs, new_params.T
