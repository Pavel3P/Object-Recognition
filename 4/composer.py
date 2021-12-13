import numpy as np
from misc import memoize
from tqdm import trange


# TODO add tests
# TODO add threading


class Chain:
    def __init__(self,
                 img_chains: np.ndarray,
                 msk_chains: np.ndarray,
                 alpha: float = 1,
                 beta: float = 1
                 ) -> None:

        self.alpha = alpha
        self.beta = beta
        self.img_chains = img_chains
        self.msk_chains = msk_chains

        self.__weights = self.alpha * (1 - self.msk_chains)
        self.__class_num = img_chains.shape[0]
        self.__pts_num = self.img_chains.shape[1]

    def point_weight(self,
                     idx: int,
                     k: int) -> float:
        if idx >= self.__pts_num or k >= self.__class_num or idx == -1 or k == -1:
            return np.inf

        return self.__weights[k, idx]

    @memoize
    def edge_weight(self,
                    idx1: int,
                    idx2: int,
                    k1: int,
                    k2: int) -> float:
        if abs(idx1 - idx2) != 1 and (idx1 < 0 or idx2 < 0):
            return np.inf

        return self.beta * (np.sum(np.abs(self.img_chains[k1, idx1] - self.img_chains[k2, idx1])) +
                            np.sum(np.abs(self.img_chains[k1, idx2] - self.img_chains[k2, idx2])))

    def compose(self) -> np.ndarray:
        _, labels = self.__f()

        return np.array([self.img_chains[labels[i], i] for i in range(self.__pts_num)])

    @memoize
    def __f(self, idx: int = 0, k: int = -1) -> tuple[float, list[int]]:
        if idx >= self.__pts_num or k >= self.__class_num:
            return 0, []

        if k == -1:
            k_range = range(self.__class_num)
        else:
            k_range = [k]

        k_min_val = np.inf
        best_labels = []
        for k in k_range:
            k_next_min_val = np.inf
            best_next_labels = []
            for k_next in range(self.__class_num):
                f_next, next_labels = self.__f(idx+1, k_next)
                val = self.point_weight(idx+1, k+1) + self.edge_weight(idx, idx+1, k, k_next) + f_next
                if val < k_next_min_val:
                    best_next_labels = next_labels
                    k_next_min_val = val

            if k_next_min_val < k_min_val:
                best_labels = [k] + best_next_labels
                k_min_val = k_next_min_val

        return k_min_val, best_labels


class Composer:
    def __init__(self,
                 images: list[np.ndarray],
                 masks: list[np.ndarray],
                 alpha: float = 1,
                 beta: float = 1
                 ) -> None:

        self.images = np.vstack(images),
        self.masks = np.vstack([m.astype(bool) for m in masks])
        self.alpha = alpha
        self.beta = beta

    def compose(self) -> np.ndarray:
        res: list[np.ndarray] = []
        for i in trange(self.images[0].shape[0]):
            # ch = Chain(self.images[:, i], self.masks[:, i])
            # line = ch.compose()
            # del ch
            # res.append(line)
            res.append(
                Chain(self.images[:, i], self.masks[:, i]).compose()
            )

        return np.array(res)


