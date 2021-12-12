import numpy as np
from tqdm import trange


# TODO add tests
# TODO add threading


class Chain:
    def __init__(self,
                 img_chains: list[np.ndarray],
                 msk_chains: list[np.ndarray],
                 alpha: float = 1,
                 beta: float = 1
                 ) -> None:

        self.alpha = alpha
        self.beta = beta
        self.img_chains = img_chains
        self.msk_chains = msk_chains

    def point_weight(self, idx: int, k: int) -> float:
        raise NotImplementedError

    def edge_weight(self, idx1: int, k1: int, idx2: int, k2: int) -> float:
        raise NotImplementedError

    def compose(self) -> np.ndarray:
        raise NotImplementedError


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


