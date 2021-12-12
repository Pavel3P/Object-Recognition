import numpy as np
import cv2


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
                 beta: float = 1) -> None:

        self.images = images,
        self.masks = masks
        self.alpha = alpha
        self.beta = beta

    def compose(self) -> np.ndarray:
        raise NotImplementedError
