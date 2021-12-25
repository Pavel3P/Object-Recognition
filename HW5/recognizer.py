import numpy as np


class Recognizer:
    def __init__(self, method: str = "simple", *args, **kwargs) -> None:
        raise NotImplementedError

    def __call__(self, image: np.ndarray, label: int) -> bool:
        raise NotImplementedError
