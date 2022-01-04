import numpy as np


class Recognizer:
    def __init__(self, method: str = "simple", **kwargs) -> None:
        self.method = method

        if self.method == "simple":
            self.samples: dict[str: np.ndarray] = kwargs["samples"]
            self.recognize: callable = self.__recognize_simple
        else:
            raise NotImplementedError

    def __recognize_simple(self, image: np.ndarray, label: str) -> bool:
        if label not in self.samples.keys():
            return False
        elif image.shape != self.samples[label].shape:
            return False
        else:
            return np.all(image == self.samples[label])
