import unittest
import cv2
from algorithms.cyk import CYK, generate_test_samples
import numpy as np


def recognizer(image: np.ndarray, label: str, samples: dict[str: np.ndarray]) -> bool:
    if label not in samples.keys():
        return False
    elif image.shape != samples[label].shape:
        return False
    else:
        return np.all(image == samples[label])


horizontal = [
    ('i', 'i', 'v0'),
    ('i', 'i_', 'v0_'),
    ('i', 'v0', 'v0'),
    ('i', 'v0_', 'v0'),
    ('i', 'v1', 'v0_'),
    ('i', 'v1_', 'v0_'),

    ('i_', 'i', 'v1'),
    ('i_', 'i_', 'v1_'),
    ('i_', 'v0', 'v1'),
    ('i_', 'v0_', 'v1'),
    ('i_', 'v1', 'v1_'),
    ('i_', 'v1_', 'v1_')
]

vertical = [
    ('A00', '0', '0'),
    ('A01', '0', '1'),
    ('A10', '1', '0'),
    ('A11', '1', '1'),

    ('v0', 'A00', '0'),
    ('v0', 'A01', '1'),
    ('v0', 'A10', '1'),

    ('v0_', 'A11', '0'),

    ('v1', 'A00', '1'),

    ('v1_', 'A01', '0'),
    ('v1_', 'A10', '0'),
    ('v1_', 'A11', '1')
]

rename = [
    ('i', 'v0'),
    ('i_', 'v1')
]

terminal = ["0", "1"]
nonterminal = ["A00", "A01", "A10", "A11", "v0", "v1", "v0_", "v1_", "i", "i_"]

path_to_zero = "data/cyk/0_w7-h8.png"
path_to_one = "data/cyk/1_w7-h8.png"

zero_sample = cv2.imread(path_to_zero, cv2.IMREAD_GRAYSCALE)
one_sample = cv2.imread(path_to_one, cv2.IMREAD_GRAYSCALE)

samples = {
        "0": zero_sample,
        "1": one_sample
    }

cyk = CYK(lambda img, l: recognizer(img, l, samples), terminal, nonterminal, "i", horizontal, vertical, rename)
height = zero_sample.shape[0]
width = zero_sample.shape[1]


class Test(unittest.TestCase):
    def test_main(self):
        tests = generate_test_samples(one=one_sample, zero=zero_sample, path_to_folder=None, n_true=1, n_false=1)
        for info, image in tests.items():
            self.assertEqual(
                cyk(
                    image, w_step=width,
                    h_step=height, min_h=height,
                    min_w=width, window_size_h_step=height,
                    window_size_w_step=width),
                info[1], f"{info} test failed")


if __name__ == '__main__':
    unittest.main()
