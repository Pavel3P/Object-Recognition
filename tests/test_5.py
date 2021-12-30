import unittest
import cv2

from HW5.rules import Rules
from HW5.misc import generate_test_samples
from HW5.recognizer import Recognizer
from HW5.cyk import CYK
from HW5.rules_variables import (horizontal,
            vertical, rename, terminal, nonterminal)


path_to_zero = "HW5/terminal_symbols/0_w7-h8.png"
path_to_one = "HW5/terminal_symbols/1_w7-h8.png"

zero_sample = cv2.imread(path_to_zero, cv2.IMREAD_GRAYSCALE)
one_sample = cv2.imread(path_to_one, cv2.IMREAD_GRAYSCALE)

samples = {
        "0": zero_sample,
        "1": one_sample
    }

rules = Rules()

for h in horizontal:
    rules.create_gh(*h)

for v in vertical:
    rules.create_gv(*v)

for r in rename:
    rules.create_g(*r)

recognizer = Recognizer(samples=samples)
cyk = CYK(rules, recognizer, terminal, nonterminal, "i")
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
