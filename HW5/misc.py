import cv2
import numpy as np


def construct_image(one: np.ndarray, zero: np.ndarray, l_b, r_b ,c_b):
    digits = {'1': one, '0': zero}

    return cv2.vconcat([
        cv2.hconcat([digits[i] for i in l_b]),
        cv2.hconcat([digits[i] for i in r_b]),
        cv2.hconcat([digits[i] for i in c_b])
    ])


def generate_test_samples(
    one: np.ndarray,
    zero: np.ndarray, 
    path_to_folder: str= None,
    n_true: int=10,
    n_false: int=5
    ) -> dict:
    
    assert one.shape == zero.shape
    
    gen_result = {}
    # true img
    numbers = set(
        map(
            tuple, np.random.randint(0,30, size=(n_true + n_false,2))
        )
    )
    
    mask  = np.zeros(len(numbers))
    mask[:n_true] = 1
    mask = mask.astype(bool)

    for n, flag in zip(numbers, mask):
        l, r = n
        if flag:
            c = l + r 
        else:
            c =l + r - 1

        l_b = "{0:b}".format(l)
        r_b = "{0:b}".format(r)
        c_b = "{0:b}".format(c)

        l_b = '0'*( len(c_b) - len(l_b) ) + l_b
        r_b = '0'*( len(c_b) - len(r_b) ) + r_b

        assert len(l_b) == len(r_b) == len(c_b)

        img = construct_image(one, zero, l_b, r_b, c_b)

        if path_to_folder:
            cv2.imwrite(f'{path_to_folder}/test_{l}+{r}_{c}_{flag}.png', img)

        gen_result[(n, flag)] = img

    return gen_result        
