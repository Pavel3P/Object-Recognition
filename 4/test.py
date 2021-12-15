import unittest
import time
from composer import Composer
from misc import preprocess, save_img


test_dir = 'data/'
imgs, masks = preprocess(test_dir)


class Test(unittest.TestCase):
    # preprocess testing
    def test_len(self):
        self.assertEqual(len(imgs), len(masks), "len should be the same")

    def test_ndim(self):
        self.assertEqual(all([mask.ndim == 2 for mask in masks]), True, "One dim mask")
    
    def test_ndim_images(self):
        self.assertEqual(all([im.ndim == 3 for im in imgs]), True, '3 dim images')
    
    # executing time
    def test_ex_time(self):
        t1 = time.time()
        imgs, masks = preprocess(test_dir)
        ans = Composer(imgs, masks).compose()
        save_img(test_dir + 'result.png', ans)
        self.assertEqual(time.time() - t1 <= 90, True)


if __name__ == '__main__':
    unittest.main()
