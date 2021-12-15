import unittest
import time
from HW4.composer import Composer
from HW4.misc import preprocess, save_img


test_dir = 'HW4/data/'
imgs, masks = preprocess(test_dir)


class Test(unittest.TestCase):
    # preprocess testing
    def test_len(self):
        self.assertEqual(len(imgs), len(masks), "Number of images and masks are not equal.")

    def test_ndim(self):
        self.assertTrue(all([mask.ndim == 2 for mask in masks]), "Incorrect masks dimensionality.")
    
    def test_ndim_images(self):
        self.assertTrue(all([im.ndim == 3 for im in imgs]), 'Incorrect images dimensionality.')

    def test_masks_images_size(self):
        self.assertTrue(all([mask.shape == image.shape[:-1] for mask, image in zip(masks, imgs)]),
                        "Some masks don't fit corresponding images.")

    # executing time
    def test_ex_time(self):
        t1 = time.time()
        imgs, masks = preprocess(test_dir)
        ans = Composer(imgs, masks).compose()
        save_img(test_dir + 'result.png', ans)
        self.assertTrue(time.time() - t1 <= 90, "Composer works too slow.")


if __name__ == '__main__':
    unittest.main()
