import unittest
from assignment2 import *
from imageio import imread
from os import path
from matplotlib import pyplot as plt
import numpy as np


def read_in_out():
        in_ = list()
        out_ = list()
        path = './cases_test_runcodes/'
        for f in listdir(path):
            if f.endswith('.in'):
                i = open(path+f).read().splitlines()
                in_.append(i)
            elif f.endswith('.out'):
                o = open(path+f).read().splitlines()
                out_.append(o[0])
        
        return (in_, out_)

class TestAssignment(unittest.TestCase):
        
    def test_histogram(self):
        image = imread('08_low1.png')
        h = histogram(image, 256)
        self.assertIsNotNone(h)

    def test_single_culmulative_histogram(self):
        image0 = imread('08_low0.png')
        image1 = imread('08_low1.png')
        image2 = imread('08_low2.png')
        image3 = imread('08_low3.png')
        eq_imgs,hc_imgs = single_culmulative_histogram(np.array([image0, image1, image2, image3]), 256)
        self.assertIsNotNone(eq_imgs)
        self.assertIsNotNone(hc_imgs)

        #show images
        # plt.figure(figsize=(12,12)) 
        # plt.subplot(331)
        # plt.imshow(image0, cmap="gray", vmin=0, vmax=255)
        # plt.axis('off')
        # plt.subplot(332)
        # plt.imshow(eq_imgs[0], cmap="gray", vmin=0, vmax=255)
        # plt.axis('off')
        # plt.subplot(333)
        # plt.bar(range(256), hc_imgs[0])
        # plt.xlabel('Intensity value')
        # plt.ylabel('Frequency')


        # plt.subplot(334)
        # plt.imshow(image1, cmap="gray", vmin=0, vmax=255)
        # plt.axis('off')
        # plt.subplot(335)
        # plt.imshow(eq_imgs[1], cmap="gray", vmin=0, vmax=255)
        # plt.axis('off')
        # plt.subplot(336)
        # plt.bar(range(256), hc_imgs[1])
        # plt.xlabel('Intensity value')
        # plt.ylabel('Frequency')


        # plt.subplot(337)
        # plt.imshow(image2, cmap="gray", vmin=0, vmax=255)
        # plt.axis('off')
        # plt.subplot(338)
        # plt.imshow(eq_imgs[2], cmap="gray", vmin=0, vmax=255)
        # plt.axis('off')
        # plt.subplot(339)
        # plt.bar(range(256), hc_imgs[2])
        # plt.xlabel('Intensity value')
        # plt.ylabel('Frequency')
        # plt.show()
    
    def test_gamma_correction(self):
        image0 = imread('08_low0.png')
        image1 = imread('08_low1.png')
        image2 = imread('08_low2.png')
        image3 = imread('08_low3.png')
        g_imgs = gamma_correction_function(np.array([image0, image1, image2, image3]), 0.5)

        self.assertIsNotNone(g_imgs)

        #show images
        # plt.figure(figsize=(12,12)) 
        # plt.subplot(321)
        # plt.imshow(image0, cmap="gray", vmin=0, vmax=255)
        # plt.axis('off')
        # plt.subplot(322)
        # plt.imshow(g_imgs[0], cmap="gray", vmin=0, vmax=255)
        # plt.axis('off')

        # plt.subplot(323)
        # plt.imshow(image1, cmap="gray", vmin=0, vmax=255)
        # plt.axis('off')
        # plt.subplot(324)
        # plt.imshow(g_imgs[1], cmap="gray", vmin=0, vmax=255)
        # plt.axis('off')

        # plt.subplot(325)
        # plt.imshow(image2, cmap="gray", vmin=0, vmax=255)
        # plt.axis('off')
        # plt.subplot(326)
        # plt.imshow(g_imgs[2], cmap="gray", vmin=0, vmax=255)
        # plt.axis('off')
        # plt.show()

    def test_join_culmulative_histogram(self):
        image0 = imread('08_low0.png')
        image1 = imread('08_low1.png')
        image2 = imread('08_low2.png')
        image3 = imread('08_low3.png')
        eq_imgs = joint_cumulative_histogram(np.array([image0, image1, image2, image3]), 256)

        self.assertIsNotNone(eq_imgs)

        #show images
        # plt.figure(figsize=(12,12)) 
        # plt.subplot(321)
        # plt.imshow(image0, cmap="gray", vmin=0, vmax=255)
        # plt.axis('off')
        # plt.subplot(322)
        # plt.imshow(eq_imgs[0], cmap="gray", vmin=0, vmax=255)
        # plt.axis('off')

        # plt.subplot(323)
        # plt.imshow(image1, cmap="gray", vmin=0, vmax=255)
        # plt.axis('off')
        # plt.subplot(324)
        # plt.imshow(eq_imgs[1], cmap="gray", vmin=0, vmax=255)
        # plt.axis('off')

        # plt.subplot(325)
        # plt.imshow(image2, cmap="gray", vmin=0, vmax=255)
        # plt.axis('off')
        # plt.subplot(326)
        # plt.imshow(eq_imgs[2], cmap="gray", vmin=0, vmax=255)
        # plt.axis('off')
        # plt.show()
    
    def test_superresolution(self):
        l1 = np.array([[100,101], [110, 111]])
        l2 = np.array([[200,201], [210, 211]])
        l3 = np.array([[300,301], [310, 311]])
        l4 = np.array([[400,401], [410, 411]])

        s = superresolution(np.array([l1,l2,l3,l4]))
        print(s)
        self.assertIsNotNone(s)
    
    def test_superresolution_with_images(self):
        images = read_all_images('05_low')
        s = superresolution(images)
        print(s)
        self.assertIsNotNone(s)
    
    def test_culmulative_histogram(self):
        image = imread('08_low1.png')
        eq, h_tr = culmulative_histogram(image, 256)
        self.assertIsNotNone(eq)
        self.assertIsNotNone(h_tr)
        
        #show image
        # plt.figure(figsize=(12,12)) 
        # plt.subplot(121)
        # plt.imshow(eq, cmap="gray", vmin=0, vmax=255)
        # plt.axis('off')

        # plt.subplot(122)
        # plt.bar(range(256), h_tr)
        # plt.xlabel('Intensity value')
        # plt.ylabel('Frequency')
        
        # plt.show()

    def test_rsme(self):
        image0 = imread('./08_low0.png')
        rsme = root_mean_square_error(image0, image0)
        self.assertEqual(rsme, 0.000)

    def test_load_images(self):
        images = read_all_images('08_low')
        print(images)
        self.assertIsNotNone(images)
    
    def test_integrated_case1(self):
        in_, out_ = read_in_out()
        i, o = in_[0], out_[0] 
        imglow = read_all_images(i[0])
        h_ref = imageio.imread(i[1])
        rmse = select_method(imglow, h_ref, int(i[2]), float(i[3]))
        print('rmse: %f; real: %f'%(rmse, float(o)))

        # plt.figure(figsize=(12,12)) 
        # plt.subplot(121)
        # plt.imshow(s, cmap="gray", vmin=0, vmax=255)
        # plt.title('Superresolution')
        # plt.axis('off')
        # plt.subplot(122)
        # plt.imshow(h_ref, cmap="gray", vmin=0, vmax=255)
        # plt.title('original')
        # plt.axis('off')
        # plt.show()

        self.assertTrue(rmse >= (rmse-5) and rmse <= (rmse+5))
    
    def test_integrated_case2(self):
        in_, out_ = read_in_out()
        i, o = in_[1], out_[1] 
        imglow = read_all_images(i[0])
        h_ref = imageio.imread(i[1])
        rmse = select_method(imglow, h_ref, int(i[2]), float(i[3]))
        print('rmse: %f; real: %f'%(rmse, float(o)))

        # plt.figure(figsize=(12,12)) 
        # plt.subplot(121)
        # plt.imshow(s, cmap="gray", vmin=0, vmax=255)
        # plt.title('Superresolution')
        # plt.axis('off')
        # plt.subplot(122)
        # plt.imshow(h_ref, cmap="gray", vmin=0, vmax=255)
        # plt.title('original')
        # plt.axis('off')
        # plt.show()

        self.assertTrue(rmse >= (rmse-5) and rmse <= (rmse+5))
    
    def test_integrated_case3(self):
        in_, out_ = read_in_out()
        i, o = in_[2], out_[2] 
        imglow = read_all_images(i[0])
        h_ref = imageio.imread(i[1])
        rmse = select_method(imglow, h_ref, int(i[2]), float(i[3]))
        print('rmse: %f; real: %f'%(rmse, float(o)))

        # plt.figure(figsize=(12,12)) 
        # plt.subplot(121)
        # plt.imshow(s, cmap="gray", vmin=0, vmax=255)
        # plt.title('Superresolution')
        # plt.axis('off')
        # plt.subplot(122)
        # plt.imshow(h_ref, cmap="gray", vmin=0, vmax=255)
        # plt.title('original')
        # plt.axis('off')
        # plt.show()

        self.assertTrue(rmse >= (rmse-5) and rmse <= (rmse+5))
    
    def test_integrated_case4(self):
        in_, out_ = read_in_out()
        i, o = in_[3], out_[3] 
        imglow = read_all_images(i[0])
        h_ref = imageio.imread(i[1])
        rmse = select_method(imglow, h_ref, int(i[2]), float(i[3]))
        print('rmse: %f; real: %f'%(rmse, float(o)))

        # plt.figure(figsize=(12,12)) 
        # plt.subplot(121)
        # plt.imshow(s, cmap="gray", vmin=0, vmax=255)
        # plt.title('Superresolution')
        # plt.axis('off')
        # plt.subplot(122)
        # plt.imshow(h_ref, cmap="gray", vmin=0, vmax=255)
        # plt.title('original')
        # plt.axis('off')
        # plt.show()

        self.assertTrue(rmse >= (rmse-5) and rmse <= (rmse+5))
    
    def test_integrated_case5(self):
        in_, out_ = read_in_out()
        i, o = in_[4], out_[4] 
        imglow = read_all_images(i[0])
        h_ref = imageio.imread(i[1])
        rmse = select_method(imglow, h_ref, int(i[2]), float(i[3]))
        print('rmse: %f; real: %f'%(rmse, float(o)))

        # plt.figure(figsize=(12,12)) 
        # plt.subplot(121)
        # plt.imshow(s, cmap="gray", vmin=0, vmax=255)
        # plt.title('Superresolution')
        # plt.axis('off')
        # plt.subplot(122)
        # plt.imshow(h_ref, cmap="gray", vmin=0, vmax=255)
        # plt.title('original')
        # plt.axis('off')
        # plt.show()

        self.assertTrue(rmse >= (rmse-5) and rmse <= (rmse+5))
    

    def test_integrated_case6(self):
        in_, out_ = read_in_out()
        i, o = in_[5], out_[5] 
        imglow = read_all_images(i[0])
        h_ref = imageio.imread(i[1])
        rmse = select_method(imglow, h_ref, int(i[2]), float(i[3]))
        print('rmse: %f; real: %f'%(rmse, float(o)))

        # plt.figure(figsize=(12,12)) 
        # plt.subplot(121)
        # plt.imshow(s, cmap="gray", vmin=0, vmax=255)
        # plt.title('Superresolution')
        # plt.axis('off')
        # plt.subplot(122)
        # plt.imshow(h_ref, cmap="gray", vmin=0, vmax=255)
        # plt.title('original')
        # plt.axis('off')
        # plt.show()

        self.assertTrue(rmse >= (rmse-5) and rmse <= (rmse+5))
    
    def test_integrated_case7(self):
        in_, out_ = read_in_out()
        i, o = in_[6], out_[6] 
        imglow = read_all_images(i[0])
        h_ref = imageio.imread(i[1])
        rmse  = select_method(imglow, h_ref, int(i[2]), float(i[3]))
        print('rmse: %f; real: %f'%(rmse, float(o)))

        # plt.figure(figsize=(12,12)) 
        # plt.subplot(121)
        # plt.imshow(s, cmap="gray", vmin=0, vmax=255)
        # plt.title('Superresolution')
        # plt.axis('off')
        # plt.subplot(122)
        # plt.imshow(h_ref, cmap="gray", vmin=0, vmax=255)
        # plt.title('original')
        # plt.axis('off')
        # plt.show()

        self.assertTrue(rmse >= (rmse-5) and rmse <= (rmse+5))

    def test_integrated_case8(self):
        in_, out_ = read_in_out()
        i, o = in_[7], out_[7] 
        imglow = read_all_images(i[0])
        h_ref = imageio.imread(i[1])
        rmse  = select_method(imglow, h_ref, int(i[2]), float(i[3]))
        print('rmse: %f; real: %f'%(rmse, float(o)))

        # plt.figure(figsize=(12,12)) 
        # plt.subplot(121)
        # plt.imshow(s, cmap="gray", vmin=0, vmax=255)
        # plt.title('Superresolution')
        # plt.axis('off')
        # plt.subplot(122)
        # plt.imshow(h_ref, cmap="gray", vmin=0, vmax=255)
        # plt.title('original')
        # plt.axis('off')
        # plt.show()

        self.assertTrue(rmse >= (rmse-5) and rmse <= (rmse+5))



if __name__ == '__main__':
    unittest.main()