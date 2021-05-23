import unittest
from assignment3 import *
from imageio import imread
from os import path
from matplotlib import pyplot as plt
import numpy as np

def read_in_out():
    in_ = list()
    out_ = list()
    path = './CasosDeTeste/'
    for f in listdir(path):
        if f.endswith('.in'):
            i = open(path+f).read().splitlines()
            in_.append(i)
        elif f.endswith('.out'):
            o = open(path+f).read().splitlines()
            out_.append(o[0])

    return (in_, out_)


class TestAssignment(unittest.TestCase):

    def test_filter_1d(self):

        imgref = imageio.imread('arara.png', as_gray=True)
        w = np.array([-2, -1, 0, 1, 2])
        img_hat = filter_1d(imgref, w)

        plt.figure(figsize=(12, 12))
        plt.subplot(121)
        plt.imshow(imgref, cmap="gray", vmin=0, vmax=255)
        plt.title('Original')
        plt.axis('off')
        plt.colorbar()
        plt.subplot(122)
        plt.imshow(img_hat, cmap="gray", vmin=0, vmax=255)
        plt.title('Filtered')
        plt.axis('off')
        plt.colorbar()
        plt.show()
        print(root_mean_square_error(imgref, img_hat))

    def test_filter_2d(self):

        imgref = imageio.imread('image02_quant.png')
        w = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]])
        img_hat = filter_2d(imgref, w)

        plt.figure(figsize=(12, 12))
        plt.subplot(121)
        plt.imshow(imgref, cmap="gray", vmin=0, vmax=255)
        plt.title('Original')
        plt.axis('off')
        plt.colorbar()
        plt.subplot(122)
        plt.imshow(img_hat, cmap="gray", vmin=0, vmax=255)
        plt.title('Filtered')
        plt.axis('off')
        plt.colorbar()
        plt.show()
        print(root_mean_square_error(imgref, img_hat))
    
    def test_filter_median(self):

        imgref = imageio.imread('image02_salted.png').astype(np.uint8)
        img_hat = median_filter(imgref, 5)
        #img_hat = ndimage.median_filter(imgref, 5)

        plt.figure(figsize=(12, 12))
        plt.subplot(121)
        plt.imshow(imgref, cmap="gray", vmin=0, vmax=255)
        plt.colorbar()
        plt.title('Original')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(img_hat, cmap="gray", vmin=0, vmax=255)
        plt.title('Filtered')
        plt.colorbar()
        plt.axis('off')
        plt.show()
        print(root_mean_square_error(imgref, img_hat))

    def test_integrated_case1(self):
        in_, out_ = read_in_out()
        i,o = in_[0], out_[0]
        method = int(i[1])
        imgref = imageio.imread(i[0], as_gray=True)
        w = np.array(i[3].split(' ')).astype(int)
        rmse = select_method(imgref, method, w)
        print('rmse: %f; real: %f' % (rmse, float(o)))
        self.assertTrue(rmse >= (float(o)-5) and rmse <= (float(o)+5))

    def test_integrated_case2(self):
        in_, out_ = read_in_out()
        i,o = in_[1], out_[1]
        method = int(i[1])
        imgref = imageio.imread(i[0], as_gray=True)
        w = int(i[2])
        rmse = select_method(imgref, method, w)
        print('rmse: %f; real: %f' % (rmse, float(o)))
        self.assertTrue(rmse >= (float(o)-5) and rmse <= (float(o)+5))
    
    def test_integrated_case3(self):
        in_, out_ = read_in_out()
        i,o = in_[2], out_[2]
        method = int(i[1])
        imgref = imageio.imread(i[0], as_gray=True)
        w_aux = []
        for row in range(3,len(i)):
            w_aux.append(i[row].split(' '))
        w = np.array(w_aux).astype(int)
        rmse = select_method(imgref, method, w)
        print('rmse: %f; real: %f' % (rmse, float(o)))
        self.assertTrue(rmse >= (float(o)-5) and rmse <= (float(o)+5))

    
    def test_integrated_case4(self):
        in_, out_ = read_in_out()
        i,o = in_[3], out_[3]
        method = int(i[1])
        imgref = imageio.imread(i[0], as_gray=True)
        w = np.array(i[3].split(' ')).astype(int)
        rmse = select_method(imgref, method, w)
        print('rmse: %f; real: %f' % (rmse, float(o)))
        self.assertTrue(rmse >= (float(o)-5) and rmse <= (float(o)+5))
    
    def test_integrated_case5(self):
        in_, out_ = read_in_out()
        i,o = in_[4], out_[4]
        method = int(i[1])
        imgref = imageio.imread(i[0], as_gray=True)
        w = int(i[2])
        rmse = select_method(imgref, method, w)
        print('rmse: %f; real: %f' % (rmse, float(o)))
        self.assertTrue(rmse >= (float(o)-5) and rmse <= (float(o)+5))
    
    def test_integrated_case6(self):
        in_, out_ = read_in_out()
        i,o = in_[5], out_[5]
        method = int(i[1])
        imgref = imageio.imread(i[0], as_gray=True)
        w_aux = []
        for row in range(3,len(i)):
            w_aux.append(i[row].split(' '))
        w = np.array(w_aux).astype(int)
        rmse = select_method(imgref, method, w)
        print('rmse: %f; real: %f' % (rmse, float(o)))
        self.assertTrue(rmse >= (float(o)-5) and rmse <= (float(o)+5))
    
    def test_integrated_case7(self):
        in_, out_ = read_in_out()
        i,o = in_[6], out_[6]
        method = int(i[1])
        imgref = imageio.imread(i[0], as_gray=True)
        w_aux = []
        for row in range(3,len(i)):
            w_aux.append(i[row].split(' '))
        w = np.array(w_aux).astype(int)
        rmse = select_method(imgref, method, w)
        print('rmse: %f; real: %f' % (rmse, float(o)))
        self.assertTrue(rmse >= (float(o)-5) and rmse <= (float(o)+5))

    def test_integrated_case8(self):
        in_, out_ = read_in_out()
        i,o = in_[7], out_[7]
        method = int(i[1])
        imgref = imageio.imread(i[0], as_gray=True)
        w = int(i[2])
        rmse = select_method(imgref, method, w)
        print('rmse: %f; real: %f' % (rmse, float(o)))
        self.assertTrue(rmse >= (float(o)-5) and rmse <= (float(o)+5))


if __name__ == '__main__':
    unittest.main()
