import unittest
from assignment4 import *
from imageio import imread
from os import path
from matplotlib import pyplot as plt
import numpy as np

def read_in_out():
    in_ = list()
    out_ = list()
    path = './TestCases/'
    for f in listdir(path):
        if f.endswith('.in'):
            i = open(path+f).read().splitlines()
            in_.append(i)
        elif f.endswith('.out'):
            o = open(path+f).read().splitlines()
            out_.append(o[0])

    return (in_, out_)


class TestAssignment(unittest.TestCase):

    def test_adaptive_filter(self):

        f_img = imageio.imread('moon_small.jpg')
        g_img = imageio.imread('case3_70.png')
        gamma = 0.8
        flat_reg = np.array([0,20,0,30])
        n = 3
        mode = 'robust'
        img_restored = adaptive_denoising_filtering(g_img, flat_reg, gamma,n, mode)
        f_hat = np.clip(img_restored.astype(int), 0, 255)

        plt.figure(figsize=(12, 12))
        plt.subplot(131)
        plt.imshow(f_img, cmap="gray", vmin=0, vmax=255)
        plt.title('Original')
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(g_img, cmap="gray", vmin=0, vmax=255)
        plt.title('Degraded')
        plt.axis('off')
        plt.subplot(133)
        plt.imshow(f_hat, cmap="gray", vmin=0, vmax=255)
        plt.title('Filtered')
        plt.show()
        print(root_mean_square_error(f_img, f_hat))
    
    def test_integreted_case1(self):
        in_, out_ = read_in_out()
        i,o = in_[0], out_[0]
        f_img = imageio.imread(i[0])
        g_img = imageio.imread(i[1])
        gamma = float(i[3])
        param = dict()
        param['flat_reg'] = np.array(i[4].split(' ')).astype(int)
        param['size'] = int(i[5])
        param['mode'] = i[6]
        rmse = select_method(f_img, g_img, gamma, param, 1)
        print('rmse: %f; real: %f' % (rmse, float(o)))
        self.assertTrue(rmse >= (float(o)-10) and rmse <= (float(o)+10))
    
    def test_integreted_case2(self):
        in_, out_ = read_in_out()
        i,o = in_[1], out_[1]
        f_img = imageio.imread(i[0])
        g_img = imageio.imread(i[1])
        gamma = float(i[3])
        param = dict()
        param['flat_reg'] = np.array(i[4].split(' ')).astype(int)
        param['size'] = int(i[5])
        param['mode'] = i[6]
        rmse = select_method(f_img, g_img, gamma, param, 1)
        print('rmse: %f; real: %f' % (rmse, float(o)))
        self.assertTrue(rmse >= (float(o)-10) and rmse <= (float(o)+10))
    
    def test_integreted_case3(self):
        in_, out_ = read_in_out()
        i,o = in_[2], out_[2]
        f_img = imageio.imread(i[0])
        g_img = imageio.imread(i[1])
        gamma = float(i[3])
        param = dict()
        param['flat_reg'] = np.array(i[4].split(' ')).astype(int)
        param['size'] = int(i[5])
        param['mode'] = i[6]
        rmse = select_method(f_img, g_img, gamma, param, 1)
        print('rmse: %f; real: %f' % (rmse, float(o)))
        self.assertTrue(rmse >= (float(o)-10) and rmse <= (float(o)+10))
    
    def test_integreted_case4(self):
        in_, out_ = read_in_out()
        i,o = in_[3], out_[3]
        f_img = imageio.imread(i[0])
        g_img = imageio.imread(i[1])
        gamma = float(i[3])
        param = dict()
        param['flat_reg'] = np.array(i[4].split(' ')).astype(int)
        param['size'] = int(i[5])
        param['mode'] = i[6]
        rmse = select_method(f_img, g_img, gamma, param, 1)
        print('rmse: %f; real: %f' % (rmse, float(o)))
        self.assertTrue(rmse >= (float(o)-10) and rmse <= (float(o)+10))
    

if __name__ == '__main__':
    unittest.main()
