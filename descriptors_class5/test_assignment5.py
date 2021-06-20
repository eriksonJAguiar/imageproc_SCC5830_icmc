import unittest
from assignment5 import *
from imageio import imread
from os import path, listdir
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

    def test_constraint_ls_filter(self):

        f_img = imageio.imread('wheres_wally.png')
        g_img = pre_processing(f_img, 3)

        plt.figure(figsize=(12, 12))
        plt.subplot(121)
        plt.imshow(f_img, cmap="gray", vmin=0, vmax=255)
        plt.title('Original')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(g_img, cmap="gray", vmin=0, vmax=255)
        plt.title('Quantise image')
        plt.axis('off')
        plt.show()
    
    def test_descriptor_histogram(self):
        
        f_img = imageio.imread('wheres_wally.png')
        g_img = pre_processing(f_img, 3)

        descritor_hist = normalized_histogram(g_img)

        print(descritor_hist)
    
    def test_descriptor_halarick(self):
        
        f_img = imageio.imread('wheres_wally.png')
        g_img = pre_processing(f_img, 3)

        descritor_hist = descriptor_halarick(g_img)

        print(descritor_hist)


    def test_descriptor_hog(self):
        
        f_img = imageio.imread('wheres_wally.png')
        g_img = pre_processing(f_img, 3)

        descritor_hist = descriptor_hog(g_img)

        print(descritor_hist)
    

if __name__ == '__main__':
    unittest.main()
