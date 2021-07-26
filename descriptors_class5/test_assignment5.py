import unittest

from assignment5 import *
from imageio import imread
from os import path, listdir
from matplotlib import pyplot as plt
import  matplotlib.patches as patches
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

def show_img(img, i, j):
    fig, ax = plt.subplots()
    ax.imshow(img)
    rect = patches.Rectangle((j * 16, i * 16), 32,32, linewidth =1, edgecolor= 'r' , facecolor='none')
    ax.add_patch(rect)
    plt.show()

class TestAssignment(unittest.TestCase):

    def test_constraint_ls_filter(self):

        f_img = imageio.imread('wheres_jumpingguy.png')
        g_img = pre_processing(f_img, 1)

        plt.figure(figsize=(12, 12))
        plt.subplot(121)
        plt.imshow(f_img, cmap="gray", vmin=0, vmax=255)
        plt.title('Original')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(g_img, cmap="binary", vmin=0, vmax=1)
        plt.title('Quantise image')
        plt.axis('off')
        plt.show()
    
    
    def test_descriptor_histogram(self):
        
        f_img = imageio.imread('wheres_jumpingguy.png')
        g_img = pre_processing(f_img, 1)

        descritor_hist = normalized_histogram(g_img, 1)

        print(descritor_hist)
    
    def test_descriptor_halarick(self):
        
        f_img = imageio.imread('wheres_jumpingguy.png')
        g_img = pre_processing(f_img, 1)

        descritor_hack = descriptor_halarick(g_img)

        print(descritor_hack)


    def test_descriptor_hog(self):
        
        f_img = imageio.imread('wheres_jumpingguy.png')
        g_img = pre_processing(f_img, 1)

        hog = descriptor_hog(g_img)
        print(hog)
    
    def test_find_objects(self):
        f_img = imageio.imread('wheres_wally.png')
        obj_g = imageio.imread('wally.png')

        w, r = find_object(f_img, obj_g, 8)
        print("{} {}".format(w,r))
        show_img(f_img, w, r)

    def test_find_object1(self):
        f_img = imageio.imread('./ImagesForDebugging/wheres_wally.png')
        obj_g = imageio.imread('./ImagesForDebugging/wally.png')

        w, r = find_object(f_img, obj_g, 3)
        print("{} {}".format(w,r))
        show_img(f_img, w, r)

        self.assertTrue((w == 12 and r == 12))
    
    def test_find_object2(self):
        f_img = imageio.imread('./ImagesForDebugging/wheres_blueumbrella.png')
        obj_g = imageio.imread('./ImagesForDebugging/blueumbrella.png')

        w, r = find_object(f_img, obj_g, 7)
        print("{} {}".format(w,r))
        show_img(f_img, w, r)

        self.assertTrue((w == 13 and r == 13))
    
    def test_find_object3(self):
        f_img = imageio.imread('./ImagesForDebugging/wheres_jumpingguy.png')
        obj_g = imageio.imread('./ImagesForDebugging/jumpingguy.png')

        w, r = find_object(f_img, obj_g, 1)
        print("{} {}".format(w,r))
        show_img(f_img, w, r)

        self.assertTrue((w == 6 and r == 8))
    
    def test_find_object4(self):
        f_img = imageio.imread('./ImagesForDebugging/wheres_pokeball.png')
        obj_g = imageio.imread('./ImagesForDebugging/pokeball.png')

        w, r = find_object(f_img, obj_g, 1)
        print("{} {}".format(w, r))
        show_img(f_img, w, r)

        self.assertTrue((w == 1 and r == 8))

    def test_find_object5(self):
        f_img = imageio.imread('./ImagesForDebugging/wheres_pumpkin.png')
        obj_g = imageio.imread('./ImagesForDebugging/pumpkin.png')

        w, r = find_object(f_img, obj_g, 8)
        print("{} {}".format(w,r))
        show_img(f_img, w, r)

        self.assertTrue((w == 6 and r == 11))


if __name__ == '__main__':
    unittest.main()
