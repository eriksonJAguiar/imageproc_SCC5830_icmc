#!/usr/bin/env python3

"""

    Image Processing -- SCC0251/5830 — Prof. Moacir A. Ponti

    Assignment 2 : image enhancement and superesolution

    assignment2: Enhancement

    author: "Erikson Aguiar -- NUSP: 11023222"

                    2021, May -- 1º Semester

"""


import numpy as np
import imageio
from os import listdir, path


def histogram(image, no_levels):
    '''
        calculate a histogram from image
        parameters: 
          - image (numpy array) -- image to generate the histogram
          - no_levels (int) -- histogram number of levels e.g, [0-255] -> 256
        return:
            - An 1d array that contains frequency of pixels (numpy array)
    '''
    #histogram
    h = np.zeros(no_levels).astype(int)

    for i in range(no_levels):
        px_value = np.sum(image == i)
        h[i] = px_value
    
    return h

def superresolution(images):
    '''
        Image resolution method
        paramters:
            - images (numpy array) -- An array with the images
        return:
            - a new  image high resolution (numpy array)
    '''
    m,n = images[0].shape
    M,N = n*2, m*2
    h_hat = np.zeros([M, N], dtype=np.uint8)
    l_img = int(len(images)/2)

    i, j = 0,0
    for x in range(m):
        for y in range(n):
            window = list()
            for img in images:
                window.append(img[x,y])
            h_hat[i:i+l_img, j:j+l_img] = np.array(window).reshape(l_img,l_img)
            j = j + l_img
        i = i + l_img
        j = 0
   

    return h_hat

def culmulative_histogram(image, no_levels):
    '''
        Calculate culmulative histogram from an image
        parameters:
            - image (numpy array)
            - no_levels (int) -- levels are considered to calculate histogram
        return:
            - eq (2d numpy array) -- equalized image
            - h_tr (1d numpy array) -- culmulative histogram
    '''
    #image = normalize(image, 0, 255).astype(np.uint8)
    hist = histogram(image, no_levels)

    #hist_c = np.cumsum(hist)
    hist_c = [np.sum(hist[:i+1]) for i in range(len(hist))]

    M, N = image.shape

    eq = np.zeros([M,N], dtype=np.uint8)

    for j in range(no_levels):
        s = ((no_levels - 1)/float(M*N))*hist_c[j]
        eq[np.where(image == j)] = s
    
    #eq = eq.astype(np.int16)
    
    return eq, hist_c

def single_culmulative_histogram(images, no_levels):
    '''
        Calculate the cumulative histogram
        parameters:
            - images (numpy array)
            - no_levels (int) -- histogram number of levels e.g, [0-255] -> 256
        return:
            - eq_imgs -- images equalized from histogram (2d numpy array)
            - hc_imgs -- histogram culmulative (1d numpy array)
    '''
    eq_imgs = list()
    hc_imgs = list()
    
    for img in images:
        eq, hist  = culmulative_histogram(img, no_levels)

        eq_imgs.append(eq)
        hc_imgs.append(hist)
    
    eq_imgs = np.array(eq_imgs)
    hc_imgs = np.array(hc_imgs)
    
    return (eq_imgs, hc_imgs)

def joint_cumulative_histogram(images, no_levels):
    '''
        Calculate the cumulative histogram
        parameters:
            - images (numpy array)
            - no_levels (int) -- histogram number of levels e.g, [0-255] -> 256
        return:
            - eq_all (numpy array) -- list of images equalized
    '''
    img_all = np.concatenate(images)
    _, hist_c_all = culmulative_histogram(img_all, no_levels)

    n_all = 0
    for img in images:
        x, y = img.shape
        n_all = n_all + (x*y)
    
    eq_all = list()
    for img in images:
        M, N = img.shape
        eq = np.zeros([M,N], dtype=np.uint8)

        for j in range(no_levels):
            s = ((no_levels - 1)/float(n_all))*hist_c_all[j]
            eq[np.where(img == j)] = s
        
        eq_all.append(eq)
    
    eq_all = np.array(eq_all)

    return eq_all
        
def gamma_correction_function(images, gamma):
    '''
        Enhancement method using gamma correction
        parameters:
            images (numpy array) -- low images
            gamma (float) -- parameters gamma correction
        return:
            - gamma_imgs (2d numpy array) -- list of images was applied gamma
                                             correction
    '''
    gamma_imgs = list()
    for img in images:
        l_hat = (255*((np.power((img/255.0), (1/gamma)))))
        gamma_imgs.append(l_hat.astype(np.uint8))
    
    gamma_imgs = np.array(gamma_imgs)

    return gamma_imgs

def root_mean_square_error(h_hat, h_ref):
    '''
        Compare enhanced image against reference
        parameters:
            - h (numpy array) -- reference image
            - h_hat (numpy array) -- image enhanced
        return:
            - rmse - root square error, which is difference between them
    '''
    N, _ = h_hat.shape
    h_ref = h_ref.astype(np.int32)
    h_hat = h_hat.astype(np.int32)
    mse = (np.sum(np.square(np.subtract(h_ref, h_hat)))/(N*N))
    rmse = np.sqrt(mse)

    return round(rmse, 4)

def read_all_images(path_files):
    '''
        read image using imageio --> 
        parameters:
            path (string) -- image path
            mathc (string) -- a string pattern to search image in repository
        return:
            all images (numpy array)
    '''
    #filename = path+'"*.png"
    #images = imageio.get_reader(path, mode='I', format='png')
    images = list()
    for f in listdir('./'):
        if f.endswith('.png') and (f.rfind(path_files) >= 0):
            images.append(imageio.imread(f))
    
    images = np.array(images)

    return images

def select_method(imglow, h_ref, method, gamma):
    '''
        function that select method used and calculate the rmse and  
        histogram equalized
        parameters:
            - imglow (numpy array) -- contains all low images
            - h_ref (numpy array) -- the high image used as reference
            - method (int) -- selected method:
                    - method 0 -- No enhancement 
                    - method 1 -- Single-image Cumulative Histogram
                    - method 2 -- Joint Cumulative Histogram
                    - method 3 -- Gamma Correction Function
            - gamma value (float) -- used just in method 3
        return:
            - rmse - difference between reference and enhanced images
    '''
    rmse = 0
    if method == 0:
        s = superresolution(imglow)
        rmse = root_mean_square_error(s, h_ref)
    elif method == 1:
        eq_imgs, _ = single_culmulative_histogram(imglow, 256)
        s = superresolution(eq_imgs)
        rmse = root_mean_square_error(s, h_ref)
    elif method == 2:
        eq_imgs = joint_cumulative_histogram(imglow, 256)
        s = superresolution(eq_imgs)
        rmse = root_mean_square_error(s, h_ref)
    elif method == 3:
        g_imgs = gamma_correction_function(imglow, gamma)
        s = superresolution(g_imgs)
        rmse = root_mean_square_error(s, h_ref)
    
    return round(rmse, 4)
        

if __name__ == '__main__':

    #load low resolution images     
    filename = input().rstrip()
    #imglow = imageio.imread('./ImagensParaTestes/'+filename+'*.png')
    imglow = read_all_images(filename)
    
    #reference image
    filename = input().rstrip()
    h = imageio.imread(filename)

    #enhancement method identifier F (0, 1, 2 or 3)
    #methods F:
    #   - method 0 -- No enhancement 
    #   - method 1 -- Single-image Cumulative Histogram
    #   - method 2 -- Joint Cumulative Histogram
    #   - method 3 -- Gamma Correction Function
    method = int(input())

    #enhancement method parameter gamma, it is used when F = 3
    gamma = float(input())

    rmse = select_method(imglow, h, method, gamma)

    print(rmse)






