#!/usr/bin/env python3

"""

    Image Processing -- SCC5830 — Prof. Moacir A. Ponti

    Assignment 4 : Recovery and Fourier

    @author: "Erikson Aguiar -- NUSP: 11023222"

                    2021, May -- 1º Semester

"""
import numpy as np
import imageio
from os import listdir
from matplotlib import pyplot as plt

def normalize(image, a, b):
    norm = (image - image.min())*((b - a)/(image.max() - image.min()))+a
    return norm

def root_mean_square_error(imgref, I_hat):
    '''
        Compare enhanced image against reference
        parameters:
            - imgref (numpy array) -- original image
            - I_hat (numpy array) -- 
        return:
            - rmse - root square error, which is difference between them
    '''
    N,M = I_hat.shape
    imgref = imgref.astype(np.int32)
    I_hat = I_hat.astype(np.int32)
    rmse = np.sqrt((np.sum(np.square(np.subtract(imgref, I_hat)))/(N*M)))

    return round(rmse, 4)

def estimation_mode(region, mode):
    '''
        calculate estimate dispersion and centrality of region
        parameters:
            - region -- region of image
            - mode -- choose mode (average or robust)
        return:
            - (1) centrality and (2) dispersion values
    '''
    center = 0
    disp = 0
    m,n = region.shape
    if mode == 'average':
        center = np.mean(region)
        disp = np.std(region)
    
    elif mode == 'robust':
        center = np.median(region)
        q75, q25 = np.percentile(region, [75,25])
        disp = q75 - q25
    
    return center, disp

def adaptive_denoising_filtering(g_img,flat_reg, gamma, n, mode):
    '''
        Adaptive denosing filter to restore an image
        parameters:
            - flat_reg -- coordinates of flat rectangle
            - n - size of filter
            - mode - denoising mode "average" or "robuts"
        return:
            - image restored
    '''
    N,M = g_img.shape
    region  = g_img[flat_reg[0]:flat_reg[1], flat_reg[2]:flat_reg[3]]
    a = n // 2
    pad_img = np.pad(g_img, (a,a-1), 'symmetric')
    _, d = estimation_mode(region, mode)
    disp_n = 1 if d == 0 else d
    
    #f_hat = np.array(g_img, copy=True)
    f_hat = np.zeros(g_img.shape)
    for x in np.arange(a, N+a):
        for y in np.arange(a, M+a):
            center, d = estimation_mode(pad_img[x-a:x+a+1,y-a:y+a+1], mode)
            disp_l = disp_n if d == 0 else d
            f_hat[x-a,y-a] = pad_img[x,y] - gamma*(disp_n/disp_l)*(pad_img[x,y]  - center)

    
    return f_hat


def select_method(f_img, g_img, gamma, parameters_f, method):
    '''
        function that select filter method and calculate the rmse between output 
        image  and original image
        parameters:
            - f_img (numpy array) -- reference image
            - g_img (numpy array) -- degraded image
            - methof (int) -- selected method:
                    - method 1 -- Adaptive Denoising
                    - method 2 -- Constraint lest square
            - parmeter_f -- parameters used in method selected 
        return:
            - rmse - difference between output and original images
    '''
    rmse = 0
    if method == 1:
        img_restored = adaptive_denoising_filtering(g_img, parameters_f['flat_reg'], gamma, parameters_f['size'], parameters_f['mode'])
        f_hat = np.clip(img_restored, 0, 255)
        plot(f_img, g_img, f_hat)
        rmse = root_mean_square_error(f_img, f_hat)
    elif method == 2:
        #method 2
        rmse = root_mean_square_error(f_img, I_hat)
    
    return round(rmse, 4)

def plot(f_img, g_img, f_hat):

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


if __name__ == '__main__':

    #load reference image f   
    filename = input().rstrip()
    f_img = imageio.imread(filename, as_gray=True)

    #load degraded image g
    filename = input().rstrip()
    g_img = imageio.imread(filename, as_gray=True)

    #type of filter F (1,2)
    #F = 1 -- denoisilg filter
    #F = 2 -- constrained
    method = int(input())

    
    gamma = float(input())
    
    parameters_f = dict()
    if method == 1:
        param = input().rstrip()
        parameters_f['flat_reg'] = array(param.split(' ')).astype(int)
        parameters_f['size'] = int(input())
        parameters_f['mode'] = input().rstrip()
    elif method == 2:
        parameters_f['size'] = int(input())
        parameters_f['var'] = float(input())

    rmse = select_method(f_img, g_img, gamma, parameters_f)

    print(rmse)






