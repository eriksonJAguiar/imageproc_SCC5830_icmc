#!/usr/bin/env python3

"""

    Image Processing -- SCC5830 — Prof. Moacir A. Ponti

    Assignment 4 : Recovery and Fourier

    @author: "Erikson Aguiar -- NUSP: 11023222"

                    2021, May -- 1º Semester

"""
import numpy as np
import imageio
from scipy.fft import rfft2, irfft2, fftshift

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

def gaussian_filter(k=3, sigma=1.0):
    '''
        gaussian filter
        paramters:
            - k -- filter size
            - sigma -- standard deviation of distribution
        return:
            - filter
    '''
    arx = np.arange((-k // 2) + 1.0, (k // 2) + 1.0)
    x, y = np.meshgrid(arx, arx)
    filt = np.exp(-(1/2)*(np.square(x) + np.square(y))/np.square(sigma))

    return filt/np.sum(filt)

def constrained_least_square_filtering(g_img, gamma, k, sigma):
    '''
        Constrained least square filtering
        parameters:
            - g_img -- degraded image
            - gamma -- gamma value
            - k -- size of filter
            - sigma -- standard deviation of filter
        return:
            - f_hat -- restored image
    '''
    h_filt = gaussian_filter(k, sigma)
    pad_s = (g_img.shape[0]//2) - h_filt.shape[0]//2
    h_filt_pad = np.pad(h_filt, (pad_s, pad_s-1))
    p = np.array([[0,-1,0], [-1,4,-1], [0, -1, 0]], dtype=float)
    pad_s = (g_img.shape[0]//2) - p.shape[0]//2
    p_pad = np.pad(p, (pad_s, pad_s-1))
    
    H = rfft2(h_filt_pad)
    G = rfft2(g_img)
    P = rfft2(p_pad)

    F_hat = np.multiply(np.divide(np.conj(H),(np.power(H, 2) + np.multiply(gamma,np.power(P,2)))), G)
    f_hat = np.real(fftshift(irfft2(F_hat)))

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
        rmse = root_mean_square_error(f_img, f_hat)
    elif method == 2:
        img_restored = constrained_least_square_filtering(g_img, gamma, parameters_f['size'], parameters_f['sigma'])
        f_hat = np.clip(img_restored, 0, 255)
        rmse = root_mean_square_error(f_img, f_hat)
    
    return round(rmse, 4)

if __name__ == '__main__':

    #load reference image f   
    filename = input().rstrip()
    f_img = imageio.imread(filename, as_gray=True)

    #load degraded image g
    filename = input().rstrip()
    g_img = imageio.imread(filename, as_gray=True)

    #type of filter F (1,2)
    #F = 1 -- denoising filter
    #F = 2 -- constrained ls
    method = int(input())

    gamma = float(input())
    
    parameters_f = dict()
    if method == 1:
        param = input().rstrip()
        parameters_f['flat_reg'] = np.array(param.split(' ')).astype(int)
        parameters_f['size'] = int(input())
        parameters_f['mode'] = input().rstrip()
    elif method == 2:
        parameters_f['size'] = int(input())
        parameters_f['sigma'] = float(input())

    rmse = select_method(f_img, g_img, gamma, parameters_f, method)

    print(rmse)






