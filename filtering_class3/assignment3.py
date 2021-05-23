#!/usr/bin/env python3

"""

    Image Processing -- SCC0251/5830 — Prof. Moacir A. Ponti

    Assignment 3 : filtering

    @author: "Erikson Aguiar -- NUSP: 11023222"

                    2021, May -- 1º Semester

"""
import numpy as np
import imageio
from os import listdir

def normalize(image, a, b):
    norm = (image - image.min())*((b - a)/(image.max() - image.min()))+a
    return norm

def filter_1d(imgref, w):
    '''
        1D filter applying in image
        paramters:
         - imageref -- original image
         - w -- 1D filter that apply in image
        return:
            - image filtered
    '''
    n = len(w)
    a = int((n-1)/2)
    M,N = imgref.shape
    
    img_1d = imgref.flatten().astype(float)
    I_hat = np.zeros([len(img_1d)], dtype=imgref.dtype)
    img_1d_pad = np.pad(img_1d, (a,), mode='wrap')

    for x in range(a, len(img_1d)+a):
        I_hat[x-a] = np.sum(np.multiply(img_1d_pad[x-a:x+a+1],w))
    
    Im_hat_new =  I_hat.reshape(M,N)
    Im_hat_norm = normalize(Im_hat_new, 0,255).astype(np.uint8)
    
    return Im_hat_norm
    
def filter_2d(imgref, w):
    '''
        2D filter applying in image
        paramters:
         - imageref -- original image
         - w -- 12D filter that apply in image
        return:
            - image filtered
    '''
    img = imgref.astype(np.float)
    n, m = w.shape
    N,M = img.shape

    a = int((n-1)/2)
    b = int((m-1)/2)

    #I_hat = np.array(imgref, copy=True)
    I_hat = np.zeros((N,M), dtype=float)
    img_pad = np.pad(img, (a,b), mode='symmetric')
    
    for x in range(a,N+a):
        for y in range(b, M+b):
            f_sub = img_pad[x-a:x+a+1, y-b:y+b+1]
            I_hat[x-a,y-b] = np.sum(np.multiply(f_sub, w))

    I_hat_norm = normalize(I_hat, 0, 255).astype(np.uint8)
    
    return I_hat_norm

def median_filter(imgref, n):
    '''
        Median filter applying in image
        paramters:
         - imageref -- original image
         - n -- size filter 2D filter to apply in image
        return:
            - image filtered
    '''
    img = imgref.astype(float)
    N,M = imgref.shape

    a = int(n/2)
    b = int(n/2)

    I_hat = np.zeros([N,M], dtype=img.dtype)
    img_pad = np.pad(img, (a,b))

    for x in range(a,N+a):
        for y in range(b, M+b):
            img_sub = img_pad[x-a:x+a+1, y-b:y+b+1]
            window = img_sub.flatten()
            I_hat[x-a,y-b] = np.median(window)
            
    I_hat_norm = normalize(I_hat, 0, 255).astype(np.uint8)
    
    return I_hat_norm

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

def select_method(imgref, method, parmeter_f):
    '''
        function that select filter method and calculate the rmse between output 
        image  and original image
        parameters:
            - imgref (numpy array) -- original image
            - methof (int) -- selected method:
                    - method 1 -- Filtering 1D 
                    - method 2 -- Filtering 2D
                    - method 3 -- 2D Median Filter
            - parmeter_f -- parameters used in method selected 
        return:
            - rmse - difference between output and original images
    '''
    rmse = 0
    if method == 1:
        I_hat = filter_1d(imgref, parmeter_f)
        rmse = root_mean_square_error(imgref, I_hat)
    elif method == 2:
        I_hat = filter_2d(imgref, parmeter_f)
        rmse = root_mean_square_error(imgref, I_hat)
    elif method == 3:
        I_hat = median_filter(imgref, parmeter_f)
        rmse = root_mean_square_error(imgref, I_hat)
    
    return round(rmse, 4)


if __name__ == '__main__':

    #load images     
    filename = input().rstrip()
    imgref = imageio.imread(filename, as_gray=True)

    #Filter method identifier F (1,2,3)
    #methods F:
    #   - method 1 -- Filtering 1D 
    #   - method 2 -- Filtering 2D
    #   - method 3 -- 2D Median Filter
    method = int(input())

    #filter size
    n = int(input())

    parameters_f = None
    if method == 1:
        param = input().rstrip()
        parameters_f = np.array(param.split(' ')).astype(float)
    elif method == 2:
        f = np.zeros([n,n], dtype=int)
        for r in range(n):
           f[r,:] = input().rstrip().split(' ')
        parameters_f = f.astype(float)
    elif method == 3:
        parameters_f = n

    rmse = select_method(imgref, method, parameters_f)

    print(rmse)






