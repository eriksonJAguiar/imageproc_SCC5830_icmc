#!/usr/bin/env python3

"""

    Image Processing -- SCC5830 — Prof. Moacir A. Ponti

    Assignment 5 : image descriptors

    @author: "Erikson Aguiar -- NUSP: 11023222"

                    2021, June -- 1º Semester

"""
import numpy as np
import imageio
from numpy.core.fromnumeric import shape
from numpy.core.numeric import NaN
import scipy.ndimage

def pre_processing(img, b):
    '''
        method for preprocessing image and quantise to b bits
        paramters:
            - img: reference image
            - b: bit to quantise
        return:
            - p_img: precessed image
    '''
    lm_img = np.floor(((0.299*img[:,:,2])+(0.587*img[:,:,1])+(0.114*img[:,:,0]))).astype(np.int)
    q_img = np.right_shift(lm_img, (8 - b))

    return q_img

def normalized_histogram(img):
    '''
        calculate normalized histogram of image
        paramters:
            - img: reference image
        return
            - dc: descritor vector
    '''
    levels = np.unique(img.flatten())
    hk_sum  = img.shape[0]*img.shape[1]
    hist = np.zeros(len(levels), dtype=np.uint8)
    for i,l in enumerate(levels):
        hk = np.sum(img == l)
        hist[i] = hk/hk_sum
    
    dc = hist/np.linalg.norm(hist)

    return dc
    
def descriptor_halarick(img):
    '''
        descriptor using halarick
        paramters:
            - img: reference image
        return
            - dc: descritor vector 
    '''
    co_mat = cooccurrence_matrix(img)

    #calculate energy
    energy = np.sum(np.power(co_mat, 2))

    #calculate entropy
    epsilon = 0.001
    entropy = -np.sum(np.multiply(co_mat,np.log(co_mat+epsilon)))

    #calculate constrast, homogeneity, and correlation
    constrast = 0.0
    homogeneity = 0.0
    correlation = 0.0
    mu_i, mu_j, sigma_i, sigma_j = 0.0, 0.0, 0.0, 0.0
    N,M = co_mat.shape
    
    for i in range(N):
        for j in range(M):
            constrast += ((i-j)**2)*co_mat[i,j]
            
            homogeneity += co_mat[i,j]/(1 + np.abs(i - j))
            
            mu_i += i*co_mat[i,j]
            mu_j += j*co_mat[i,j]
            sigma_i += ((i - mu_i)**2)*co_mat[i,j]
            sigma_j += ((j - mu_j)**2)*co_mat[i,j]
            if sigma_i > 0 and sigma_j > 0:
                correlation += (float(i)*float(j)*co_mat[i,j] - (mu_i*mu_j))/(sigma_i*sigma_j)
            else:
                correlation += 0.0

    constrast = (1/N**2)*constrast 

    dt = [energy, entropy, constrast, correlation, homogeneity]
    dt = np.array(dt, dtype=float)
    dt = dt/np.linalg.norm(dt)

    return dt

def cooccurrence_matrix(img):
    '''
        Calculate co-occurrence matrix
        parameters:
            - img: source image
        return:
            - co_mat: co-ocurrence matrix 
    '''
    N,M = img.shape
    levels = np.max(img)
    co_mat = np.zeros((levels+1,levels+1), dtype=np.int)
    
    for x in range(N-2):
        for y in range(M-2):
            i,j = img[x,y], img[x+1,y+1]
            co_mat[i,j] += 1

    cn = np.divide(co_mat, np.sum(co_mat))

    return cn 

def descriptor_hog(img):
    '''
        It represents the descriptor histogram of Oriented Gradients
        parameters:
            - img: source image
        return:
            - hog: hog features extracted  
    '''
    wsx = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    wsy = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

    #img = img.reshape((int(img.shape[0]//2), int(img.shape[1]//2)))
    gx = scipy.ndimage.convolve(img, wsx)
    gy = scipy.ndimage.convolve(img, wsy)

    np.seterr(divide='ignore', invalid='ignore')

    M = np.sqrt(gx**2 + gy**2)/np.sum(np.sqrt((gx**2)+(gy**2)))
    #M = np.sqrt(np.power(gx,2)+np.power(gy,2))/np.sum(np.sqrt(np.power(gx,2) +np.power(gy,2)))
    #M = np.divide((np.sqrt((np.power(gx,2))+(np.power(gy,2)))),(np.sum(np.power(gx,2))+(np.power(gy,2))))
    theta = np.arctan(gy/gx)
    #theta + (np.pi/2)
    theta = np.add(theta, np.pi/2)
    theta_dg = np.degrees(theta)

    A,B = M.shape
    dg = np.zeros(9, dtype=np.float)
    
    for x in range(A):
        for y in range(B):
            idx = convert_to_bins(theta_dg[x,y])
            dg[idx] += M[x,y]


    dg = dg/np.linalg.norm(dg)

    return  dg

def convert_to_bins(degree):
    '''
        convert angles to bins (bins 0 to 8) range 20
        parameters:
            - angle: angle to convert
        return:
            - bin: angle bin
    '''
    bin_ = 8
    for i in range(0,9):
        if np.isnan(degree):
           bin_ = 8
           break
        if degree in list(range(i, i+20)):
            bin_ = i
            break
    
    return bin_

def find_object(obj_g, img_f):
    '''
        Find object in image F
        parameters:
            - obj_g: object selected
            - img_f: image source 
        return:
            - d: distance 
    '''
    C = img_f.shape[0]*img_f[1]
    W = int(C/32)
    



def select_method(f_img, g_img, b):
    '''
        function that select filter method and calculate the rmse between output 
        image  and original image
        parameters:
            - f_img (numpy array): image with objet f
            - g_img (numpy array): larger image g
            - b: quatisation parameter
        return:
            - ???
    '''

if __name__ == '__main__':

    #load reference image f   
    filename = input().rstrip()
    f_img = imageio.imread(filename)

    #load degraded image g
    filename = input().rstrip()
    g_img = imageio.imread(filename)

    #quantisation parameter
    b = int(input())






