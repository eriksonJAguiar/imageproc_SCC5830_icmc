#!/usr/bin/env python3

"""

    Image Processing -- SCC5830 — Prof. Moacir A. Ponti

    Assignment 5 : image descriptors

    @author: "Erikson Aguiar -- NUSP: 11023222"

                    2021, June -- 1º Semester

"""
import numpy as np
import imageio
import scipy.ndimage
from matplotlib import pyplot as plt

def pre_processing(img, b):
    '''
        method for preprocessing image and quantise to b bits
        paramters:
            - img: reference image
            - b: bit to quantise
        return:
            - p_img: precessed image
    '''
    lm_img = np.floor(((0.299*img[:,:,2])+(0.587*img[:,:,1])+(0.114*img[:,:,0]))).astype(np.uint8)
    q_img = np.right_shift(lm_img, (8 - b))

    return q_img.astype(np.float64)

def normalized_histogram(img, no_levels):
    '''
        calculate normalized histogram of image
        paramters:
            - img: reference image
        return
            - dc: descritor vector
    '''
    # for i in range(no_levels):
    #     hk = np.sum(img == i)
    #     hist[i] = hk
    # n_levels = img.max()
    hist = np.histogram(img, bins=no_levels)[0]
    hist_norm = hist/np.sum(hist)

    dc = hist_norm/np.linalg.norm(hist_norm)

    return dc
    
def descriptor_halarick(img, no_levels):
    '''
        descriptor using halarick
        paramters:
            - img: reference image
        return
            - dc: descritor vector 
    '''
    co_mat = co_occurrence_matrix(img, no_levels)

    #calculate energy
    energy = np.sum(np.power(co_mat,2))

    #calculate entropy
    epsilon = 0.001
    entropy = -np.sum(co_mat*np.log(co_mat+epsilon))

    #calculate constrast, homogeneity, and correlation
    N,M = co_mat.shape 
    x = np.arange(0, no_levels)
    y = np.arange(0, no_levels)
    i, j = np.meshgrid(x, y, sparse=True, indexing='xy')
    constrast = (1/np.power((N*M),2))*np.sum((np.power((i - j), 2))*co_mat)
    mu_i = np.sum(i)*np.sum(co_mat)
    mu_j = np.sum(j)*np.sum(co_mat)
    sigma_i = np.sum((i - mu_i)**2)*np.sum(co_mat)
    
    sigma_j = np.sum((j - mu_j)**2)*np.sum(co_mat)
    if sigma_i > 0 and sigma_j > 0:
        correlation = (np.sum(i*j*co_mat) - (mu_i*mu_j))/(sigma_i*sigma_j)
    else:
        correlation = 0.0 
    homogeneity = np.sum(co_mat/(1 + np.abs(i - j)))

    dt = np.array([energy, entropy, constrast, correlation, homogeneity])
    dt = dt/np.linalg.norm(dt)

    return dt

def co_occurrence_matrix(img, no_levels):
    '''
        Calculate co-occurrence matrix
        parameters:
            - img: source image
        return:
            - co_mat: co-ocurrence matrix 
    '''
    N,M = img.shape
    co_mat = np.zeros((no_levels,no_levels))
    
    for x in range(N-1):
        for y in range(M-1):
            i,j = int(img[x,y]), int(img[x+1,y+1])
            co_mat[i,j] += 1

    cn = co_mat/np.sum(co_mat)

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


    gx = scipy.ndimage.convolve(img, wsx)
    gy = scipy.ndimage.convolve(img, wsy)

    np.seterr(divide='ignore', invalid='ignore')

    m = np.sqrt(np.power(gx, 2) + (np.power(gy, 2)))
    M = np.nan_to_num(m/np.sum(m))
    theta = np.nan_to_num(np.arctan(gy/gx))
    theta += (np.pi/2)
    theta_dg = np.degrees(theta)

    theta_d = ((theta_dg-1)/(20)).astype(np.int)
    #theta_d[theta_d >= 9] = 8

    A,B = M.shape
    dg = np.zeros(9, dtype=np.float64)
    
    for x in range(A):
        for y in range(B):
            bins = theta_d[x,y]
            dg[bins] += M[x,y]

    dg = np.nan_to_num(dg/np.linalg.norm(dg))

    return  dg



def apply_descriptors(img, no_levels):
    '''
        aux method to apply descriptors
        paramters:
            - img: image selected
        return:
            - des: concat descriptors
    '''
    dc = normalized_histogram(img, no_levels)
    dt = descriptor_halarick(img, no_levels)
    dg = descriptor_hog(img)
    des = np.concatenate((dc,dt,dg))

    return des

def find_object(img_f, obj_g, b):
    '''
        Find object in image F
        parameters:
            - obj_g: object selected
            - img_f: image source 
        return:
            - D: distance 
    '''
    N, M, _ = img_f.shape
    C = N
    W = int(np.floor(C/16))
    img_f_tranf = pre_processing(img_f, b)
    no_levels = np.unique(img_f_tranf.flatten()).shape[0]
    obj_g_tranf = pre_processing(obj_g, b)

    d = np.array(apply_descriptors(obj_g_tranf, no_levels))

    dist, w, r  = [],[],[]
    i = 0
    
    for win, x in enumerate(range(0, N, W)):
        for row, y in enumerate(range(0,M, W)):
            window = img_f_tranf[x:x+32, y:y+32]
            di = np.array(apply_descriptors(window, no_levels))
            dist.append(np.sqrt(np.sum(np.power((d - di), 2))))
            w.append(win)
            r.append(row)
    
    min_dist = np.argmin(dist)
    
    return w[min_dist], r[min_dist]

if __name__ == '__main__':

    #load degraded image g
    filename = input().rstrip()
    g_img = imageio.imread(filename)

    #load reference image f   
    filename = input().rstrip()
    f_img = imageio.imread(filename)

    #quantisation parameter
    b = int(input())

    w,r = find_object(f_img, g_img, b)
    print("{} {}".format(w,r))






