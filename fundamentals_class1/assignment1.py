#!/usr/bin/env python3

"""

Image Processing -- SCC0251/5830— Prof. Moacir A. Ponti

Assignment 1 : image generation

assignment1: fundamentals of image processing -- generation, sampling, and quantization

                2021, April -- 1º Semester

"""

__author__      = "Erikson Aguiar -- NUSP: 11023222"


import numpy as np
import random
'''
    normalize values of pixels 

    parameters: z is an image (numpy array), c is min value (int) and d is max value
    return: image normalized from min to max values of pixels
'''
def normalize(z, c, d):
    a = z.min()
    b = z.max()

    tz = (z-a)*((d-c)/(b-a)) + c

    return tz

'''
    generate synthetic image using function one 
    
    parameters: C is the image size (int)
    return: synthetic image
'''
def gen_func_one(c):
    new_image = np.zeros((c,c), dtype=float)
    for x in range(c):
        for y in range(c):
            new_image[x,y] = ((x*y) + (2*y))
    
    n_img = normalize(new_image, 0, ((2**16)-1))
    
    return n_img


'''
    generate synthetic image using function two
    
    parameters: C is the image size (int), Q is parameter for image generation (int)
    return: synthetic image
'''
def gen_func_two(c,q):
    new_image = np.zeros((c,c), dtype=float)
    for x in range(c):
        for y in range(c):
            new_image[x,y] = np.abs(np.cos(x/q) + (2*np.sin(y/q)))
    
    n_img = normalize(new_image, 0, ((2**16)-1))
    
    return n_img

'''
    generate synthetic image using function three
    
    parameters: C is the image size (int), Q is parameter for image generation (int)
    return: synthetic image
'''
def gen_func_three(c,q):
    new_image = np.zeros((c,c), dtype=float)
    for x in range(c):
        for y in range(c):
            new_image[x,y] = float(np.abs((3*(x/q)) - (np.cbrt(y/q))))
    
    n_img = normalize(new_image, 0, ((2**16)-1))

    return n_img

'''
    generate synthetic image using function four
    
    parameters: C is the image size (int), S is a seed value for random number generation (int)
    return: synthetic image
'''
def gen_func_four(c,s):
    random.seed(s)
    new_image = np.zeros((c,c), dtype=float)
    for x in range(c):
        for y in range(c):
            new_image[x,y] = random.random()
    
    n_img = normalize(new_image, 0, ((2**16)-1))

    return n_img

'''
    function to walking on image coordinates
    
    parameters: C is the image size (int), x coordinate with axis x (int), y is coordinate with axis y (int)
    return: coordinate x (int) and coordinate y (int)
'''
def random_walk(c,x, y):
    dx = random.randint(-1,1)
    dy = random.randint(-1,1)
    new_x = np.mod((x+dx),c)
    new_y = np.mod((y+dy),c)

    return new_x,new_y

'''
    generate synthetic image using function four that represent a random walk
    
    parameters: C is the image size (int), S is a seed value for random number generation (int)
    return: synthetic image
'''
def gen_func_five(c,s):
    random.seed(s)
    new_image = np.zeros((c,c), dtype=float)
    x, y = 0,0
    new_image[x,y] = 1
    for _ in range(1, (1+c**2)):
        x, y = random_walk(c,x,y)
        new_image[x,y] = 1
    
    
    return new_image

'''
    downsampling image CxC following regions r, which select first value of the region
    parameters: image (numpy array), c is image size c*c (int), N is size sampled image (int)
    return: image reduced 
'''
def downsampling(image, c,n):
    r = int((c/n)**2)
    tam = int(c/(r/2))
    new_image = np.zeros((tam,tam), dtype=float)
    px, py = 0,0
    for x in range(0, c, int(c/n)):
        for y in range(0, c, int(c/n)):
            px, py = int(x/(c/n)), int(y/(c/n))
            new_image[px,py] = image[x,y]
                    
    return new_image

'''
    Quantize images with B bits
    parameters: image (numpy array), B is number of bitwise(B)
    return image quantized
'''
def quantization(image, B):
    #normalize image
    new_image = normalize(image, 0, 255).astype(np.uint8)
    #new_image = new_image & 0b11100000
    #calculate bitwise
    zeros = 8 - B
    b = '0b'
    b += '1' * B
    b += '0' * zeros
    #apply AND between pixel value and mask of bits
    new_image = new_image & int(b, base=2)

    return new_image

'''
    calculate difference between two images using root square error
    parameters: reference image (numpy array), generation image (numpy array)
    return: root square error (float)
'''
def difference_img(ref_img, gen_img):
    rse = np.sqrt(np.sum((gen_img - ref_img)**2))

    return rse

'''
    function to select method for generate images
    paramters: func is a method (int), C image size C*C (int), Q generation (int)
               S seed for random number generation (int)
    return: return a synthetic image from a method selected
'''
def generate_image(func, C, Q, S):
    synthetic_img = None
    #swich function
    if func == 1:
        synthetic_img = gen_func_one(C)
    elif func == 2:
        synthetic_img = gen_func_two(C,Q)
    elif func == 3:
        synthetic_img = gen_func_three(C,Q)
    elif func == 4:
        synthetic_img = gen_func_four(C,S)
    elif func == 5:
        synthetic_img = gen_func_five(C,S)

    return synthetic_img


if __name__ == '__main__':
    #input image name
    filename = input().rstrip()
    R = np.load(filename)
    #image size C
    C = int(input())
    #function choose (1,2,3,4,5)
    func = int(input())
    # generation parameter
    Q = int(input())
    # lateral size N
    N = int(input())
    #number of bits per pixel B
    B = int(input())
    #random function seed S
    S = int(input())

    #generate image
    img = generate_image(func, C, Q, S)
    #downscalling image
    down_img = downsampling(img, C,N)
    #apply quantization with B bits
    quant_img = quantization(down_img, B)
    #calculate difference between two images
    diff = difference_img(R, quant_img)
    print(format(diff, '.4f'))
    


