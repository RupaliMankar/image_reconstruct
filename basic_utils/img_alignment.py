# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 15:49:35 2019

@author: Rupali
"""
#  from __future__ import print_function
import cv2
import numpy as np
import envi
import matplotlib.pyplot as plt

def align_Images(im2, imref):
    #im2 = im2.astype('float32')
    #blur with 5X% box kernel
    #$blur_im = cv2.blur(im2, (5,5));
    #downsampling to match size of FTIR standard mag
    #downsampled = blur_im[3:blur_im.shape[0]:5, 3:blur_im.shape[1]:5]
    if len(np.shape(im2))>2:
        # Convert images to grayscale
        im2 = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
    #im2_gray = cv2.cvtColor(reduced_blurred,cv2.COLOR_BGR2GRAY)
    if len(np.shape(imref))>2:
        imref = cv2.cvtColor(imref,cv2.COLOR_BGR2GRAY)
    
    # Find size of image1
    sz = imref.shape
     
    # Define the motion model
    warp_mode = cv2.MOTION_AFFINE
     
    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)
     
    # Specify the number of iterations.
    number_of_iterations = 5000;
     
    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10;
     
#    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
     
    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (imref,im2,warp_matrix, warp_mode, criteria)
     
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography 
        im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
        
    cv2.imwrite("Aligned_Image.jpg", im2);
   
    return im2_aligned, warp_matrix

 
#if __name__ == '__main__':
#  path1 = r"Y:/Rupali/pansharpening/core_g2/"
#  refFilename = path1+'G2_dark.png'
#  imFilename = path1+'g2_1650.bmp' 
##  # Read reference image
#  imRef = cv2.imread(refFilename, cv2.IMREAD_COLOR) #FTIR-HD
#  sz = imRef.shape
##  # Read image to be aligned 
#  im = cv2.imread(imFilename, cv2.IMREAD_COLOR) #mIRage
#
#  #resizing FTIR image to dark field 10x  image 
#  dim = (imRef.shape[0], imRef.shape[1])
#  im_new_size = cv2.resize(im,dim,interpolation = cv2.INTER_AREA)
#  
#  
#  #imReference = EnviImg
#  print("Aligning images ...")
#  # Registered image will be resotred in imReg. 
#  # The estimated homography will be stored in h. 
#  img_aligned, warp_matrix = align_Images( im_new_size, imRef); #function written to align high resolution images to low resolution images 
