# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 13:30:29 2021

@author: Rupali
"""
import pandas as pd
import numpy as np

import skimage as img
import skimage.io
import cv2

import envi


def csv2image(input_file, output_file):
    I = pd.read_csv(input_file,dtype="float32").to_numpy(dtype=np.float32)
    skimage.io.imsave(output_file, I)

def hsicsv_to_envi(infname, *outfname):
    
    df =pd.read_csv(infname , engine='python')
    waves = df.columns.tolist()[2:df.shape[1]]
    waves = [int(i) for i in waves] 
    C = df['Unnamed: 0'].unique() # X 
    R = df['Unnamed: 1'].unique() # Y
    
    hsi = np.zeros((len(waves),len(R),len(C)),dtype='float32')

    for i in range(len(R)):
        for j in range(len(C)):
            #consider image indexing while loading data in HSI file 
            hsi[:,len(R)-1-i,j] = df[(df['Unnamed: 1']==R[i])&(df['Unnamed: 0']==C[j])].to_numpy()[0,2:df.shape[1]]
    
    if not outfname:
        print('array is not saved as ENVI file')
    else:
        envi.save_envi(hsi, ''.join(outfname), 'BSQ', waves)
    return hsi

def bandcsv_to_envi(csvlist, wavenumberlist, *outfname):
    """
    csvlist: list of filenames for CSVs at single wavenumbers
    wavenumberlist: list of wavenumbers at which CSV files were taken
    fname: file to save ENVI

    ex: csvlist = ["900.csv", "1000.csv", "1500.csv"], wavenumberlist = [900, 1000, 1500]

    """
    ziplist = list(zip(csvlist, wavenumberlist))
    ziplist.sort(key = lambda tup: tup[1])
    firstband = pd.read_csv(ziplist[0][0])
    array = np.empty((0,firstband.shape[0],firstband.shape[1]), dtype=np.float32)
    waves = []
    for tup in ziplist:
        arrayslice = pd.read_csv(tup[0]).to_numpy(dtype=np.float32)
        if arrayslice.shape != firstband.shape:
            print("ERROR: CSV files must have same dimensions!")
       # arrayslice, warp_affine = align_Images( arrayslice, firstband.to_numpy(dtype=np.float32))
        arrayslice = np.reshape(arrayslice,(1,arrayslice.shape[0],arrayslice.shape[1]))
        array = np.append(array,arrayslice,axis=0)
        waves.append(tup[1])
    
    if not outfname:
        print('array is not saved as ENvi file')
    else:       
        envi.save_envi(array, ''.join(outfname), 'BSQ', waves)
    return array


def align_Images(im, imref):
 
    im1_gray = np.float32(imref)
    #im2_gray = cv2.cvtColor(reduced_blurred,cv2.COLOR_BGR2GRAY)
    im2_gray = np.float32(im)
    
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
    (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)
     
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography 
        im2_aligned = cv2.warpPerspective (im2_gray, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2_gray, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
        
    cv2.imwrite("Aligned_Image.jpg", im2_aligned);
   
    return im2_aligned, warp_matrix   


