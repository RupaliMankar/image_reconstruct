# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 20:15:27 2021

@author: Rupali
"""

import envi
import numpy 
import pandas 
import glob, os, sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import floor
import timeit
from scipy.signal import windows


# adding the parent directory to basic_utils
sys.path.append(r'D:\software\python\mIRage')
  
from basic_utils.mIRage2envi import bandcsv_to_envi
from basic_utils.img_alignment import align_Images


def reshape_to_ref(Iref, rectE):
    """
    # Reshape all the band images from rectE to amide I band image from high_resolution image
    # Although it is time consuming to align all bands from rect image to amide I band, keeping 
    #same shape as high resolution image ensures same masks as high-resolution data for classification
    """
    dim = (Iref.shape[1], Iref.shape[0])
    rect_inter = np.zeros([rectE.shape[0], dim[1], dim[0]], dtype=np.float32)
    for i in range(rectE.shape[0]):
        rect_inter[i,:,:] = cv2.resize(rectE[i,:,:],dim,interpolation = cv2.INTER_AREA)
        
    return rect_inter

def inter_interleave(Iref, rectE, pixsize = (0.5,5), keep_high_res_shape = False):
    """
    This function converts undersampled/interleaved image (rectangular pixel image) to a square pixel image by zero padding
    
    Iref    :   high resolution single band image
    rectE   :   undersampled/interleaved data, numpy array of rectangulat image in shape (bands,x,y)
    keep_high_res_shape: decide final size ofinterpolated image. If keep_high_res_shape == True then after zero padding rescaled image will be matched to size of Iref
    
    """
    #rescaling rect pixels
    b, y, x = rectE.shape 
    rescaling_factor = pixsize[1]/pixsize[0] #converting 5X0.5 to 0.5X0.5
    y = int(y*rescaling_factor)
    rect_scale = np.zeros((b, y, x), dtype = np.float32)
    
    #fill evry tenth row with the value of rectangular pixels 
    for i in range(rectE.shape[1]):
        rect_scale[:,y,:] = rectE[:,i,:] 
    
    if keep_high_res_shape:
        rect_scale = reshape_to_ref(Iref, rect_scale)
    else:                   
        #for time efficiency and no requirment to have same annotations masks as high resolution mask
        #change shape of reference image 
        Iref = cv2.resize(Iref,(x,y), interpolation = cv2.INTER_AREA)
        rect_Scale_band = cv2.resize(rectE[8,:,:],(x,y),interpolation = cv2.INTER_AREA)
    
    
    
    return Iref, rect_scale, rect_Scale_band

def generate_1Dwindow(wintype, option, img_ydim):
    
    if wintype == 'guassian':
        win = windows.gaussian(img_ydim, option)
    elif wintype == 'hamming':
        win = windows.hamming(img_ydim)
    elif wintype == 'kaiser':
        win = windows.kaiser(img_ydim, beta = option)
    elif wintype == '':
        print('please select a window type')
    
    return win
    
def inter_fft_window(rectE, pixsize = (0.5,5),  wintype = 'guassian', option = 0):
    """"
    This function generates interpolated images using FFT and zero padding it to the desired image size. 
    Zero padding introduces ringing artifacts which are removed by applying window on FFT
    rectE     :   undersampled/interleaved data, numpy array of rectangulat image in shape (bands,x,y)
    pixsize   :   represents interleave properties, i.e. (0.5,5) means pixels are sampled at x = 0.5um and y = 5um spacing
    wintype   :   window type to remove ringing artifact
    option    :   window options, sigma for guassian and beta for kaiser
             
    """
    
    B, y, x = rectE.shape
    
    new_y = int(y*pixsize[1]/pixsize[0])
    zero_pad_ifft = np.zeros(((B, new_y, x)), dtype = complex)
    
    start = int((new_y - y)/2)

    for i in range(B):    
        rect_fourier = np.fft.fftshift(np.fft.fft2(rectE[i,:,:]))
        zero_pad_fft = np.zeros((new_y, x), dtype = complex)
        zero_pad_fft[start:start+y,:] = rect_fourier
        
        if wintype == 'hamming':
            hamming_win = generate_1Dwindow(wintype, option, y)
            win = np.zeros((new_y), dtype = complex)
            win[start:start+y] = hamming_win
        else:
            win = generate_1Dwindow(wintype, option, new_y)
        
        #inverse fourier transform of windowed (only in y-dimension) zero padded fft
        zero_pad_ifft[i,:,:]=  np.fft.ifft2( np.fft.ifftshift((zero_pad_fft.T*win).T))
    
    return np.abs(zero_pad_ifft, dtype= np.float32)
        
    
    
    
# # #----------------------------converting band cvs files to Envi files----------------------------------------------
# path1 = r"Y:\Rect_pix\2022\A4/"
# cores = os.listdir(path1)
# for i in range(len(cores)):
#     path11 = path1+cores[i]
#     csvlist = glob.glob(path11+"/*.csv")
#     wavenumberlist = np.zeros(len(csvlist))
#     fname = path1+cores[i]+'/Envi'+cores[i]
#     for j in range(len(csvlist)):
#         # wavenumberlist[j] = int(csvlist[j][-8:-4]) #if there is no number after band number
#         wavenumberlist[j] = int(csvlist[j][-10:-6]) #if there is one number after band number
      
#     BandsHSI = bandcsv_to_envi(csvlist,wavenumberlist,fname)    
#     print(fname)


#path1 = r"T:\Chalapathi\Project Flash\2022\cervical_27_wavenumbers_csv\D2/"
path1 = r"Y:\Rect_pix\2022\A4/"
pix_spacing = [2, 3, 5, 10, 20]

for p in pix_spacing:
    
    #read high resolution (square pixel envi file)
    sq = envi.envi(r'Y:\Rect_pix\2022\A4/05X05/Envi05X05')
    sqE = sq.loadall()
    sq_W = np.asarray(sq.header.wavelength)
    
    #read high resolution (square pixel envi file)
    rect = envi.envi((r'Y:\Rect_pix\2022\A4/#X05/Envi#X05').replace("#", str(p)))
    pixsize = (0.5,p)
    rectE = rect.loadall()
    rect_x, rect_y, rect_W = rect.header.samples, rect.header.lines, np.asarray(rect.header.wavelength)
    
    #find band images correspodnding to major amide bands and select amide I band image from sq Envi file as a reference image
    amide = [1660, 1540, 1233]
    amide_ind = [np.argmin(np.abs(sq_W-w)) for w in amide]
    Iref = sqE[amide_ind[0],:,:]  
    
    keep_high_res_shape = True  #True if must have size of high resolution image
    
    
    
    
    # --------------------resizing of rectangular pixels using interleaved zero padding-----------------------------
    # Iref_new, rect_inter, rect_Scale_band = inter_interleave(Iref, rectE, pixsize, keep_high_res_shape)
        
    '''-----------intgerpolation using zero padding of fft--------------------'''
    wintype = 'guassian'
    option = 150
    Rect_inter = inter_fft_window(rectE, pixsize, wintype, option)
    
    
    warp_affine = []
    
    if keep_high_res_shape:
        sz = Iref.shape # Find size of image1
        rect_Scale_band = Rect_inter[amide_ind[0],:,:]
        rect_intern = np.zeros((len(rect_W),sz[0],sz[1]))
        im_new_size = cv2.resize(Rect_inter[amide_ind[0],:,:],sz,interpolation = cv2.INTER_AREA)
     
        aligned_img, warp_matrix = align_Images(im_new_size, Iref)
        #align rest of the bands
        for i in range(len(rect_W)):
            #resizing FTIR image 
            im_new_size = cv2.resize(Rect_inter[i,:,:],sz,interpolation = cv2.INTER_AREA)
     
            im_aligned = cv2.warpAffine(im_new_size, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
            rect_intern[i,:,:] = im_aligned
            
        rect_intern_ref = numpy.concatenate((rect_intern, np.reshape(Iref,(1,Iref.shape[0],Iref.shape[1]))), axis=0)
    else:
        
        Iref_new = cv2.resize(Iref, (Rect_inter.shape[2], Rect_inter.shape[1]), interpolation = cv2.INTER_AREA)
        rect_Scale_band = Rect_inter[amide_ind[0],:,:]
        Iref_new, warp_matrix = align_Images(Iref_new, rect_Scale_band)  # reference is interpolated image
        
        rect_intern_ref = numpy.concatenate((Rect_inter, np.reshape(Iref_new,(1,Iref_new.shape[0],Iref_new.shape[1]))), axis=0)

    
    
    
    # #saving enviFile
    # outfname  = path1+'EnviG9_305_inter'
    # envi.save_envi(rect_inter, ''.join(outfname), 'BSQ',rect_W)
    outfname  = path1+'/'+wintype+str(option)+'/'+'Envi_'+str(p)+'05_fft_inter_win'+wintype
    envi.save_envi(rect_intern_ref.astype(np.float32), ''.join(outfname), 'BSQ', np.append(rect_W, 1900))
