  # -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 16:36:08 2020

@author: Rupali
"""
import envi
import numpy 
import pandas 
import glob, os
#from myUtils import align_Images
import numpy as np
import matplotlib.pyplot as plt


def hsicsv_to_envi(infname, *outfname):
    
    df =pandas.read_csv(infname , engine='python')
    waves = df.columns.tolist()[2:df.shape[1]]
    waves = [int(i) for i in waves] 
    C = df['Unnamed: 0'].unique() # X 
    R = df['Unnamed: 1'].unique() # Y
    
    hsi = numpy.zeros((len(waves),len(R),len(C)),dtype='float32')

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
    firstband = pandas.read_csv(ziplist[0][0])
    array = numpy.empty((0,firstband.shape[0],firstband.shape[1]), dtype=numpy.float32)
    waves = []
    for tup in ziplist:
        arrayslice = pandas.read_csv(tup[0]).to_numpy(dtype=numpy.float32)
        if arrayslice.shape != firstband.shape:
            print("ERROR: CSV files must have same dimensions!")
       # arrayslice, warp_affine = align_Images( arrayslice, firstband.to_numpy(dtype=numpy.float32))
        arrayslice = numpy.reshape(arrayslice,(1,arrayslice.shape[0],arrayslice.shape[1]))
        array = numpy.append(array,arrayslice,axis=0)
        waves.append(tup[1])
    
    if not outfname:
        print('array is not saved as ENvi file')
    else:       
        envi.save_envi(array, ''.join(outfname), 'BSQ', waves)
    return array




if __name__ == '__main__':
    #read hyperspectral data
    infile =r"T:\Rupali\2021\small area\#.csv"
    
    Pol_angles = [" 1"," 2"," 3"]
    # for ang in Pol_angles:
    #     fname = infile.replace('#', ang)
    #     outfname = fname.replace('.csv','')
    #     array = hsicsv_to_envi(fname,outfname)
    
    #read band images
    path1 = r"T:\Rupali\reticulin\2021\25 February\island band images"
    path1 = r"T:\Rupali\reticulin\2021\6 May\#"
    path1 = r"Y:\Kidney_polarization_data\4325-ROI\glo1/#"
    path1 = r"Y:\Kidney_polarization_data\hkb_4325_glo1 - Copy/#"
    pol_bands = []
    Pol_angles =["pol45","pol90","pol135"]
    csvlist = ['OPTIR 1540.csv','OPTIR 1660.csv']
    wavenumberlist = list(map(lambda csv: int(csv[-8:-4]),csvlist))
    # csvlist = ['OPTIR 1240 1.csv','OPTIR 1660 1.csv']
    # wavenumberlist = list(map(lambda csv: int(csv[-10:-6]),csvlist))
    
    #read one band from one of the polarization to kniow dimensions of the file 
    firstband = pandas.read_csv(path1.replace('#',Pol_angles[0]+"/"+csvlist[0]))
    a = numpy.zeros((3, firstband.shape[0], firstband.shape[1]))
    
    i=0
    
    for ang in Pol_angles:
        os.chdir(path1.replace('#',ang))    
        outfname = 'b'+ang
        bands = bandcsv_to_envi(csvlist,wavenumberlist,outfname)
        pol_bands.append(bands)
        a[i,:,:] = bands[0,0:firstband.shape[0],0:firstband.shape[1]]/bands[1,0:firstband.shape[0],0:firstband.shape[1]]
        i= i+1
    
    #create a mask
    mask = np.zeros_like(pol_bands[0][1,:,:])
    mask[pol_bands[0][1,:,:]>0.12]=1    
    
    amide  = "1540"    
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(pol_bands[0][0,:,:],cmap='inferno',vmax = 0.4)
    plt.colorbar()
    plt.title(amide +" @ 45 degrees")
    
    plt.subplot(2, 3, 2)
    plt.imshow(pol_bands[1][0,:,:],cmap='inferno',vmax = 0.4)
    plt.colorbar()
    plt.title(amide +" @ 90 degrees")
    
    plt.subplot(2, 3, 3)
    plt.imshow(pol_bands[2][0,:,:],cmap='inferno',vmax = 0.4)
    plt.colorbar()
    plt.title(amide +" @ 135 degrees")
    
    plt.subplot(2, 3, 4)
    plt.imshow(pol_bands[0][1,:,:],cmap='inferno',vmax = 0.4)
    plt.colorbar()
    plt.title("1660 @ 45 degrees")
    
    plt.subplot(2, 3, 5)
    plt.imshow(pol_bands[1][1,:,:],cmap='inferno',vmax = 0.4)
    plt.colorbar()
    plt.title("1660 @ 90 degrees")
    
    plt.subplot(2, 3, 6)
    plt.imshow(pol_bands[2][1,:,:],cmap='inferno',vmax = 0.4)
    plt.colorbar()
    plt.title("1660 @ 135 degrees") 
    
    
    cmin = 0
    cmax = 0.8
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(a[0,:,:]*mask,cmap='inferno',vmax =cmax)
    plt.colorbar()
    plt.title("Ratio at 45 degrees")
    
    plt.subplot(1, 3, 2)
    plt.imshow(a[1,:,:]*mask,cmap='inferno',vmax = cmax)
    plt.colorbar()
    plt.title("Ratio at 90 degrees")
    
    plt.subplot(1, 3, 3)
    plt.imshow(a[2,:,:]*mask,cmap='inferno',vmax = cmax)
    plt.colorbar()
    plt.title("Ratio at 135 degrees")
    
    d = np.zeros_like(a)
    d[0,:,:] = np.abs(a[0,:,:]-a[1,:,:])
    d[1,:,:] = np.abs(a[0,:,:]-a[2,:,:])
    d[2,:,:] = np.abs(a[2,:,:]-a[1,:,:])
    
    Psensitivity = np.max(d, axis=0)
    plt.figure()
    plt.imshow(Psensitivity*mask,cmap='inferno',vmin = 0, vmax = 0.5)
    plt.colorbar()
    plt.title("Polarization sensitivity")

    
    # a = os.listdir(r'T:\Rupali\reticulin\2021\8 January\T_shape_tilted_visible_focus')
    # os.chdir(r'T:\Rupali\reticulin\2021\8 January\T_shape_tilted_visible_focus')
    
    # w = numpy.zeros(len(a))
    # for i in range(len(a)):
    #     w[i] = int(a[i][6:10])
        
    # ind = numpy.argsort(w)
    # csvlist = []
    # wavenumberlist = []
    # for i in range(len(ind)):
    #     csvlist.append(a[ind[i]])
    #     wavenumberlist.append(w[ind[i]])
# fname = "T_shape_tilted_visible_focus"
    
    # for i in range(2):
    #     i = i+2
    #     path1=r"D:\reticulin processing\2021\24 June/"+str(i)+"/OPTIR"
    #     csvlist = [path1+"1540.csv", path1+"1660.csv"]
    #     fname= path1+"pol"+str(i)
    #     wavenumberlist = [ 1540, 1660]
    #     BandsHSI = bandcsv_to_envi(csvlist,wavenumberlist,fname)
        #aligneed_ig, warp_affine = align_Images(BandsHSI[0,:,:], BandsHSI[1,:,:])
    
    # for i in range(3):
    #     i = i+1
    #     path1=r"D:\reticulin processing\2021\9 July\slope/"+str(i)+"/OPTIR "
    #     csvlist = [path1+"1540.csv", path1+"1665.csv"]
    #     fname= path1+"pol"+str(i)
    #     wavenumberlist = [ 1540, 1665]
    #     BandsHSI = bandcsv_to_envi(csvlist,wavenumberlist,fname)
        #aligneed_ig, warp_affine = align_Images(BandsHSI[0,:,:], BandsHSI[1,:,:])
    # E = envi.envi(fname)
    # hsi = E.loadall()
    # E.close
    # bkg = numpy.array([0.682,0.394,3.323,2.962,1.358],dtype=numpy.float32)# for C1 to D3
    # bkg = numpy.array([1.05, 0.554, 0.362, 2.896, 2.551, 1.176],dtype=numpy.float32) #for D4 to J10
    # hsi_bkg = numpy.divide(hsi.T,bkg).T
    # E.save_envi(hsi_bkg, fname+"-bkg", wavenumberlist)
    
    
    # path1 = r"Y:/Rect_pix/cervix/sq_pix/"
    # cores = os.listdir(path1)
    # for i in range(len(cores)):
    #     path11 = path1+cores[i]
    #     csvlist = glob.glob(path11+"/*.csv")
    #     wavenumberlist = np.zeros(len(csvlist))
    #     fname = path1+cores[i]+'/Envi'+cores[i]
    #     for j in range(len(csvlist)):
    #        # wavenumberlist[j] = int(csvlist[j][-8:-4]) #if there is no number after band number
    #         wavenumberlist[j] = int(csvlist[j][-10:-6]) #if there is one number after band number
          
    #     BandsHSI = bandcsv_to_envi(csvlist,wavenumberlist,fname)
        

    # filepath = r'Y:\Rect_pix\cervix/A1_1660norm'
    # infile = envi.envi(filepath)
    # hsi_full = infile.loadall()


    # filepath = r'Y:\Rect_pix\cervix/EnviA1_505'
    # infile = envi.envi(filepath)
    # hsi_rect = infile.loadall()
    
    # aligneed_ig, warp_affine = align_Images(hsi_rect[9,:,:], hsi_full[6,:,:])
 

