import vigra
from vigra import numpy as np
import h5py
import os
from skimage.io import imread
import dask.array as da

#convert tiff groundtruth to hdf5 in order for the Transition Classifier to work

def tiff_to_HDF5(imName, filepath):
    if imName<10:
        imNameM = '0'+ str(imName)
    else: imNameM=str(imName)
    tiff = vigra.impex.readVolume(filepath+imName+'.tif')
	vigra.impex.writeHDF5(tiff, filepath+'my_segmentation.h5', 'exported_data', compression='gzip')

tiff_filepath = '/export/home/lparcala/Fluo-N2DH-SIM/01_GT/TRA/'
imName='man_seg000'
tiff_to_HDF5(imName, tiff_filepath)