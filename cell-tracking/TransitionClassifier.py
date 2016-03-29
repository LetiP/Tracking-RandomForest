import vigra
import matplotlib.pyplot as plt
from vigra import numpy as np
import h5py
from sklearn.neighbors import KDTree
from sklearn.cross_validation import KFold
from sklearn.metrics import precision_recall_fscore_support
from compiler.ast import flatten
import os
# np.seterr(all='raise')


#read in 'n2-n1' of images
def read_in_images(n1,n2, filepath, fileFormatString='{:05}.h5',path_in_hdf5='volume/data'):
    gt_labelimage_filename = [0]*(n2-n1)
    for i in range(n1,n2):
        # gt_labelimage_filename[i-n1] = os.path.join(str(filepath), fileFormatString.format(i))
    # gt_labelimage = [vigra.impex.readHDF5(gt_labelimage_filename[i], 'segmentation/labels') for i in range(0,n2-n1)]
 ##edited HERE!
        gt_labelimage_filename[i-n1] = str(filepath)+'my_segmentation.h5'
    gt_labelimage = [vigra.impex.readHDF5(gt_labelimage_filename[i], path_in_hdf5)[...,i,:] for i in range(0,n2-n1)]
    return gt_labelimage

#compute features from input data and return them
def compute_features(raw_image, labeled_image, n1, n2):
    #perhaps there is an elegant way to get into the RegionFeatureAccumulator. For now, the new feature are a separate vector
    features = [0]*(n2-n1)
    allFeat = [0]*(n2-n1)
    for i in range(0,n2-n1):
        if len(labeled_image[i].shape) < len(raw_image.shape) - 1:
            # this was probably a missing channel axis, thus adding one at the end
            labeled_image = np.expand_dims(labeled_image, axis=-1)

        # features[i] = vigra.analysis.extractRegionFeatures(raw_image[...,i,0].astype('float32'),labeled_image[i][...,0].astype('uint32'), ignoreLabel=0)
        #edited here, because the coming datasets have the timeaxis as the first axis compared to the Malaria dataset
        try:
            features[i] = vigra.analysis.extractRegionFeatures(raw_image[...,i,0].astype('float32'),labeled_image[i][...,0].astype('uint32'), ignoreLabel=0)
        except:
            try:
                #this version should work for CTC the way it is converted to HDF5 and for the Pgject5646.h5 file
                features[i] = vigra.analysis.extractRegionFeatures(raw_image[...,i].astype('float32'),labeled_image[i][...,0].astype('uint32'), ignoreLabel=0)
            except:
                features[i] = vigra.analysis.extractRegionFeatures(raw_image[i,...,0].astype('float32'),labeled_image[i][...,0].astype('uint32'), ignoreLabel=0)
        if len(raw_image.shape) < 5:
            tempnew1 = vigra.analysis.extractConvexHullFeatures(labeled_image[i][...,0].squeeze().astype(np.uint32), ignoreLabel=0)
            tempnew2 = vigra.analysis.extractSkeletonFeatures(labeled_image[i][...,0].squeeze().astype(np.uint32))
            allFeat[i] = dict(features[i].items()+tempnew1.items()+tempnew2.items())
        else:
            allFeat[i] = dict(features[i].items())
    return allFeat

#return a feature vector of two objects (f1-f2,f1*f2)
def getFeatures(f1,f2,o1,o2):
    res=[]; res2=[]
    for key in f1:
        if key == "Global<Maximum >" or key=="Global<Minimum >":
            # the global min/max intensity is not interesting
            continue
        elif key == 'RegionCenter':
            res.append(np.linalg.norm(f1[key][o1]-f2[key][o2])) #difference of features
            res2.append(np.linalg.norm(f1[key][o1]*f2[key][o2])) #product of features
        elif key=='Histogram': #contains only zeros, so trying to see what the prediction is without it
            continue
        elif key == 'Polygon': #vect has always another length for different objects, so center would be relevant
            continue
        else:
            res.append((f1[key][o1]-f2[key][o2]).tolist() )  #prepare for flattening
            res2.append((f1[key][o1]*f2[key][o2]).tolist() )  #prepare for flattening
    x= np.asarray(flatten(res)) #flatten
    x2= np.asarray(flatten(res2)) #flatten
    #x= x[~np.isnan(x)]
    #x2= x2[~np.isnan(x2)] #not getting the nans out YET
    return np.concatenate((x,x2))

#read in 'n2-n1' of labels
def read_positiveLabels(n1,n2, filepath, fileFormatString='{:04}'):
    gt_labels_filename = [0]*(n2-n1)
    gt_labelimage = [vigra.impex.readHDF5(str(filepath)+'groundtruth.h5', 'tracking/'+fileFormatString.format(i)+'/Moves') for i in range(1,n2-n1)]
    #old version, above is edited for CTC
    # for i in range(n1+1,n2 ): #the first one contains no moves data
    #     gt_labels_filename[i-n1] = os.path.join(str(filepath), fileFormatString.format(i))
    # gt_labelimage = [vigra.impex.readHDF5(gt_labels_filename[i], 'tracking/Moves') for i in range(1,n2-n1)]
    return gt_labelimage

# compute negative labels by nearest neighbor
def negativeLabels(features, positiveLabels):
    numFrames = len(features)
    neg_lab = []
    for i in range(1, numFrames):  # for all frames but the first
        # print("Frame ", i)
        frameNegLab = []
        # build kdtree for frame i
        kdt = KDTree(features[i]['RegionCenter'][1:,...], metric='euclidean')
        # find k=3 nearest neighbors of each object of frame i-1 in frame i
        neighb = kdt.query(features[i-1]['RegionCenter'][1:,...], k=3, return_distance=False)
        for j in range(0, neighb.shape[0]):  # for all objects in frame i-1
            for m in range(0, neighb.shape[1]):  # for all neighbors
                pair = [j + 1, neighb[j][m] + 1]
                if pair not in positiveLabels[i-1].tolist():
                    frameNegLab.append(pair) # add one because we've removed the first element when creating the KD tree
                    # print(pair)
                # else:
                #     print("Discarding negative example {} which is a positive annotation".format(pair))
        neg_lab.append(frameNegLab)
    return neg_lab


def find_features_without_NaNs(features):
    """
    Remove all features from the list of selected features which have NaNs
    """
    selectedFeatures = features[0].keys()
    for featuresPerFrame in features:
        for key, value in featuresPerFrame.iteritems():
            if not isinstance(value, list) and (np.any(np.isnan(value)) or np.any(np.isinf(value))):
                try:
                    selectedFeatures.remove(key)
                except:
                    pass # has already been deleted
    forbidden = ["Global<Maximum >", "Global<Minimum >", 'Histogram', 'Polygon']
    for f in forbidden:
        if f in selectedFeatures:
            selectedFeatures.remove(f)
    for featuresPerFrame in features:
        for key in featuresPerFrame.keys():
            if key not in selectedFeatures:
                del featuresPerFrame[key]
    return features

class TransitionClassifier:
    def __init__(self, selectedFeatures):
        self.rf = vigra.learning.RandomForest()
        self.mydata = None
        self.labels = []
        self.selectedFeatures = selectedFeatures
        
    def addSample(self, f1, f2, label,currFrame):
        #if self.labels == []:
        self.labels.append(label)
        #else:
        #    self.labels = np.concatenate((np.array(self.labels),label)) # for adding batches of features
        res=[]
        res2=[]
        mykeys = [x for x in self.selectedFeatures[currFrame]]
        for key in mykeys:
            if key == "Global<Maximum >" or key=="Global<Minimum >":
                # the global min/max intensity is not interesting
                continue
            elif key == 'RegionCenter':
                res.append(np.linalg.norm(f1[key]-f2[key])) #difference of features
                res2.append(np.linalg.norm(f1[key]*f2[key])) #product of features
            elif key == 'Histogram': #contains only zeros, so trying to see what the prediction is without it
                continue
            elif key == 'Polygon': #vect has always another length for different objects, so center would be relevant
                continue
            else:
                if not isinstance(f1[key], np.ndarray):
                    calc1=float(f1[key]) - float(f2[key])
                    calc2=float(f1[key]) * float(f2[key])
                    # res.append(float(f1[key]) - float(f2[key]) )  #prepare for flattening
                    # res2.append(float(f1[key]) * float(f2[key]) )  #prepare for flattening
                else:
                    calc1=(f1[key]-f2[key]).tolist() 
                    calc2=(f1[key]*f2[key]).tolist() 
                    # res.append((f1[key]-f2[key]).tolist() )  #prepare for flattening
                    # res2.append((f1[key]*f2[key]).tolist() )  #prepare for flattening
                # if np.any(np.isnan(calc1)) or np.any(np.isnan(calc2)) or np.any(np.isinf(calc1)) or np.any(np.isinf(calc2)):
                #     for frame in range(0, len(selectedFeatures)):
                #         if key in self.selectedFeatures[frame].keys():
                #             print key
                #             del self.selectedFeatures[frame][key]
                # else:
                    res.append(calc1)
                    res2.append(calc2)
        x= np.asarray(flatten(res)) #flatten
        x2= np.asarray(flatten(res2)) #flatten
        # assert(np.any(np.isnan(x)) == False)
        # assert(np.any(np.isnan(x2)) == False)
        # assert(np.any(np.isinf(x)) == False)
        # assert(np.any(np.isinf(x2)) == False) 
        #x= x[~np.isnan(x)]
        #x2= x2[~np.isnan(x2)] #not getting the nans out YET
        features = np.concatenate((x,x2))
        if self.mydata is None:
            self.mydata = features
        else:
            self.mydata = np.vstack((self.mydata, features))

    #adding a comfortable function, where one can easily introduce the data
    def add_allData(self, mydata, labels):
        self.mydata = mydata
        self.labels = labels
    def train(self):
        print("Training classifier from {} positive and {} negative labels".format(np.count_nonzero(np.asarray(self.labels)),
                                                                                   len(self.labels)- np.count_nonzero(np.asarray(self.labels))))
        oob = self.rf.learnRF(self.mydata.astype("float32"), (np.asarray(self.labels)).astype("uint32").reshape(-1,1))
        print("RF trained with OOB Error ", oob)
    
    def predictSample(self, test_data):
        return self.rf.predictLabels(test_data.astype('float32'))
    
    def predictProbabilities(self, test_data):
        return self.rf.predictProbabilities(test_data.astype('float32'))
        
    def predictLabels(self, test_data, threshold=0.5):
        prob = self.rf.predictProbabilities(test_data.astype('float32'))
        res = np.copy(prob)
        for i in range(0,len(prob)):
            if prob[i][1]>= threshold:
                res[i]=1.
            else:
                res[i]=0
        return np.delete(res, 0, 1)
                                            
    def writeRF(self, outputFilename):
        self.rf.writeHDF5(outputFilename, pathInFile='/ClassifierForests/Forest0000')

        # write selected features
        with h5py.File(outputFilename, 'r+') as f:
            featureNamesH5 = f.create_group('SelectedFeatures')
            featureNamesH5 = featureNamesH5.create_group('Standard Object Features')
            for feature in self.selectedFeatures[0]:
                featureNamesH5.create_group(feature)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="trainRF",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("filepath",
                        help="read ground truth from this folder", metavar="FILE")
    #CTC need another filepath for the tracking moves (positive labels)
    parser.add_argument("filepath_of_tracks_poslabels",
                        help="read ground truth of moves from this folder", metavar="FILE")
    parser.add_argument("rawimage_filename",
                        help="filepath+name of the raw image", metavar="FILE")
    parser.add_argument("--rawimage-h5-path", dest='rawimage_h5_path', type=str,
                        help="Path inside the rawimage HDF5 file", default='volume/data')
    parser.add_argument("initFrame", default=0, type=int, 
                        help="where to begin reading the frames")
    parser.add_argument("endFrame", default=0, type=int, 
                        help="where to end frames")
    parser.add_argument("outputFilename",
                        help="save RF into file", metavar="FILE")
    parser.add_argument("--filename-zero-padding", dest='filename_zero_padding', default=5, type=int,
                        help="Number of digits each file name should be long")
    parser.add_argument("--path-in-hdf5", dest='path_in_hdf5', default='volume/data', type=str,
                        help="Path inside the hdf5-File to look in when reading raw data")
    parser.add_argument("--time-axis-index", dest='time_axis_index', default=2, type=int,
                        help="Zero-based index of the time axis in your raw data. E.g. if it has shape (x,t,y,c) this value is 1. Set to -1 to disable any changes")

    args = parser.parse_args()

    filepath = args.filepath
    filepath_movesGT = args.filepath_of_tracks_poslabels
    rawimage_filename = args.rawimage_filename
    initFrame = args.initFrame
    endFrame = args.endFrame
    fileFormatString = '{'+':0{}'.format(args.filename_zero_padding)+'}.h5'
    path_in_hdf5 = args.path_in_hdf5
    print 'Transition Classifier is running...'

    rawimage = vigra.impex.readHDF5(rawimage_filename, args.rawimage_h5_path)
    try:
        print(rawimage.axistags)
    except:
        pass
    # transform such that the order is the following: X,Y,(Z),T, C
    rawimage = rawimage[:,:,0,:,:]

    if args.time_axis_index != -1:
        rawimage = np.rollaxis(rawimage, args.time_axis_index, -1)
        rawimage = np.swapaxes(rawimage,0,1)

    features = compute_features(rawimage,read_in_images(initFrame,endFrame, filepath, fileFormatString, path_in_hdf5),initFrame,endFrame)
    selectedFeatures = find_features_without_NaNs(features)
    mylabels = read_positiveLabels(initFrame,endFrame,filepath_movesGT)
    neg_labels = negativeLabels(features,mylabels)
    TC = TransitionClassifier(selectedFeatures)

    # compute featuresA for each object A from the feature matrix from Vigra
    def compute_ObjFeatures(features, obj):
        dict={}
        for key in features:
            if key == "Global<Maximum >" or key=="Global<Minimum >": #this ones have only one element
                 dict[key] = features[key]
            else:
                 dict[key] = features[key][obj]
        return dict
    for k in range(0,len(selectedFeatures)-1):
        for i in mylabels[k]:
            TC.addSample(compute_ObjFeatures(TC.selectedFeatures[k], i[0]), compute_ObjFeatures(TC.selectedFeatures[k+1], i[1]), 1,k)   #positive
        for i in neg_labels[k]:
            TC.addSample(compute_ObjFeatures(TC.selectedFeatures[k], i[0]), compute_ObjFeatures(TC.selectedFeatures[k+1], i[1]), 0,k)    #negative

    TC.mydata = TC.mydata[:,~np.isnan(TC.mydata).any(axis=0)] #erasing the NaNs
    TC.mydata = TC.mydata[:,~np.isinf(TC.mydata).any(axis=0)] #erasing the infinites
    print TC.mydata.shape
        
    print 
    TC.train()

    # delete file before writing
    if os.path.exists(args.outputFilename):
        os.remove(args.outputFilename)
    TC.writeRF(args.outputFilename) #writes learned RF to disk

    print "The Transition Classifier did his job."
    print 
