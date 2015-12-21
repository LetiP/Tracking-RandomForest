import vigra
import matplotlib.pyplot as plt
from vigra import numpy as np
import h5py
from sklearn.neighbors import KDTree
from sklearn.cross_validation import KFold
from sklearn.metrics import precision_recall_fscore_support
from compiler.ast import flatten
import os

#read in 'n2-n1' of images
def read_in_images(n1,n2, filepath, fileFormatString='{:05}.h5'):
    gt_labelimage_filename = [0]*(n2-n1)
    for i in range(n1,n2):
        gt_labelimage_filename[i-n1] = os.path.join(str(filepath), fileFormatString.format(i))
    gt_labelimage = [vigra.impex.readHDF5(gt_labelimage_filename[i], 'segmentation/labels') for i in range(0,n2-n1)]
    return gt_labelimage

#compute features from input data and return them
def compute_features(raw_image, labeled_image, n1, n2):
    #perhaps there is an elegant way to get into the RegionFeatureAccumulator. For now, the new feature are a separate vector
    features = [0]*(n2-n1)
    allFeat = [0]*(n2-n1)
    for i in range(0,n2-n1):
        features[i] = vigra.analysis.extractRegionFeatures(raw_image[:,:,i,0].astype('float32'),labeled_image[i][:,:,0], ignoreLabel=0)
        tempnew1 = vigra.analysis.extractConvexHullFeatures(labeled_image[i][:,:,0].squeeze().astype(np.uint32), ignoreLabel=0)
        tempnew2 = vigra.analysis.extractSkeletonFeatures(labeled_image[i][:,:,0].squeeze().astype(np.uint32))
        allFeat[i] = dict(features[i].items()+tempnew1.items()+tempnew2.items())
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
def read_positiveLabels(n1,n2, filepath, fileFormatString='{:05}.h5'):
    gt_labels_filename = [0]*(n2-n1)
    for i in range(n1+1,n2 ): #the first one contains no moves data
        gt_labels_filename[i-n1] = os.path.join(str(filepath), fileFormatString.format(i))
    gt_labelimage = [vigra.impex.readHDF5(gt_labels_filename[i], 'tracking/Moves') for i in range(1,n2-n1)]
    return gt_labelimage

# compute negative labels by nearest neighbor
def negativeLabels(features, positiveLabels):
    neg_lab = [[]]*len(features)
    for i in range(1, len(features)):
        kdt = KDTree(features[i]['RegionCenter'], metric='euclidean')
        neighb = kdt.query(features[i-1]['RegionCenter'], k=3, return_distance=False)
        for j in range(1, len(features[i])):
            for m in range(0, neighb.shape[1]):
                neg_lab[i].append([j,neighb[j][m]])
    return neg_lab

def allFeatures(features, labels, neg_labels):
    j=0
    lab=[]
    for i in range(0,len(features)-1):
        for k in labels[i]:
            if j == 0:
                x = getFeatures(features[i],features[i+1],k[0],k[1])
                j+=1
            else:
                x = np.vstack((x,getFeatures(features[i],features[i+1],k[0],k[1])))
            lab.append(1)
        for k in neg_labels[i]:
            if k not in labels[i].tolist():
                x = np.vstack((x,getFeatures(features[i],features[i+1],k[0],k[1])))
                lab.append(0)
    x = x[:,~np.isnan(x).any(axis=0)] #now removing the nans
    return x,np.asarray(lab)

def allFeatures_for_prediction(features, labels):
    j=0
    for i in range(0,len(features)-1):
        for k in labels[i]:
            if j == 0:
                x = getFeatures(features[i],features[i+1],k[0],k[1])
                j+=1
            else:
                x = np.vstack((x,getFeatures(features[i],features[i+1],k[0],k[1])))
    x = x[:,~np.isnan(x).any(axis=0)] #now removing the nans
    return x

class TransitionClassifier:
    def __init__(self):
        self.rf = vigra.learning.RandomForest()
        self.mydata = None
        self.labels = []
        
    def addSample(self, f1, f2, label):
        #if self.labels == []:
        self.labels.append(label)
        #else:
        #    self.labels = np.concatenate((np.array(self.labels),label)) # for adding batches of features
        res=[]; res2=[]
        res=[]; res2=[]

        for key in f1:
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
                res.append((f1[key]-f2[key]).tolist() )  #prepare for flattening
                res2.append((f1[key]*f2[key]).tolist() )  #prepare for flattening
        x= np.asarray(flatten(res)) #flatten
        x2= np.asarray(flatten(res2)) #flatten
        #x= x[~np.isnan(x)]
        #x2= x2[~np.isnan(x2)] #not getting the nans out YET
        features = np.concatenate((x,x2))
        if self.mydata == None:
            self.mydata = features
        else:
            self.mydata = np.vstack((self.mydata, features))
            #self.mydata = np.delete(self.mydata,0, axis=0)
        #self.mydata = self.mydata[:,~np.isnan(self.mydata).any(axis=0)] #erasing the NaNs
        
    #adding a comfortable function, where one can easily introduce the data
    def add_allData(self, mydata, labels):
        self.mydata = mydata
        self.labels = labels
    def train(self):
        self.rf.learnRF(self.mydata.astype("float32"), (np.asarray(self.labels)).astype("uint32").reshape(-1,1))
    
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
        self.rf.writeHDF5(outputFilename)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="trainRF",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("filepath",
                        help="read ground truth from this folder", metavar="FILE")
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
    args = parser.parse_args()

    filepath = args.filepath
    rawimage_filename = args.rawimage_filename
    initFrame = args.initFrame
    endFrame = args.endFrame
    fileFormatString = '{'+':0{}'.format(args.filename_zero_padding)+'}.h5'

    rawimage = vigra.impex.readHDF5(rawimage_filename, args.rawimage_h5_path)
    features = compute_features(rawimage,read_in_images(initFrame,endFrame, filepath, fileFormatString),initFrame,endFrame)
    mylabels = read_positiveLabels(initFrame,endFrame,filepath, fileFormatString)
    neg_labels = negativeLabels(features,mylabels)
    TC = TransitionClassifier()
    # compute featuresA for each object A from the feature matrix from Vigra
    def compute_ObjFeatures(features, obj):
        dict={}
        for key in features:
            if key == "Global<Maximum >" or key=="Global<Minimum >": #this ones have only one element
                 dict[key] = features[key]
            else:
                 dict[key] = features[key][obj]
        return dict
    
    for k in range(0,len(features)-1):
        for i in mylabels[k]:
            TC.addSample(compute_ObjFeatures(features[k], i[0]), compute_ObjFeatures(features[k+1], i[1]), 1)   #positive
        for i in neg_labels[k]:
            TC.addSample(compute_ObjFeatures(features[k], i[0]), compute_ObjFeatures(features[k+1], i[1]), 0)    #negative
    TC.train()
    TC.writeRF(args.outputFilename) #writes learned RF to disk
