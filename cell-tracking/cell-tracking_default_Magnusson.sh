# project settings
GT_PATH=/export/home/lparcala/Fluo-N2DH-SIM/01_GT/TRA/groundtruth-frame
ILP=/export/home/lparcala/Fluo-N2DH-SIM/01/tracking-2015-02-27.ilp
SEG=/export/home/lparcala/Fluo-N2DH-SIM/01/Magnusson_segmentation.h5 
RAW_FILE=/export/home/lparcala/Fluo-N2DH-SIM/01/stack.h5
TRANSITION_CLASSIFIER_FILE=/export/home/lparcala/Fluo-N2DH-SIM/01/struct-learn/classifier.h5 #???
JSON_GRAPH=/export/home/lparcala/Fluo-N2DH-SIM/graphRFTransitions.json
JSON_WEIGHTS=/export/home/lparcala/Fluo-N2DH-SIM/weights.json
JSON_RESULTS=/export/home/lparcala/Fluo-N2DH-SIM/resultRFTransitions.json
EVENT_OUT=/export/home/lparcala/Fluo-N2DH-SIM/01_RES  #DONE
#
# executable paths
PATH_TRANS_CLASSIFIER_REPO=/export/home/lparcala/cell-tracking
PATH_EMBRYONIC_TOOLBOX=/export/home/lparcala/Development/embryonic/toolbox
PATH_TRACKING_BINARY=/export/home/lparcala/multiHypothesesTracking/build/bin
#

# convert Magnussons tiff segmentation to hdf5 format    DO THIS ONCE
# cd ${PATH_EMBRYONIC_TOOLBOX}
# python segmentation_to_hdf5.py --tif `find /export/home/lparcala/Magnusson-SEG/01/*.tif | sort` --hdf5Path ${ILP} --hdf5-image-path '/TrackingFeatureExtraction/LabelImage/0/[[%d, 0, 0, 0, 0], [%d, %d, %d, %d, 1]]'

# create JSON graph
cd ${PATH_EMBRYONIC_TOOLBOX}
export LD_LIBRARY_PATH=/export/home/lparcala/gurobi650/linux64/lib
python hypotheses_graph_to_json.py --method=conservation --without-tracklets --max-nearest-neighbors=1 --json-output ${JSON_GRAPH} --max-number-objects=2 --raw-data-file ${RAW_FILE} --raw-data-path data --labelImgPath '/TrackingFeatureExtraction/LabelImage/0/[[%d, 0, 0, 0, 0], [%d, %d, %d, %d, 1]]' ${SEG} --objCountFile ${ILP} --divFile ${ILP} #--transition-classifier-filename ${TRANSITION_CLASSIFIER_FILE} 

#track
cd ${PATH_TRACKING_BINARY}
./track -m ${JSON_GRAPH} -w ${JSON_WEIGHTS} -o ${JSON_RESULTS}

# translate results to HDF
cd ${PATH_EMBRYONIC_TOOLBOX}
python json_result_to_events.py --model ${JSON_GRAPH} --result ${JSON_RESULTS} --ilp ${ILP} --labelImagePath '/TrackingFeatureExtraction/LabelImage/0/[[%d, 0, 0, 0, 0], [%d, %d, %d, %d, 1]]'  --out ${EVENT_OUT}

# translate results to CTC tiff
python hdf5_to_ctc.py --output-dir ${EVENT_OUT} --input-files `find /export/home/lparcala/Fluo-N2DH-SIM/01_RES/*.h5 | sort` --label-image-path segmentation/labels

# check how good we are
#python compare_tracking.py ${GT_PATH} ${EVENT_OUT}