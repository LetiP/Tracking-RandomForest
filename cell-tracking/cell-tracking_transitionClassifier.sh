# project settings
GT_PATH=/export/home/lparcala/Fluo-N2DH-SIM/01_GT/SEG/  #sure?? or GT from 01 folder?
GT_PATH_moves=/export/home/lparcala/Fluo-N2DH-SIM/01_GT/TRA/
ILP=/export/home/lparcala/Fluo-N2DH-SIM/01/tracking-2015-02-27.ilp  #many ilps in the folder, taken one of it
RAW_FILE=/export/home/lparcala/Fluo-N2DH-SIM/01/Objects-2015-02-26.h5 ###???? RawData-singleChannel.h5
TRANSITION_CLASSIFIER_FILE=/export/home/lparcala/Fluo-N2DH-SIM/01/struct-learn/transitionClassifier.h5 #???
JSON_GRAPH=/export/home/lparcala/Fluo-N2DH-SIM/graphRFTransitions.json
JSON_WEIGHTS=/export/home/lparcala/Fluo-N2DH-SIM/weights.json
JSON_RESULTS=/export/home/lparcala/Fluo-N2DH-SIM/resultRFTransitions.json
EVENT_OUT=/export/home/lparcala/cell-tracking/result  #DONE
#
# executable paths
PATH_TRANS_CLASSIFIER_REPO=/export/home/lparcala/cell-tracking
PATH_EMBRYONIC_TOOLBOX=/export/home/lparcala/embryonic/toolbox
PATH_TRACKING_BINARY=/export/home/lparcala/multiHypothesesTracking/build/bin
#
# train classifier
cd ${PATH_TRANS_CLASSIFIER_REPO}
python TransitionClassifier.py ${GT_PATH} ${GT_PATH_moves} ${RAW_FILE} 0 10 ${TRANSITION_CLASSIFIER_FILE} --rawimage-h5-path exported_data --time-axis-index 0 --filename-zero-padding 04 --path-in-hdf5 'exported_data'

# create JSON graph
cd ${PATH_EMBRYONIC_TOOLBOX}
export LD_LIBRARY_PATH=/export/home/lparcala/gurobi650/linux64/lib
python hypotheses_graph_to_json.py --method=conservation --without-tracklets --max-nearest-neighbors=1 --json-output ${JSON_GRAPH} --max-number-objects=2 --raw-data-file ${RAW_FILE} --raw-data-path exported_data --labelImgPath '/TrackingFeatureExtraction/LabelImage/0/[[%d, 0, 0, 0, 0], [%d, %d, %d, %d, 1]]' --transition-classifier-filename ${TRANSITION_CLASSIFIER_FILE} ${ILP}

#track
cd ${PATH_TRACKING_BINARY}
./track -m ${JSON_GRAPH} -w ${JSON_WEIGHTS} -o ${JSON_RESULTS}


# cd ${PATH_EMBRYONIC_TOOLBOX}
# python json_result_to_events.py --model ${JSON_GRAPH} --result ${JSON_RESULTS} --ilp ${ILP} --labelImagePath '/TrackingFeatureExtraction/LabelImage/0000/[[%d, 0, 0, 0, 0], [%d, %d, %d, %d, 1]]'  --out ${EVENT_OUT}
#
# check how good we are
#python compare_tracking.py ${GT_PATH} ${EVENT_OUT}
