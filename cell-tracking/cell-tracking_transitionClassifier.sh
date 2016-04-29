# project settings
GT_PATH=/export/home/lparcala/Fluo-N2DH-SIM/01_GT/TRA/groundtruth-frame/
# GT_PATH_moves=/export/home/lparcala/Fluo-N2DH-SIM/01_GT/TRA/
ILP=/export/home/lparcala/Fluo-N2DH-SIM/01/tracking-2015-02-27.ilp  #many ilps in the folder, taken one of it
RAW_FILE=/export/home/lparcala/Fluo-N2DH-SIM/01/stack.h5
TRANSITION_CLASSIFIER_FILE=/export/home/lparcala/Fluo-N2DH-SIM/01/struct-learn/transitionClassifier.h5 #I generate this
JSON_GRAPH=/export/home/lparcala/Fluo-N2DH-SIM/graphRFTransitions.json
JSON_WEIGHTS=/export/home/lparcala/Fluo-N2DH-SIM/weights.json
JSON_RESULTS=/export/home/lparcala/Fluo-N2DH-SIM/resultRFTransitions.json
EVENT_OUT=/export/home/lparcala/Fluo-N2DH-SIM/01_GT/TRA  #DONE
#
# executable paths
PATH_TRANS_CLASSIFIER_REPO=/export/home/lparcala/cell-tracking  #not used now
PATH_EMBRYONIC_TOOLBOX=/export/home/lparcala/Development/embryonic/toolbox
PATH_TRACKING_BINARY=/export/home/lparcala/multiHypothesesTracking/build/bin

#CTC Groundtruth
MANUAL_IMG=/export/home/lparcala/Fluo-N2DH-SIM/01_GT/TRA/man_track000.tif
MANUAL=/export/home/lparcala/Fluo-N2DH-SIM/01_GT/TRA/man_track.txt

# convert CTC tif to HDF5                 DO IT ONCE
# cd ${PATH_EMBRYONIC_TOOLBOX}
# python ctc_gt_to_hdf5.py --input-tif ${MANUAL_IMG} --input-track ${MANUAL} --output-file ${GT_PATH} --single-frames

# # train classifier
cd ${PATH_EMBRYONIC_TOOLBOX}
python train_transition_classifier.py ${GT_PATH} ${RAW_FILE} 0 10 ${TRANSITION_CLASSIFIER_FILE} --rawimage-h5-path data --time-axis-index 0 --filename-zero-padding 4
#
# create JSON graph
# cd ${PATH_EMBRYONIC_TOOLBOX}
# export LD_LIBRARY_PATH=/export/home/lparcala/gurobi650/linux64/lib
# python hypotheses_graph_to_json.py --method=conservation --without-tracklets --max-nearest-neighbors=1 --json-output ${JSON_GRAPH} --max-number-objects=2 --raw-data-file ${RAW_FILE} --raw-data-path data --labelImgPath '/TrackingFeatureExtraction/LabelImage/0/[[%d, 0, 0, 0, 0], [%d, %d, %d, %d, 1]]' --featsPath '/TrackingFeatureExtraction/RegionFeaturesDivision/0000/[[%d], [%d]]/Cell Division/%s'  --transition-classifier-filename ${TRANSITION_CLASSIFIER_FILE} ${ILP}

# #track
# cd ${PATH_TRACKING_BINARY}
# ./track -m ${JSON_GRAPH} -w ${JSON_WEIGHTS} -o ${JSON_RESULTS}


# cd ${PATH_EMBRYONIC_TOOLBOX}
# python json_result_to_events.py --model ${JSON_GRAPH} --result ${JSON_RESULTS} --ilp ${ILP} --labelImagePath '/TrackingFeatureExtraction/LabelImage/0000/[[%d, 0, 0, 0, 0], [%d, %d, %d, %d, 1]]'  --out ${EVENT_OUT}
#
# check how good we are
#python compare_tracking.py ${GT_PATH} ${EVENT_OUT}
