# Inference on ActivityNet
DATASET=ActivityNetTrimflow

for SPLIT_NUM in 1 2 3
do
    for STREAM in rgb flow
    do
	    echo dataset=${DATASET}, split=${SPLIT_NUM}, stream=${STREAM}	    

	    # data folder storing frames and optical flow
	    FRAME_PATH=../../data/${DATASET}_frame_and_flow
	    # output score file
	    SCORE_FILE=results/score_${DATASET}_${STREAM}_${SPLIT_NUM}

	    python tools/eval_net.py ${DATASET} ${SPLIT_NUM} ${STREAM} ${FRAME_PATH} \
 	    models/${DATASET}/tsn_bn_inception_${STREAM}_deploy.prototxt models/${DATASET}_split_${SPLIT_NUM}_tsn_${STREAM}_reference_bn_inception.caffemodel \
 	    --num_worker ${NUM_WORKER} --save_scores ${SCORE_FILE}
	done
done
