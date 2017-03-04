"""
Test classifier
"""

import os
import sys
import argparse
from pyActionRec.action_classifier import ActionClassifier
from pyActionRec.anet_db import ANetDB
import numpy as np


anet_home = os.environ['ANET_HOME']
sys.path.append(anet_home)


def test_classifier(cls, video_names, labels):
    """
    evaluate the classifier on test data
    :param cls: classifier which implements classify()
    :param video_names: a list of video names to be processed
    :param labels: the ground truth label of the above videos
    :param use_flow: whether to use optical flow (temporal network)
    :return: (accuracy)
    """

    results = [cls.classify(video_name) for video_name in video_names]
    predicts = np.array([np.argmax(result[0]) for result in results])
    correct = np.sum(labels == predicts)
    accuracy = float(correct)/float(len(labels))
    return accuracy


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('test_video', type=str)
parser.add_argument('test_label', type=str)
parser.add_argument("--use_flow", action="store_true", default=False)
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()

# unpack arguments
test_video = args.test_video
test_label = args.test_label
use_flow = args.use_flow
gpu_id = args.gpu

# construct classifier
models = [('models/resnet200_anet_2016_deploy.prototxt',
           'models/resnet200_anet_2016.caffemodel',
           1.0, 0, True, 224)]
if use_flow:
    models.append(('models/bn_inception_anet_2016_temporal_deploy.prototxt',
                   'models/bn_inception_anet_2016_temporal.caffemodel.v5',
                   0.2, 1, False, 224))
cls = ActionClassifier(models, dev_id=gpu_id)

# loading test video file list and labels


# start testing



