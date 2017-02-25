"""
grab a list of files corresponding to the most easily confused cases.
"""

import numpy as np
import argparse
import cPickle as pickle
import sys
import os

from pyActionRecog import parse_directory
from pyActionRecog import parse_split_file
from pyActionRecog.utils.video_funcs import default_aggregation_func
from pyActionRecog.utils.metrics import mean_class_accuracy

sys.path.append('.')

##############
# hard-coding is nasty, but meh ~
split = 0
##############

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('score_files', nargs='+', type=str)
parser.add_argument('--score_weights', nargs='+', type=float, default=None)
parser.add_argument('--crop_agg', type=str, choices=['max', 'mean'], default='mean')
parser.add_argument('cf_analysis_file', type=str)
parser.add_argument('num_sample', type=int)
parser.add_argument('dump_file', type=str)
parser.add_argument('frame_path', type=str)
parser.add_argument('data_set', type=str)
parser.add_argument('--rgb_prefix', type=str, help="prefix of RGB frames", default='img_')
parser.add_argument('--flow_x_prefix', type=str, help="prefix of x direction flow images", default='flow_x_')
parser.add_argument('--flow_y_prefix', type=str, help="prefix of y direction flow images", default='flow_y_')
args = parser.parse_args()

# obtain video file path
f_info = parse_directory(args.frame_path, args.rgb_prefix, args.flow_x_prefix, args.flow_y_prefix)
video_list = parse_split_file(args.data_set)[split][1]
video_path = [f_info[0][video[0]] for video in video_list]

score_npz_files = [np.load(x) for x in args.score_files]

if args.score_weights is None:
    score_weights = [1] * len(score_npz_files)
else:
    score_weights = args.score_weights
    if len(score_weights) != len(score_npz_files):
        raise ValueError("Only {} weight specifed for a total of {} score files"
                         .format(len(score_weights), len(score_npz_files)))

score_list = [x['scores'][:, 0] for x in score_npz_files]
label_list = [x['labels'] for x in score_npz_files]

# score_aggregation
agg_score_list = []
for score_vec in score_list:
    agg_score_vec = [default_aggregation_func(x, normalization=False, crop_agg=getattr(np, args.crop_agg)) for x in score_vec]
    agg_score_list.append(np.array(agg_score_vec))

final_scores = np.zeros_like(agg_score_list[0])
for i, agg_score in enumerate(agg_score_list):
    final_scores += agg_score * score_weights[i]

# generate prediction
predict = np.argmax(final_scores, axis=1)
label = label_list[0]
num_classes = label.max() + 1

analysis_file = args.cf_analysis_file
num_sample = args.num_sample
output_file = args.dump_file

# load confusion matrix analysis result
analysis_f = open(analysis_file, 'rb')
analysis = pickle.load(analysis_f)

file_list = []
for class_id, class_label, _ , confused_id, confused_label, _ in analysis:
    count = 0
    sub_list = []
    for i in range(len(label)):
        if label[i] == class_id and predict[i] == confused_id:
            sub_list.append(os.path.abspath(video_path[i]))
            count += 1
            if count == num_sample:
                break
    file_list.append(sub_list)

output_f = open(output_file, 'wb')
pickle.dump(file_list, output_f)

for sublist in file_list:
    str = ''
    for item in sublist:
        str += item + '\t'
    print(str)
