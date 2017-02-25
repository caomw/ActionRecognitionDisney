"""
Generate Confusion Matrix from the scores and labels
"""

import argparse
import sys
import numpy as np
import cPickle as pickle
from cmu import confusion_matrix as cm

sys.path.append('.')

from pyActionRecog.utils.video_funcs import default_aggregation_func

parser = argparse.ArgumentParser()
parser.add_argument('score_files', nargs='+', type=str)
parser.add_argument('output_matrix_file', type=str)
parser.add_argument('output_image_file', type=str)
parser.add_argument('--score_weights', nargs='+', type=float, default=None)
parser.add_argument('--crop_agg', type=str, choices=['max', 'mean'], default='mean')
args = parser.parse_args()

output_matrix_file = args.output_matrix_file
output_image_file = args.output_image_file

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

# label verification

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

# produce confusion matrix
confusion_mat = cm.produce_confusion_mat(predict, label, num_classes)

# dump confusion matrix and the plot
cm.plot_confusion_mat(confusion_mat, output_image_file)
matfile = open(output_matrix_file, 'wb')
pickle.dump(confusion_mat, matfile)
matfile.close()
