"""
Analyze confusion matrix and output the information about the classes that are mostly easily misclassfied.
"""

import numpy as np
import argparse
import cPickle as pickle
import cmu.confusion_matrix
from tabulate import tabulate

parser = argparse.ArgumentParser()
parser.add_argument('confusion_mat_file', type=str)
parser.add_argument('class_index_file', type=str)
parser.add_argument('k')
parser.add_argument('--dump_file', type=str, default=None)
parser.add_argument('--type', type=str, default='worst')
args = parser.parse_args()

k = int(args.k)
dump_file = args.dump_file
analyze_type = args.type

# load confusion matrix
file_path = args.confusion_mat_file
conf_mat_file = open(file_path)
confusion_mat = pickle.load(conf_mat_file)

# load class labels
classidx_file = open(args.class_index_file)
labels = []
for line in classidx_file:
    fields = line.strip().split()
    class_name = fields[-1]
    labels.append(class_name)
labels = np.array(labels)

# analyze the confusion matrix
if analyze_type != 'best' and analyze_type != 'worst':
    print('unrecognized analyze type: ' + analyze_type)
    exit(-1)

result = cmu.confusion_matrix.analyze_confusion_mat(confusion_mat, labels, k, analyze_type)

if analyze_type == 'worst':
    print(tabulate(result, headers=['class_id', 'class_label', 'class_accuracy', 'confused_id', 'confused_label', 'confused_accuracy']))
else:   # best
    print(tabulate(result, headers=['class_id', 'class_label', 'class_accuracy']))


if dump_file is not None:
    print('dumping analysis result to ' + dump_file)
    dumpf = open(dump_file, 'wb')
    pickle.dump(result, dumpf)
