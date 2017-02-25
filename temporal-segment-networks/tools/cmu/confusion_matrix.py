"""
everything related to confusion matrix
"""

import numpy as np
import matplotlib.pyplot as plt


def produce_confusion_mat(preds, labels, num_class):
    """
    Produce confusion matrix
    :param preds: predicted labels
    :param labels: ground-truth labels
    :param num_class: number of classes
    :return: confusion matrix with shape (num_class, num_class)
    """
    confusion = np.zeros((num_class, num_class))
    for (pred, label) in zip(preds, labels):
        confusion[label, pred] += 1
    return confusion


def plot_confusion_mat(confusion_mat, dump_file):
    """
    Plot and dump confusion matrix.
    :param confusion_mat: confusion_mat of shape (num_classes, num_classes)
    :param dump_file: output image file
    :return:
    """
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(confusion_mat, cmap=plt.cm.Blues)
    # put the major ticks at the middle of each cell
    ax.set_xlim([0, confusion_mat.shape[1]])
    ax.set_ylim([0, confusion_mat.shape[0]])
    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    plt.savefig(dump_file)


def analyze_confusion_mat(confusion_mat, labels, k, analyze_type='worst'):
    """
    Analyze confusion matrix by finding the top k classes that are most easily getting confused.
    :param confusion_mat: confusion matrix
    :param labels: class labels
    :param k: k as mentioned in the description
    :param analyze_type: worst/best
    :return: a list of tuple (class_id, class_name, accuracy, 1st_confused_class_id, 1st_confused_class_name, 1st_confused_rate)
    """
    num_classes = confusion_mat.shape[0]
    row_sum = confusion_mat.sum(axis=1).reshape(num_classes, 1)
    confusion_mat_ratio = confusion_mat/row_sum
    accuracy = np.array([confusion_mat_ratio[i, i] for i in range(0, num_classes)])

    rank = None
    if analyze_type == 'worst':
        rank = accuracy.argsort()
    elif analyze_type == 'best':
        rank = accuracy.argsort()[::-1]
    else:
        print('unrecognized analyze type: ' + analyze_type)
        exit(-1)

    class_ids = rank[:k]
    class_labels = labels[class_ids]
    class_accuracy = accuracy[class_ids]

    if analyze_type == 'worst':
        tmp = confusion_mat_ratio.argsort(axis=1)
        confused_ids = []
        for i in range(num_classes):
            if tmp[i, -1] == i:
                confused_ids.append(tmp[i, -2])
            else:
                confused_ids.append(tmp[i, -1])
        confused_ids = np.array(confused_ids)
        confused_labels = labels[confused_ids]
        confused_labels = confused_labels[class_ids]
        confused_accuracy = np.array([confusion_mat_ratio[i][confused_ids[i]] for i in range(num_classes)])
        confused_accuracy = confused_accuracy[class_ids]
        confused_ids = confused_ids[class_ids]
        return list(zip(class_ids, class_labels, class_accuracy, confused_ids, confused_labels, confused_accuracy))
    else:
        return list(zip(class_ids, class_labels, class_accuracy))