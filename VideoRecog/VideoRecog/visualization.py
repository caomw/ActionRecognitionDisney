"""
This module provides all the visualization routines
"""

import matplotlib.pyplot as plt


def plot_confusion_mat(confusion_mat, dump_file):
    """
    Plot and dump confusion matrix.
    :param confusion_mat: confusion_mat of shape (num_classes, num_classes)
    :param dump_file: output image file
    :return:
    """
    fig, ax = plt.subplots()
    ax.pcolor(confusion_mat, cmap=plt.cm.Blues)
    # put the major ticks at the middle of each cell
    ax.set_xlim([0, confusion_mat.shape[1]])
    ax.set_ylim([0, confusion_mat.shape[0]])
    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    plt.savefig(dump_file)