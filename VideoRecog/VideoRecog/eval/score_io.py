"""
routines for loading & writing prediction scores
"""

import numpy as np


def softmax(raw_score, T=1):
    exp_s = np.exp((raw_score - raw_score.max(axis=-1)[..., None])*T)
    sum_s = exp_s.sum(axis=-1)
    return exp_s / sum_s[..., None]


def default_aggregation_func(score_arr, normalization=True, crop_agg=None):
    """
    This is the default function for make video-level prediction
    :param score_arr: a 3-dim array with (frame, crop, class) layout
    :return:
    """
    crop_agg = np.mean if crop_agg is None else crop_agg
    if normalization:
        return softmax(crop_agg(score_arr, axis=1).mean(axis=0))
    else:
        return crop_agg(score_arr, axis=1).mean(axis=0)


def load_scores_an(file_path):
    """
    special version for handling activity net score dump
    :param file_path:
    :return:
    """
    result = np.load(file_path)
    scores = np.array(map(default_aggregation_func, result['scores'][:,0]))
    labels = None
    if 'labels' in result:
        labels = result['labels']
    return scores, labels


def load_scores(file_path):
    """
    load prediction scores
    :return: a tuple containing (scores, labels) or (scores, None) if the dump contains no labels.
    """
    result = np.load(file_path)
    scores = result['scores']
    labels = None
    if 'labels' in result:
        labels = result['labels']
    return scores, labels


def save_scores(file_path, scores, labels=None):
    """
    save prediciton scores
    :return:
    """
    np.savez(file_path, scores=scores, labels=labels)


def test():
    score_file_path = '/data01/mscvproject/code/temporal-segment-networks/results/scores/score_hmdb51_flow_1.npz'
    scores, labels = load_scores_an(score_file_path)
    print(scores.shape)

if __name__ == '__main__':
    test()
