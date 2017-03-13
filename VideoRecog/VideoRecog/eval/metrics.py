"""
common routines for evaluation metrics
"""

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize


def get_confusion_matrix(scores, labels):
    """
    generate confusion matrix
    """
    predicts = np.argmax(scores, axis=1)
    cf = confusion_matrix(labels, predicts).astype(float)
    print(cf[:, 1])
    return cf


def mean_class_recall(scores, labels):
    """
    class_precision = (number of hit for the class)/(number of instances predicted as the class)
    """
    if len(labels) == 0:
        return None
    predicts = np.argmax(scores, axis=1)
    cf = confusion_matrix(labels, predicts).astype(float)
    cls_cnt = cf.sum(axis=0)    # number of instances of each classes
    cls_hit = np.diag(cf)       # number of correct predicted instances of each classes
    return np.mean(cls_hit/cls_cnt)


def mean_class_precision(scores, labels):
    """
    class_precision = (number of hit for the class)/(number of instances predicted as the class)
    """
    if len(labels) == 0:
        return None
    predicts = np.argmax(scores, axis=1)
    cf = confusion_matrix(labels, predicts).astype(float)
    cls_cnt = cf.sum(axis=1)    # number of instances predicted as the class
    cls_hit = np.diag(cf)       # number of correct predicted instances of each classes
    return np.mean(cls_hit/cls_cnt)


def top_k_accuracy(scores, labels, k):
    """
    top k accuracy
    """
    N = len(labels)
    assert(N != 0)
    top_k_predicts = np.argsort(scores, axis=1)[:, -k:]    # N x K
    hit = np.array([1 if labels[i] in top_k_predicts[i] else 0 for i in range(N)])
    return float(np.sum(hit))/N


def mean_average_precision(scores, labels, classes, average='macro'):
    """
    mean average precision for multi-class classification
    :param scores: NxC numpy array
    :param labels: Nx1
    :param classes: list of class labels for label binarizing
    :param average: average method
    :return:
    """
    labels = labels.copy()
    if max(labels) > 1:
        labels = label_binarize(labels, classes=classes)
    return average_precision_score(labels, scores, average)


def accuracy(scores, labels):
    """
    Accuracy for binary classification
    accuracy = (TP+TN)/(P+N)
    """
    scores = scores.copy()
    assert (len(scores) == len(labels))
    scores[scores > 0.5] = 1
    scores[scores <= 0.5] = 0
    return float(np.sum(scores == labels))/len(labels)


def precision(scores, labels):
    """
    Precision for Binary Classification
    precision = TP / (TP+FP)
    """
    scores = scores.copy()
    assert(len(scores) == len(labels))
    scores[scores > 0.5] = 1
    scores[scores <= 0.5] = 0
    true_pos = np.sum([scores[i] and labels[i] for i in range(len(scores))])
    test_pos = np.sum(scores)
    return float(true_pos)/test_pos


def recall(scores, labels):
    """
    Recall for Binary Classification
    Recall = TP / TP + FN
    """
    scores = scores.copy()
    assert (len(scores) == len(labels))
    scores[scores > 0.5] = 1
    scores[scores <= 0.5] = 0
    true_pos = np.sum([scores[i] and labels[i] for i in range(len(scores))])
    real_pos = np.sum(labels)
    return float(true_pos) / real_pos


def eval_all(scores, labels, num_classes):

    # total performance analysis
    overall = {'map': mean_average_precision(scores, labels, range(num_classes)),
               'top1_acc': top_k_accuracy(scores, labels, 1),
               'top3_acc': top_k_accuracy(scores, labels, 3),
               'mean_class_precision': mean_class_precision(scores, labels),
               'mean_class_recall': mean_class_recall(scores, labels)}

    class_indices = np.arange(num_classes)

    # class performance analysis
    class_results = []
    for i in class_indices:
        indices = np.argwhere(labels == i).flatten()

        class_scores = scores[:, i]
        class_labels = np.zeros(labels.shape)
        class_labels[indices] = 1

        class_results.append({'class_id': i,
                              'map': mean_average_precision(class_scores, class_labels, [1]),
                              'precision': precision(class_scores, class_labels),
                              'recall': recall(class_scores, class_labels),
                              'accuracy': accuracy(class_scores, class_labels)})

    # generate confusion matrix
    cm = get_confusion_matrix(scores, labels)

    return overall, class_results, cm


def test():
    from score_io import load_scores_an
    score_file_path = '/data01/mscvproject/code/temporal-segment-networks/results/scores/score_hmdb51_flow_1.npz'
    scores, labels = load_scores_an(score_file_path)
    total_performance, class_performance, _ = eval_all(scores, labels, 51)
    print(total_performance)

if __name__ == '__main__':
    test()
