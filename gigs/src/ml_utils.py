from gigs.config.strings import *
from copy import deepcopy
from gen_utils import *
from matplotlib import pyplot as plt
from sklearn import base, cross_validation, metrics

import os


def kfold_cross_val(classifiers, x, y, train_test_indices_pair=None, \
        cv=5, do_shuffle=True):
    if type(classifiers) is not list:
        classifiers = [classifiers]
    scores = kfold_cross_val_multi_clf(classifiers, x, y, \
            train_test_indices_pair, cv=cv, do_shuffle=do_shuffle)
    if len(classifiers) == 1:
        return scores[classifiers[0][STRING]]
    return scores


def kfold_cross_val_multi_clf(classifiers, x, y, train_test_indices_pair=None,\
        cv=5, do_shuffle=True):
    default_score = {ACCURACY:[], F1_SCORE:[], PRECISION:[], RECALL:[]}
    result = dict((clf[STRING], deepcopy(default_score)) for clf in classifiers)
    if train_test_indices_pair is None:
        kf = cross_validation.KFold(len(x), n_folds=cv, shuffle=do_shuffle)
    else:
        kf = [train_test_indices_pair]
    for train_i, test_i in kf:
        train_x, train_y, test_x, test_y = [], [], [], []
        for index in train_i:
            train_x.append(x[index])
            train_y.append(y[index])
        for index in test_i:
            test_x.append(x[index])
            test_y.append(y[index])
        for clf_obj in classifiers:
            # Train classifier for actual scores
            clf_str = clf_obj[STRING]
            temp_classifier = base.clone(clf_obj[CLASSIFIER])
            temp_classifier.fit(train_x, train_y)
            test_y_pred = temp_classifier.predict(test_x)
            result[clf_str][ACCURACY].append(float(\
                    metrics.accuracy_score(test_y, test_y_pred)))
            result[clf_str][PRECISION].append(float(\
                    metrics.precision_score(test_y, test_y_pred, average='binary')))
            result[clf_str][RECALL].append(float(\
                    metrics.recall_score(test_y, test_y_pred, average='binary')))
            result[clf_str][F1_SCORE].append(float(\
                    metrics.f1_score(test_y, test_y_pred, average='binary')))
    return result


def get_predictions(classifiers, x, y, train_test_indices_pair):
    if type(classifiers) is not list:
        classifiers = [classifiers]
    preds = get_predictions_multi_clf(classifiers, x, y, train_test_indices_pair)
    if len(classifiers) == 1:
        return preds[classifiers[0]]
    return preds


def get_predictions_multi_clf(classifiers, x, y, train_test_indices_pair):
    if type(classifiers) is not list:
        classifiers = [classifiers]
    result = dict((clf, {}) for clf in classifiers)
    kf = [train_test_indices_pair]
    for train_i, test_i in kf:
        train_x, train_y, test_x, test_y = [], [], [], []
        for index in train_i:
            train_x.append(x[index])
            train_y.append(y[index])
        for index in test_i:
            test_x.append(x[index])
            test_y.append(y[index])
        for clf in classifiers:
            temp_classifier = base.clone(clf)
            temp_classifier.fit(train_x, train_y)
            test_y_pred = temp_classifier.predict(test_x)
            result[clf] = [int(i) for i in test_y_pred]
    return result


def save_plots(score_uni_map, output_dir):
    try:
        ensure_dir_exists(output_dir)
    except OSError:
        return None
    for uni in score_uni_map.keys():
        xs = score_uni_map[uni][X]
        plt.figure(figsize=(10,10))
        plt.axis([0, max(xs), 0, 1])
        for key in score_uni_map[uni].keys():
            if key == X:
                continue
            if BASE in key:
                linestyle = 'dotted'
                linewidth = 4.0
            else:
                linestyle = 'solid'
                linewidth = 2.0
            plt.plot(xs, score_uni_map[uni][key], label=key, \
                    linestyle=linestyle, linewidth=linewidth)
        plt.legend(bbox_to_anchor=(0.,-0.01,1.0,-0.02),loc=1, ncol=2, mode='expand')
        plt.suptitle(uni)
        plt.savefig(output_dir + uni.replace('/','-') + '.png')
        plt.clf()
        plt.cla()
    return None


