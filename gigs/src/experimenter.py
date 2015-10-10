from datamodel import DataModel
from gen_utils import *
from gigs.config.strings import *
from ml_utils import *
from sklearn import ensemble, svm

import json
import math


class Experimenter:
    """Execute and manage machine learning experiments"""
    def __init__(self, dm):
        self.dm = dm
    #enddef


    def set_datamodel(self, dm):
        self.dm = dm
        return None
    #enddef


    def get_classifier_list(self):
        clfs = [{CLASSIFIER:svm.SVC(kernel='rbf',class_weight='auto'), \
                STRING:'SVC_RBF_CLASS_WEIGHT_AUTO'},\
                {CLASSIFIER:ensemble.RandomForestClassifier(\
                max_depth=4,class_weight='auto'), \
                STRING:'RANDOM_FOREST_DEPTH_4_CLASS_WEIGHT_AUTO'},\
                ]
        return clfs
    #enddef


    def get_svm(self):
        clfs = [{CLASSIFIER:svm.SVC(kernel='rbf',class_weight='auto'), \
                STRING:'SVC_RBF_CLASS_WEIGHT_AUTO'}
                ]
        return clfs
    #enddef


    def classify_gigs(self):
        print('Calculating prediction accuracy using 5-fold cross-validation:\n')
        print('Extracting features...\n')
        featured_rows = self.dm.get_featured_gigs()
        clfs = self.get_classifier_list()
        samples = []
        labels = []
        for row in featured_rows[POS]:
            samples.append(row)
            labels.append(1)
        for row in featured_rows[NEG]:
            samples.append(row)
            labels.append(-1)
        scores = kfold_cross_val(clfs, samples, labels)
        scores = float_precise_str(means(scores))
        print(json.dumps(scores, indent=4, sort_keys=True))
        print('')
        return scores
    #enddef


    def evaluate_feature_values(self):
        print('Calculating Discriminative Value of each feature:\n')
        print('Extracting features...\n')
        feature_values = {}
        featured_rows = self.dm.get_featured_gigs()
        feature_labels = self.dm.get_feature_labels()
        clfs = self.get_svm()
        labels = []
        for row in featured_rows[POS]:
            labels.append(1)
        for row in featured_rows[NEG]:
            labels.append(-1)
        for i in range(len(feature_labels)):
            feature = feature_labels[i]
            samples = [[row[i]] for row in featured_rows[POS]]
            samples += [[row[i]] for row in featured_rows[NEG]]
            scores = kfold_cross_val(clfs, samples, labels)
            feature_values[feature] = scores[F1_SCORE] 
        #endfor
        feature_values = float_precise_str(means(feature_values))
        print(json.dumps(feature_values, indent=4, sort_keys=True))
        print('')
        return feature_values
    #enddef

#endclass
