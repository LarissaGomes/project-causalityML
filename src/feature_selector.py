import pandas as pd
import numpy as np
import sys
import os
import random

from scipy import stats
import lightgbm as lgb

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold

import warnings
warnings.filterwarnings('ignore')

# from tscv import GapRollForward

# Parameters
LABEL_COLUMN_NAME = 'voto'

N_FOLDS = 5
RANDOM_STATE = 1

class FeatureSelector:
    def __init__(self, df, features, target_col, set_size):
        self.df = df
        self.features = features
        self.target_col = target_col
        self.set_size = set_size
        self.DEFAULT_LGB_PARAMS = {
            "max_bin": 512,
            "learning_rate": 0.05,
            "boosting_type": "gbdt",
            "objective": "binary",
            "metric": "auc",
            "num_leaves": 150,
            "verbose": -1,
            "boost_from_average": True,
            "random_state": 1
        }

    def eval_bootstrap(self,features):
        X = self.df[features].values
        y = self.df[self.target_col].values

        aa = []
        bb = []
        cc = []
        dd = []
        for i in range(1,5):
            a = []
            b = []
            c = []
            d = []
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
            for (train, val) in cv.split(X, y):
                classifier = lgb.LGBMClassifier(**self.DEFAULT_LGB_PARAMS)

                classifier = classifier.fit(X[train], y[train])

                pred = classifier.predict(X[val])
                probas_ = classifier.predict_proba(X[val])

                auc = roc_auc_score(y[val], probas_[:, 1])
                prec = precision_score(y[val], pred)
                rec =  recall_score(y[val], pred)
                f1 = f1_score(y[val], pred)

                a.insert(len(a), auc)
                b.insert(len(b), prec)
                c.insert(len(c), rec)
                d.insert(len(d), f1)

            aa.append(np.mean(a))
            bb.append(np.mean(b))
            cc.append(np.mean(c))
            dd.append(np.mean(d))
        return np.mean(aa),np.mean(bb),np.mean(dd)

    def back_one(self, f):
        v = 0
        f1 = []
        f2 = []
        for i in f:
            f1.insert(len(f1), i)
            f2.insert(len(f2), i)
        A,B,C = self.eval_bootstrap(f1)
        z = A
        for i in f:
            f1.remove(i)
            A,B,C = self.eval_bootstrap(f1)
            print("%s,%f,%f,%f" % (f1,A,B,C))
            if A > z:
                v = 1
                z = A
                f2 = []
                for j in f1:
                    f2.insert(len(f2), j)
            f1.insert(len(f1), i)
        return v,f2
    
    def select_features(self,all_features):
        f = []
        i = 0
        best_features = ([],0)
        for f1 in all_features:
            if i == self.set_size+1: break
            if f1 in f: continue
            k = 0
            x = f1
            i = i + 1
            j = 0
            for f2 in all_features:
                if f2 in f: continue
                j = j + 1
                f.insert(len(f), f2)
                A,B,C = self.eval_bootstrap(f)
                print("%s,%f,%f,%f" % (f,A,B,C))
                if A > best_features[1]:
                    best_features = (f,A)
                z = A
                f.remove(f2)
                sys.stdout.flush()
                if z > k:
                    x = f2
                    k = z
            f.insert(len(f), x)
            if i > 2:
                v,f = self.back_one(f)
                while v == 1:
                    v,f = self.back_one(f)
                i = len(f)
        return best_features

    def rashomon(self):
        all_features = self.features
        rashomon = []
        while all_features:
            best_features = self.select_features(all_features)
            rashomon.append(best_features)
            for f in best_features[0]:
                all_features.remove(f)
        return rashomon
