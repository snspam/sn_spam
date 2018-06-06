"""
Module that classifies data using an independent model.
"""
import os
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.sparse import csr_matrix


class Classification:
    def __init__(self, config_obj, features_obj, util_obj):
        self.config_obj = config_obj
        self.feats_obj = features_obj
        self.util_obj = util_obj

    # public
    def main(self, train_df, test_df, dset='test'):
        """Constructs paths, merges data, converts, and processes data to be
        read by the independent model.
        train_df: original training comments dataframe.
        test_df: original testing comments dataframe.
        dset: datatset to test (e.g. 'val', 'test')."""
        stacking = self.config_obj.stacking

        if stacking > 0:
            self.do_stacking(train_df, test_df, dset, stacking=stacking)
        else:
            self.do_normal(train_df, test_df, dset)

    # private
    def do_stacking(self, train_df, test_df, dset='test', stacking=1):
        ut = self.util_obj
        fold = self.config_obj.fold
        clf = self.config_obj.classifier
        ps = self.config_obj.param_search
        ts = self.config_obj.tune_size
        fsets = self.config_obj.featuresets
        ev = self.config_obj.evaluation
        ss = self.config_obj.stack_splits

        ut.out('stacking with %d stack(s)...' % stacking)

        image_f, pred_f, model_f = self.file_folders()
        trains = self.split_training_data(train_df, splits=stacking + 1, ss=ss)
        test_df = test_df.copy()

        s = '\nbuilding features for %s: %d, stack: %d'

        for i in range(len(trains)):
            ut.out(s % ('train', i, i))
            d_tr, cv = self.build_and_merge(trains[i], 'train', t=i)
            learner = self.util_obj.train(d_tr, clf, ps, ts)

            for j in range(i + 1, len(trains)):
                ut.out(s % ('train', j, i))
                d_te, _ = self.build_and_merge(trains[j], 'test', cv=cv, t=i)
                te_preds, ids = self.util_obj.test(d_te, learner, fsets)
                trains[j] = self.append_preds(trains[j], te_preds, ids)

            ut.out(s % ('test', i, i))
            d_te, _ = self.build_and_merge(test_df, 'test', cv=cv, t=i)
            te_preds, ids = self.util_obj.test(d_te, learner, fsets)
            test_df = self.append_preds(test_df, te_preds, ids)

        self.util_obj.evaluate(d_te, te_preds)
        self.util_obj.save_preds(te_preds, ids, fold, pred_f, dset, ev)

        if not any(x in fsets for x in ['ngrams', 'all']):
            _, _, _, feats = d_te
            self.util_obj.plot_features(learner, clf, feats, image_f + fold)

    def do_normal(self, train_df, test_df, dset='test'):
        ut = self.util_obj
        fold = self.config_obj.fold
        clf = self.config_obj.classifier
        ps = self.config_obj.param_search
        ts = self.config_obj.tune_size
        fsets = self.config_obj.featuresets
        ev = self.config_obj.evaluation

        ut.out('normal...')

        image_f, pred_f, model_f = self.file_folders()

        # train base learner using training set.
        ut.out('\nbuilding features for train...')
        d_tr, cv = self.build_and_merge(train_df, 'train')
        learner = self.util_obj.train(d_tr, clf, ps, ts)

        # test learner on test set.
        ut.out('\nbuilding feautures for test...')
        d_te, _ = self.build_and_merge(test_df, 'test', cv=cv)
        y_score, ids = self.util_obj.test(d_te, learner, fsets)
        self.util_obj.evaluate(d_te, y_score)
        self.util_obj.save_preds(y_score, ids, fold, pred_f, dset, ev)

        if not any(x in fsets for x in ['ngrams', 'all']):
            _, _, _, feats = d_te
            self.util_obj.plot_features(learner, clf, feats, image_f + fold)

    def file_folders(self):
        ind_dir = self.config_obj.ind_dir
        domain = self.config_obj.domain

        out_f = ind_dir + 'output/' + domain
        image_f = out_f + '/images/'
        pred_f = out_f + '/predictions/'
        model_f = out_f + '/models/'

        if not os.path.exists(image_f):
            os.makedirs(image_f)
        if not os.path.exists(pred_f):
            os.makedirs(pred_f)
        if not os.path.exists(model_f):
            os.makedirs(model_f)
        return image_f, pred_f, model_f

    def dataframe_to_matrix(self, feats_df):
        return csr_matrix(feats_df.astype(float).as_matrix())

    def stack_matrices(self, feats_m, c_csr):
        stack = [feats_m]
        if c_csr is not None:
            stack.append(c_csr)
        return hstack(stack).tocsr()

    def extract_ids_and_labels(self, df):
        ids = df['com_id'].values
        labels = df['label'].values if 'label' in list(df) else None
        return ids, labels

    def prepare(self, df, fdf, m):
        t1 = self.util_obj.out('merging features...')

        feats_m = self.dataframe_to_matrix(fdf)
        x = self.stack_matrices(feats_m, m)
        self.util_obj.out(str(x.shape) + '...', 0)

        ids, y = self.extract_ids_and_labels(df)

        self.util_obj.time(t1)
        return x, y, ids

    def build_and_merge(self, df, dset, cv=None, t=0):
        fdf, fl, m, cv = self.feats_obj.build(df, dset, stack=t, cv=cv)
        x, y, ids = self.prepare(df, fdf, m)
        return (x, y, ids, fl), cv

    def append_preds(self, test_df, test_probs, id_te):
        if 'noisy_label' in test_df:
            del test_df['noisy_label']

        preds = list(zip(id_te, test_probs[:, 1]))
        preds_df = pd.DataFrame(preds, columns=['com_id', 'noisy_label'])
        new_test_df = test_df.merge(preds_df, on='com_id', how='left')
        return new_test_df

    def split_training_data(self, train_df, splits=2, ss=[]):
        train_dfs = []

        if ss == []:
            train_dfs = np.array_split(train_df, splits)

        else:
            leftover = train_df

            for s in ss:
                ndx = int(s * len(leftover))
                save = leftover[:ndx]
                leftover = leftover[ndx:]
                train_dfs.append(save)
            train_dfs.append(leftover)

        return train_dfs
