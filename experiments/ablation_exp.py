"""
Module to test differing featuresets.
"""
import os
import itertools
import pandas as pd


class Ablation_Experiment:

    # public
    def __init__(self, config_obj, app_obj, util_obj):
        self.config_obj = config_obj
        self.app_obj = app_obj
        self.util_obj = util_obj

    def run_experiment(self, start=0, end=1000000, domain='twitter',
                       featuresets=['base', 'content', 'graph', 'sequential'],
                       clf='lr', fold=0, train_size=0.8, relations=[],
                       analyze_subgraphs=False, param_search='single'):
        rel_dir = self.config_obj.rel_dir
        out_dir = rel_dir + 'output/' + domain + '/experiments/'
        self.util_obj.create_dirs(out_dir)

        fold = str(fold)
        fn = fold + '_abl.csv'

        combos = self._create_combinations(featuresets)

        # filter combos
        new_combos = []
        for combo in combos:
            if 'sequential' in combo and 'aggregate' in combo:
                new_combos.append(combo)
            elif 'sequential' not in combo and 'aggregate' not in combo:
                new_combos.append(combo)
        combos = new_combos

        print(combos)

        rows = []
        cols = ['featureset', 'aupr', 'auroc']

        for featuresets in combos:
            row = ['+'.join(featuresets)]
            d = self.app_obj.run(domain=domain, start=start, end=end,
                                 fold=fold, engine=None, clf=clf,
                                 stacking=0, data='both',
                                 train_size=train_size, val_size=0,
                                 relations=relations,
                                 featuresets=featuresets,
                                 analyze_subgraphs=analyze_subgraphs,
                                 param_search=param_search)
            row.append(d['ind']['aupr'])
            row.append(d['ind']['auroc'])
            rows.append(row)
            self._write_scores_to_csv(rows, cols=cols, out_dir=out_dir,
                                      fname=fn)

    # private
    def _clear_data(self, domain='twitter'):
        ind_dir = self.config_obj.ind_dir
        rel_dir = self.config_obj.rel_dir

        fold_dir = ind_dir + '/data/' + domain + '/folds/'
        ind_pred_dir = ind_dir + '/output/' + domain + '/predictions/'
        rel_pred_dir = rel_dir + '/output/' + domain + '/predictions/'

        os.system('rm %s*.csv' % (fold_dir))
        os.system('rm %s*.csv' % (ind_pred_dir))
        os.system('rm %s*.csv' % (rel_pred_dir))

    def _create_combinations(self, fsets):
        all_sets = []

        for L in range(1, len(fsets) + 1):
            for combo in itertools.combinations(fsets, L):
                all_sets.append(list(combo))
        return all_sets

    def _write_scores_to_csv(self, rows, cols=[], out_dir='',
                             fname='results.csv'):
        df = pd.DataFrame(rows, columns=cols)
        df.to_csv(out_dir + fname, index=None)
