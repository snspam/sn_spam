"""
Module to test differing featuresets.
"""
import os
import itertools
import pandas as pd


class Relations_Experiment:

    # public
    def __init__(self, config_obj, app_obj, util_obj):
        self.config_obj = config_obj
        self.app_obj = app_obj
        self.util_obj = util_obj

    def run_experiment(self, start=0, end=1000000, domain='twitter',
                       relations=['posts', 'intext', 'inhash', 'inment'],
                       clf='lgb', engine='psl', fold=0, train_size=0.8,
                       val_size=0.1, subsets=10, subset_size=100,
                       sim_dirs=[None]):

        rel_dir = self.config_obj.rel_dir
        out_dir = rel_dir + 'output/' + domain + '/experiments/'
        self.util_obj.create_dirs(out_dir)

        fold = str(fold)
        fn = fold + '_rel.csv'

        combos = self._create_combinations(relations)
        combos = [x for x in combos if len(x) == 1]

        if subset_size != -1:
            subsets = self._staggered_divide(subset_size=subset_size,
                                             start=start, end=end,
                                             subsets=subsets)
        else:
            subsets = self._divide_data(start=start, end=end, subsets=subsets)

        epsilons = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
        epsilons = [0.1] if engine == 'psl' else epsilons

        print('combos: %s' % str(combos))
        print('subsets: %s' % str(subsets))
        print('epsilons: %s' % str(epsilons))

        rows = []
        cols = ['experiment']

        for start, end in subsets:
            for relation in combos:
                for epsilon in epsilons:
                    row = ['_'.join([str(start), str(end),
                           '+'.join(relation), str(epsilon)])]

                    d = self.app_obj.run(domain=domain, start=start, end=end,
                                         fold=fold, engine=engine, clf=clf,
                                         stacking=0, data='both',
                                         train_size=train_size,
                                         val_size=val_size,
                                         relations=relation, sim_dir=None,
                                         epsilons=[epsilon])

                    if cols == ['experiment']:
                        for metric in ['aupr', 'auroc']:
                            for model in ['ind', 'psl', 'mrf']:
                                if model == 'ind':
                                    cols.append(model + '_' + metric)
                                elif model == engine or engine == 'all':
                                    cols.append(model + '_' + metric)

                    for metric in ['aupr', 'auroc']:
                        for model in ['ind', 'psl', 'mrf']:
                            if d.get(model) is not None:
                                row.append(d[model][metric])
                    rows.append(row)

                    self._write_scores_to_csv(rows, cols=cols,
                                              out_dir=out_dir, fname=fn)

    def _clear_data(self, domain='twitter'):
        ind_dir = self.config_obj.ind_dir
        rel_dir = self.config_obj.rel_dir

        fold_dir = ind_dir + '/data/' + domain + '/folds/'
        ind_pred_dir = ind_dir + '/output/' + domain + '/predictions/'
        rel_pred_dir = rel_dir + '/output/' + domain + '/predictions/'

        os.system('rm %s*.csv' % (fold_dir))
        os.system('rm %s*.csv' % (ind_pred_dir))
        os.system('rm %s*.csv' % (rel_pred_dir))

    def _staggered_divide(self, subset_size=100, subsets=10, start=0,
                          end=1000):
        data_size = end - start
        assert subset_size + subsets <= data_size
        assert subsets > 0

        incrementer = int((data_size - subset_size) / (subsets - 1))
        subsets_list = [(start, subset_size)]

        for i in range(1, subsets):
            sub_start = int(start + i * incrementer)
            sub_end = int(sub_start + subset_size)
            subset = (sub_start, sub_end)
            subsets_list.append(subset)
        return subsets_list

    def _divide_data(self, subsets=100, start=0, end=1000):
        data_size = end - start
        subset_size = data_size / subsets
        subsets_list = []

        for i in range(subsets):
            sub_start = int(start + (i * subset_size))
            sub_end = int(sub_start + subset_size)
            subset = (sub_start, sub_end)
            subsets_list.append(subset)
        return subsets_list

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
