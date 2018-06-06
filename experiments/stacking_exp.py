"""
Module to test increasing levels of stacking.
"""
import os
import pandas as pd


class Stacking_Experiment:

    # public
    def __init__(self, config_obj, app_obj, util_obj):
        self.config_obj = config_obj
        self.app_obj = app_obj
        self.util_obj = util_obj

    def run_experiment(self, start=0, end=2000000, domain='twitter',
                       clf='lr', start_stack=0, end_stack=7,
                       relations=[], fold=0, train_size=0.8,
                       subsets=100, subset_size=1000, sim_dir=None):
        assert end_stack >= start_stack

        rel_dir = self.config_obj.rel_dir
        out_dir = rel_dir + 'output/' + domain + '/experiments/'
        self.util_obj.create_dirs(out_dir)

        fold = str(fold)
        fn = fold + '_stk.csv'

        if subset_size != -1:
            subsets = self._staggered_divide(subset_size=subset_size,
                                             start=start, end=end,
                                             subsets=subsets)
        else:
            subsets = self._divide_data(start=start, end=end, subsets=subsets)

        rows = []
        cols = ['experiment', 'ind_aupr', 'ind_auroc']

        for sub_num, (start, end) in enumerate(subsets):
            for i, stack in enumerate(range(start_stack, end_stack + 1)):
                for stack_splits in self._stack_splits(stack=stack):
                    row = ['_'.join([str(sub_num), str(start), str(end),
                                     str(stack), str(stack_splits)])]

                    d = self.app_obj.run(domain=domain, start=start, end=end,
                                         fold=fold, engine=None, clf=clf,
                                         stacking=stack, data='both',
                                         train_size=train_size,
                                         val_size=0, relations=relations,
                                         sim_dir=sim_dir,
                                         stack_splits=stack_splits)

                    for metric in ['aupr', 'auroc']:
                        row.append(d['ind'][metric])
                    rows.append(row)

                    self._write_scores_to_csv(rows, cols=cols,
                                              out_dir=out_dir, fname=fn)

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

    def _stack_splits(self, stack=0):
        ss = []

        splits = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]

        if stack == 0:
            ss.append([])

        if stack == 1:
            for split in splits:
                ss.append([split])

        elif stack == 2:
            for split1 in splits:
                for split2 in splits:
                    ss.append([split1, split2])

        return ss

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

    def _write_scores_to_csv(self, rows, cols=[], out_dir='',
                             fname='results.csv'):
        df = pd.DataFrame(rows, columns=cols)
        df.to_csv(out_dir + fname, index=None)
