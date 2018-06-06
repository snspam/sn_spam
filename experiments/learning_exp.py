"""
Module to generate learning curves.
"""
import os
import pandas as pd


class Learning_Experiment:

    # public
    def __init__(self, config_obj, app_obj, util_obj):
        self.config_obj = config_obj
        self.app_obj = app_obj
        self.util_obj = util_obj

    def run_experiment(self, test_start=200, test_end=300,
                       train_start=0, train_end=100,
                       learn_sizes=[100000], domain='twitter', fold=0,
                       clf='lr', engine=None, relations=[],
                       testing_relational=True, super_train=False,
                       sim_dir=None):
        assert train_end > train_start
        assert test_start >= train_end
        assert test_end > test_start
        assert fold >= 0

        rel_dir = self.config_obj.rel_dir
        out_dir = rel_dir + 'output/' + domain + '/experiments/'
        self.util_obj.create_dirs(out_dir)

        fold = str(fold)
        fn = fold + '_lrn.csv'

        train_sizes = [int(x) for x in learn_sizes]

        if not testing_relational:
            train_sizes = [int(x) for x in learn_sizes]
            ranges = self._create_ranges_independent(test_start=test_start,
                                                     test_end=test_end,
                                                     train_sizes=train_sizes)
        else:
            val_sizes = [int(x) for x in learn_sizes]
            ranges = self._create_ranges_relational(test_start=test_start,
                                                    test_end=test_end,
                                                    train_start=train_start,
                                                    train_end=train_end,
                                                    val_sizes=val_sizes)
        print(ranges)

        rows = []
        cols = ['learn_size']

        for start, end, train_size, val_size, val_split, lrn_pts in ranges:
            row = [lrn_pts]

            d = self.app_obj.run(domain=domain, start=start, end=end,
                                 fold=fold, engine=engine, clf=clf,
                                 stacking=0, data='both',
                                 train_size=train_size, val_size=val_size,
                                 val_split=val_split, relations=relations,
                                 sim_dir=sim_dir, super_train=super_train)

            if cols == ['learn_size']:
                cols.extend(['ind_aupr', 'ind_auroc'])
                for model in ['psl', 'mrf']:
                    for metric in ['aupr', 'auroc']:
                        if model == engine or engine == 'all':
                            cols.append(model + '_' + metric)

            for model in ['ind', 'psl', 'mrf']:
                for metric in ['aupr', 'auroc']:
                    if d.get(model) is not None:
                        row.append(d[model][metric])
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

    def _create_ranges_relational(self, test_start=100, test_end=200,
                                  train_start=0, train_end=40,
                                  val_sizes=[10, 20, 30, 60]):
        assert train_start >= 0
        assert train_start <= train_end
        assert test_start >= train_end
        assert test_start <= test_end

        test_pts = test_end - test_start
        train_pts = train_end - train_start
        val_pts = test_start - train_end
        train_pct = train_pts / (train_pts + val_pts + test_pts)
        val_pct = val_pts / (train_pts + val_pts + test_pts)

        range_list = []

        for i, vp in enumerate(val_sizes):
            assert vp <= test_start - train_end
            start = train_start
            end = test_end
            train_size = train_pct
            val_size = val_pct
            val_split = vp / val_pts

            rng = (start, end, train_size, val_size, val_split, vp)
            range_list.append(rng)
        return range_list

    def _create_ranges_independent(self, test_start=100, test_end=200,
                                   train_sizes=[]):
        test_size = test_end - test_start
        range_list = []

        for i, train_size in enumerate(train_sizes):
            tp = train_size / (train_size + test_size)
            start = test_start - train_size
            if start >= 0:
                range_list.append((start, test_end, train_size, 0, 0, tp))
        return range_list

    def _write_scores_to_csv(self, rows, cols=[], out_dir='',
                             fname='results.csv'):
        df = pd.DataFrame(rows, columns=cols)
        df.to_csv(out_dir + fname, index=None)
