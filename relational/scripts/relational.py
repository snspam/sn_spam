"""
This module generates predicates for and runs the relational model.
"""
import os
import pandas as pd


class Relational:

    def __init__(self, connections_obj, config_obj, psl_obj, tuffy_obj,
                 mrf_obj, util_obj):
        self.conns_obj = connections_obj
        self.config_obj = config_obj
        self.psl_obj = psl_obj
        self.tuffy_obj = tuffy_obj
        self.mrf_obj = mrf_obj
        self.util_obj = util_obj

    # public
    def compile_reasoning_engine(self):
        """Uses the psl module to compile the PSL and groovy scripts."""
        psl_f, _, _, _, _, _, _, _ = self._folders()
        self.psl_obj.compile(psl_f)

    def main(self, val_df, test_df, engine='all'):
        """Sets up file structure, merges predictions, and runs the model.
        val_df: validation dataframe.
        test_df: testing dataframe."""
        f = self._folders()
        psl_f, psl_d_f, tuffy_f, mrf_f, folds_f, pred_f, rel_pred_f, sts_f = f

        if val_df is not None:
            val_df = self._merge_ind_preds(val_df, 'val', pred_f)
        test_df = self._merge_ind_preds(test_df, 'test', pred_f)

        self._run_relational_model(val_df, test_df, psl_f, psl_d_f, tuffy_f,
                                   mrf_f, rel_pred_f, engine=engine)

    # private
    def _folders(self):
        """Returns absolute path directories."""
        ind_dir = self.config_obj.ind_dir
        rel_dir = self.config_obj.rel_dir
        domain = self.config_obj.domain

        psl_f = rel_dir + 'psl/'
        psl_data_f = psl_f + 'data/' + domain + '/'
        tuffy_f = rel_dir + 'tuffy/' + domain + '/'
        mrf_f = rel_dir + 'mrf/' + domain + '/'
        folds_f = ind_dir + 'data/' + domain + '/folds/'
        pred_f = ind_dir + 'output/' + domain + '/predictions/'
        rel_pred_f = rel_dir + 'output/' + domain + '/predictions/'
        status_f = rel_dir + 'output/' + domain + '/status/'
        if not os.path.exists(psl_data_f):
            os.makedirs(psl_data_f)
        if not os.path.exists(status_f):
            os.makedirs(status_f)
        folders = (psl_f, psl_data_f, tuffy_f, mrf_f, folds_f, pred_f,
                   rel_pred_f, status_f)
        return folders

    def _merge_ind_preds(self, df, dset, ind_pred_f):
        fold = self.config_obj.fold
        fname = ind_pred_f + dset + '_' + fold + '_preds.csv'

        preds_df = pd.read_csv(fname)
        df = df.merge(preds_df, on='com_id')
        return df

    def _run_psl(self, val_df, test_df, psl_f, psl_d, rel_d):
        self.psl_obj.train(val_df, psl_d, psl_f)
        self.psl_obj.infer(test_df, psl_d, psl_f, rel_d, max_size=500000)

    def _run_tuffy(self, val_df, test_df, tuffy_f):
        self.tuffy_obj.clear_data(tuffy_f)

        self.util_obj.start('\nbuilding predicates...')
        self.tuffy_obj.gen_predicates(val_df, 'val', tuffy_f)
        self.tuffy_obj.gen_predicates(test_df, 'test', tuffy_f)
        self.util_obj.end('\n\ttime: ')
        self.tuffy_obj.run(tuffy_f)
        pred_df = self.tuffy_obj.parse_output(tuffy_f)
        self.tuffy_obj.evaluate(test_df, pred_df)

    def _run_mrf(self, val_df, test_df, mrf_f, rel_d, max_edges=10000):
        if self.config_obj.epsilons == {} and val_df is not None:
            ep = self.mrf_obj.tune_epsilon(val_df, mrf_f, rel_d)
            self.mrf_obj.infer(test_df, mrf_f, rel_d, max_size=7500, ep=ep,
                               max_edges=max_edges)
        else:
            self.mrf_obj.infer(test_df, mrf_f, rel_d, max_size=7500,
                               max_edges=max_edges)

    def _run_relational_model(self, val_df, test_df, psl_f, psl_data_f,
                              tuffy_f, mrf_f, rel_pred_f, engine='psl'):
        if engine == 'psl':
            self._run_psl(val_df, test_df, psl_f, psl_data_f, rel_pred_f)
        elif engine == 'tuffy':
            self._run_tuffy(val_df, test_df, tuffy_f)
        elif engine == 'mrf':
            self._run_mrf(val_df, test_df, mrf_f, rel_pred_f)
