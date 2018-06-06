"""
Spam comments to be classified by the relational model.
"""
import os
import numpy as np


class Comments:

    def __init__(self, config_obj, util_obj):
        self.config_obj = config_obj
        self.util_obj = util_obj

    # public
    def build(self, df, dset, data_f=None, tuffy=False, iden='0'):
        """Writes predicate info to the designated data folder.
        df: comments dataframe.
        dset: dataset (e.g. val or test).
        data_f: data folder to save predicate files.
        tuffy: boolean indicating if tuffy is the engine being used."""
        if data_f is None:
            data_f = self.define_file_folders()
        unique_df = self.drop_duplicate_comments(df)

        if tuffy:
            self.write_tuffy_predicates(unique_df, dset, data_f)
        else:
            self.write_psl_predicates(unique_df, dset, data_f, iden=iden)

    # private
    def define_file_folders(self):
        rel_dir = self.config_obj.rel_dir
        domain = self.config_obj.domain

        data_f = rel_dir + 'data/' + domain + '/'
        if not os.path.exists(data_f):
            os.makedirs(data_f)
        return data_f

    def drop_duplicate_comments(self, df):
        temp_df = df.filter(['com_id', 'ind_pred', 'label'], axis=1)
        unique_df = temp_df.drop_duplicates()
        return unique_df

    def write_psl_predicates(self, df, dset, data_f, iden='0'):
        df.to_csv(data_f + dset + '_no_label_' + iden + '.tsv',
                  columns=['com_id'], sep='\t', header=None, index=None)
        df.to_csv(data_f + dset + '_' + iden + '.tsv',
                  columns=['com_id', 'label'], sep='\t', header=None,
                  index=None)
        df.to_csv(data_f + dset + '_pred_' + iden + '.tsv',
                  columns=['com_id', 'ind_pred'], sep='\t', header=None,
                  index=None)

    def write_tuffy_predicates(self, df, dset, data_f):
        ev = open(data_f + dset + '_evidence.txt', 'w')
        q = open(data_f + dset + '_query.txt', 'w')

        for index, row in df.iterrows():
            pred = row.ind_pred
            com_id = str(int(row.com_id))
            wgt = str(np.log(self.util_obj.div0(pred, (1 - pred))))
            ev.write('Indpred(' + com_id + ', ' + wgt + ')\n')
            q.write('Spam(' + com_id + ')\n')

        ev.close()
        q.close()
