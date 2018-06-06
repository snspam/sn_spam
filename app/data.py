import pandas as pd


class Data:

    data_dir = 'independent/data/'

    # public
    def __init__(self, generator_obj, util_obj):
        self.gen_obj = generator_obj
        self.util_obj = util_obj

    def get_rel_ids(self, dfs, domain='twitter', relations=[], sim_dir=None,
                    exact=True):
        sim_path = '%s%s/%s/' % (Data.data_dir, domain, sim_dir)
        dd = None if sim_dir is None else sim_path

        new_dfs = {}
        for dset in ['train', 'val', 'test']:
            df = dfs[dset]
            if df is not None:
                df = self.gen_obj.gen_relational_ids(df, relations,
                                                     data_dir=dd, exact=exact,
                                                     dset=dset)
            new_dfs[dset] = df
        return new_dfs

    def get_data(self, start=0, end=1000, domain='twitter', evaluation='cc'):
        t1 = self.util_obj.out('reading in data...')

        skiprows = range(1, start)
        nrows = end - start
        result = None

        if evaluation == 'tt':
            train_path = Data.data_dir + domain + '/train.csv'
            test_path = Data.data_dir + domain + '/test.csv'
            train_df = pd.read_csv(train_path, lineterminator='\n',
                                   skiprows=skiprows, nrows=nrows)
            train_df = train_df.reset_index().drop(['index'], axis=1)
            test_df = pd.read_csv(test_path, lineterminator='\n')
            result = (train_df, test_df)

        elif evaluation == 'cc':
            path = Data.data_dir + domain + '/comments.csv'
            coms_df = pd.read_csv(path, lineterminator='\n',
                                  skiprows=skiprows, nrows=nrows)
            coms_df = coms_df.reset_index().drop(['index'], axis=1)
            result = coms_df

        self.util_obj.time(t1)
        return result

    def sep_data(self, df, relations=[], domain='twitter', data='both'):
        if data == 'both':
            return df

        ids = set()
        list_filter = lambda x: True if x != [] else False
        for relation, group, group_id in relations:
            q_df = df[df[group_id].apply(list_filter)]
            ids.update(set(q_df['com_id']))

        ind_df = df[~df['com_id'].isin(ids)]
        rel_df = df[df['com_id'].isin(ids)]

        result_df = ind_df if data == 'ind' else rel_df
        return result_df

    def split_data(self, df, train_size=0.7, val_size=0.15, val_split=0.0):
        num_coms = len(df)

        if train_size == 0 and val_size == 0:  # used for tt eval
            data = {'train': df, 'val': None, 'test': None}

        elif val_size == 0:  # no relational training data
            split_ndx = int(num_coms * train_size)
            train_df = df[:split_ndx]
            test_df = df[split_ndx:]
            data = {'train': train_df, 'val': None, 'test': test_df}

        else:
            split_ndx1 = int(num_coms * train_size)
            split_ndx2 = split_ndx1 + int(num_coms * val_size)

            train_df = df[:split_ndx1]
            val_df = df[split_ndx1:split_ndx2]
            test_df = df[split_ndx2:]

            if val_split > 0.0:
                val_split_ndx = int(len(val_df) * (1 - val_split))
                val_df = val_df[val_split_ndx:]

            data = {'train': train_df, 'val': val_df, 'test': test_df}
        return data
