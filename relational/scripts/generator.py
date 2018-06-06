"""
Module to generate ids for relationships between data points.
"""
import os
import re
import time
import numpy as np
import pandas as pd


class Generator:

    def __init__(self, util_obj):
        self.util_obj = util_obj

    # public
    def gen_relational_ids(self, df, relations, data_dir=None, exact=True,
                           dset='train'):
        """Generates relational ids for a given dataframe."""
        df = df.copy()

        if len(relations) > 0:
            self.util_obj.out('generating relational ids for %s:' % dset)

        for relation, group, group_id in relations:
            t1 = time.time()
            self.util_obj.out(relation + '...')
            if exact:
                df = self._gen_group_id(df, group_id)
            else:
                df = self._gen_group_id_lists(df, group_id, data_dir=data_dir)
            self.util_obj.time(t1)
        return df

    def rel_df_from_df(self, df, g_id, exact=True):
        """Obtain just the relational rows given the g_id."""
        if exact:
            r_df = self._rel_df_from_id(df, g_id)
        else:
            r_df = self._rel_df_from_id_lists(df, g_id)
        return r_df

    # private
    def _gen_group_id(self, df, g_id):
        r_df = self._rel_df(df, g_id, data_dir=None)
        df = df.merge(r_df, on='com_id', how='left')
        df[g_id] = df[g_id].fillna(-1).apply(int)
        return df

    def _gen_group_id_lists(self, df, g_id, data_dir=None):
        r_df = self._rel_df(df, g_id, data_dir=data_dir)

        if len(r_df) == 0:
            if g_id in list(df):
                df = df.rename(columns={g_id: g_id.replace('_id', '')})

            df[g_id] = np.nan
            df[g_id] = df[g_id].astype(object)
        else:
            g = r_df.groupby('com_id')

            d = {}
            for com_id, g_df in g:
                d[com_id] = [list(g_df[g_id])]

            r_df = pd.DataFrame.from_dict(d, orient='index').reset_index()
            r_df.columns = ['com_id', g_id]

            if g_id in list(df):
                df = df.rename(columns={g_id: g_id.replace('_id', '')})
            df = df.merge(r_df, on='com_id', how='left')

        for row in df.loc[df[g_id].isnull(), g_id].index:
            df.at[row, g_id] = []
        return df

    def _rel_df(self, df, g_id, data_dir=None):
        df = df.copy()

        if g_id not in list(df):
            if g_id == 'text_gid':
                r_df = self._text_ids(df, g_id, data_dir=data_dir)

            elif g_id == 'hash_gid':
                cols = ['hashtag']
                df['hashtag'] = df.text.str.extractall(r'(#\w+)')\
                    .reset_index().groupby('level_0')[0]\
                    .agg(lambda x: ''.join([i.lower() for i in x]))
                r_df = self._cols_to_ids(df, g_id, cols=cols)

            elif g_id == 'ment_gid':
                cols = ['mention']
                df['mention'] = df.text.str.extractall(r'(@\w+)')\
                    .reset_index().groupby('level_0')[0]\
                    .agg(lambda x: ''.join([i.lower() for i in x]))
                r_df = self._cols_to_ids(df, g_id, cols=cols)

            elif g_id == 'link_gid':
                cols = ['link']
                df['link'] = df.text.str.extractall(r'(http[^\s]+)')\
                    .reset_index().groupby('level_0')[0]\
                    .agg(lambda x: ''.join([i.lower() for i in x]))
                r_df = self._cols_to_ids(df, g_id, cols=cols)

            elif g_id == 'unicodecnt_gid':
                cols = ['unicode_cnt']
                df['unicode_cnt'] = df.text.str.count(r'(\\u\S\S\S\S)')
                r_df = self._cols_to_ids(df, g_id, cols=cols)

            elif g_id == 'unicode_gid':
                cols = ['unicode']
                df['unicode'] = df.text.str.extractall(r'(\\u\S\S\S\S)')\
                    .reset_index().groupby('level_0')[0]\
                    .agg(lambda x: ''.join([i.lower() for i in x]))
                r_df = self._cols_to_ids(df, g_id, cols=cols)

            elif g_id in ['ip_gid', 'channel_gid', 'app_gid', 'os_gid',
                          'device_gid']:
                col = [g_id.replace('_gid', '')]
                r_df = self._cols_to_ids(df, g_id, cols=col)

            elif g_id == 'usrapp_gid':
                cols = ['ip', 'os', 'device', 'app']
                r_df = self._cols_to_ids(df, g_id, cols=cols)

            elif g_id == 'usrad_gid':
                cols = ['ip', 'os', 'device', 'app', 'channel']
                r_df = self._cols_to_ids(df, g_id, cols=cols)

            elif g_id == 'usrhour_gid':
                df['click_time'] = pd.to_datetime(df['click_time'])
                df['hour'] = df['click_time'].dt.hour
                cols = ['ip', 'os', 'device', 'app', 'hour']
                r_df = self._cols_to_ids(df, g_id, cols=cols)

            elif g_id == 'usrmin_gid':
                df['click_time'] = pd.to_datetime(df['click_time'])
                df['hour'] = df['click_time'].dt.hour
                df['min'] = df['click_time'].dt.minute
                cols = ['ip', 'os', 'device', 'app', 'hour', 'min']
                r_df = self._cols_to_ids(df, g_id, cols=cols)

            elif g_id == 'usrsec_gid':
                df['click_time'] = pd.to_datetime(df['click_time'])
                df['hour'] = df['click_time'].dt.hour
                df['min'] = df['click_time'].dt.minute
                df['sec'] = df['click_time'].dt.second
                cols = ['ip', 'os', 'device', 'app', 'hour', 'min', 'sec']
                r_df = self._cols_to_ids(df, g_id, cols=cols)

            elif g_id == 'post_gid':
                cols = ['user_id']
                r_df = self._cols_to_ids(df, g_id, cols=cols)

            elif g_id == 'track_gid':
                cols = ['track_id']
                r_df = self._cols_to_ids(df, g_id, cols=cols)

            elif g_id == 'usrtrack_gid':
                cols = ['user_id', 'track_id']
                r_df = self._cols_to_ids(df, g_id, cols=cols)

            elif g_id == 'usrtext_gid':
                cols = ['user_id', 'text']
                r_df = self._cols_to_ids(df, g_id, cols=cols)

            elif g_id == 'usrrt_gid':
                cols = ['user_id', 'retweet']
                df['retweet'] = df.text.apply(lambda x: 1 if 'RT' in x else 0)
                r_df = self._cols_to_ids(df, g_id, cols=cols)

            elif g_id == 'usrhashment_gid':
                cols = ['user_id', 'hashtag', 'mention']
                df['hashtag'] = df.text.str.extractall(r'(#\w+)')\
                    .reset_index().groupby('level_0')[0]\
                    .agg(lambda x: ''.join([i.lower() for i in x]))
                df['mention'] = df.text.str.extractall(r'(@\w+)')\
                    .reset_index().groupby('level_0')[0]\
                    .agg(lambda x: ''.join([i.lower() for i in x]))
                r_df = self._cols_to_ids(df, g_id, cols=cols)

            elif g_id == 'hashment_gid':
                cols = ['hashtag', 'mention']
                df['hashtag'] = df.text.str.extractall(r'(#\w+)')\
                    .reset_index().groupby('level_0')[0]\
                    .agg(lambda x: ''.join([i.lower() for i in x]))
                df['mention'] = df.text.str.extractall(r'(@\w+)')\
                    .reset_index().groupby('level_0')[0]\
                    .agg(lambda x: ''.join([i.lower() for i in x]))
                r_df = self._cols_to_ids(df, g_id, cols=cols)

            elif g_id == 'hashmentlink_gid':
                cols = ['hashtag', 'mention', 'link']
                df['hashtag'] = df.text.str.extractall(r'(#\w+)')\
                    .reset_index().groupby('level_0')[0]\
                    .agg(lambda x: ''.join([i.lower() for i in x]))
                df['mention'] = df.text.str.extractall(r'(@\w+)')\
                    .reset_index().groupby('level_0')[0]\
                    .agg(lambda x: ''.join([i.lower() for i in x]))
                df['link'] = df.text.str.extractall(r'(http[^\s]+)')\
                    .reset_index().groupby('level_0')[0]\
                    .agg(lambda x: ''.join([i.lower() for i in x]))
                r_df = self._cols_to_ids(df, g_id, cols=cols)

            elif g_id == 'rthash_gid':
                cols = ['retweet', 'hashtag']
                df['retweet'] = df.text.apply(lambda x: 1 if 'RT' in x else 0)
                df['hashtag'] = df.text.str.extractall(r'(#\w+)')\
                    .reset_index().groupby('level_0')[0]\
                    .agg(lambda x: ''.join([i.lower() for i in x]))
                r_df = self._cols_to_ids(df, g_id, cols=cols)

            elif g_id == 'usrrthash_gid':
                cols = ['user_id', 'retweet', 'hashtag']
                df['retweet'] = df.text\
                    .apply(lambda x: 1 if 'RT' in x else 0)
                df['hashtag'] = df.text.str.extractall(r'(#\w+)')\
                    .reset_index().groupby('level_0')[0]\
                    .agg(lambda x: ''.join([i.lower() for i in x]))
                r_df = self._cols_to_ids(df, g_id, cols=cols)

            elif g_id == 'usrhash_gid':
                cols = ['user_id', 'hashtag']
                df['hashtag'] = df.text.str.extractall(r'(#\w+)')\
                    .reset_index().groupby('level_0')[0]\
                    .agg(lambda x: ''.join([i.lower() for i in x]))
                r_df = self._cols_to_ids(df, g_id, cols=cols)

            elif g_id == 'usrment_gid':
                cols = ['user_id', 'mention']
                df['mention'] = df.text.str.extractall(r'(@\w+)')\
                    .reset_index().groupby('level_0')[0]\
                    .agg(lambda x: ''.join([i.lower() for i in x]))
                r_df = self._cols_to_ids(df, g_id, cols=cols)

            elif g_id == 'usrlink_gid':
                cols = ['user_id', 'link']
                df['link'] = df.text.str.extractall(r'(http[^\s]+)')\
                    .reset_index().groupby('level_0')[0]\
                    .agg(lambda x: ''.join([i.lower() for i in x]))
                r_df = self._cols_to_ids(df, g_id, cols=cols)
        else:
            r_df = self._keep_relational_data(df, g_id)
        return r_df

    def _cols_to_ids(self, df, g_id='text_id', cols=['text']):
        g_df = df.groupby(cols).size().reset_index().rename(columns={0: 's'})
        g_df = g_df[g_df['s'] > 1]
        g_df[g_id] = list(range(1, len(g_df) + 1))
        g_df = g_df.drop(['s'], axis=1)

        r_df = df.merge(g_df, on=cols)
        r_df = r_df.filter(items=['com_id', g_id])
        r_df[g_id] = r_df[g_id].apply(int)
        return r_df

    def _text_ids(self, df, g_id, data_dir=None):
        fp = None if data_dir is None else data_dir + 'text_sim.csv'

        if data_dir is not None and os.path.exists(fp):
            self.util_obj.out('reading sim file...', 0)
            r_df = pd.read_csv(fp)
            r_df = r_df[r_df['com_id'].isin(df['com_id'])]
            g_df = r_df.groupby(g_id).size().reset_index()
            g_df = g_df[g_df[0] > 1]
            r_df = r_df[r_df[g_id].isin(g_df[g_id])]
        else:
            df = df[df['text'] != '']
            r_df = self._cols_to_ids(df, g_id=g_id, cols=['text'])
        return r_df

    def _string_ids(self, df, g_id, regex=r'(#\w+)', data_dir=None):
        fp = ''
        if data_dir is not None:
            hash_path = data_dir + 'hashtag_sim.csv'
            ment_path = data_dir + 'mention_sim.csv'
            link_path = data_dir + 'link_sim.csv'

            if regex == r'(#\w+)':
                fp = hash_path
            elif regex == r'(@\w+)':
                fp = ment_path
            elif regex == r'(http[^\s]+)':
                fp = link_path

        if data_dir is not None and os.path.exists(fp):
            self.util_obj.out('reading sim file...', 0)
            r_df = pd.read_csv(fp)
            r_df = r_df[r_df['com_id'].isin(df['com_id'])]
            g_df = r_df.groupby(g_id).size().reset_index()
            g_df = g_df[g_df[0] > 1]
            r_df = r_df[r_df[g_id].isin(g_df[g_id])]

        else:
            group = g_id.replace('_id', '')
            regex = re.compile(regex)
            inrel = []

            for _, row in df.iterrows():
                s = self._get_items(row.text, regex)
                inrel.append({'com_id': row.com_id, group: s})

            inrel_df = pd.DataFrame(inrel).drop_duplicates()
            inrel_df = inrel_df[inrel_df[group] != '']
            r_df = self._cols_to_ids(inrel_df, g_id=g_id, cols=[group])
        return r_df

    def _get_items(self, text, regex, str_form=True):
        items = regex.findall(str(text))[:10]
        result = sorted([x.lower() for x in items])
        if str_form:
            result = ''.join(result)
        return result

    def _keep_relational_data(self, df, g_id):
        g_df = df.groupby(g_id).size().reset_index()
        g_df = g_df[g_df[0] > 1]
        r_df = df[df[g_id].isin(g_df[g_id])]
        return r_df

    def _rel_df_from_id(self, df, g_id):
        r_df = df[df[g_id] != -1]
        return r_df

    def _rel_df_from_id_lists(self, df, g_id):
        rows = []

        headers = list(df)
        h = {h: i + 1 for i, h in enumerate(headers)}

        for r in df.itertuples():
            com_id = r[h['com_id']]
            rel_ids = r[h[g_id]]

            for rel_id in rel_ids:
                rows.append((com_id, rel_id))

        rel_df = pd.DataFrame(rows, columns=['com_id', g_id])
        return rel_df
