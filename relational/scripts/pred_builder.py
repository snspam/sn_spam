"""
Module to group comments of a certain relation together.
"""


class PredicateBuilder:

    def __init__(self, config_obj, comments_obj, gen_obj, util_obj):
        self.config_obj = config_obj
        self.comments_obj = comments_obj
        self.gen_obj = gen_obj
        self.util_obj = util_obj

    # public
    def build_comments(self, df, dset, data_f, tuffy=False, iden='0'):
        self.comments_obj.build(df, dset, data_f, tuffy=tuffy, iden=iden)

    def build_relations(self, relation, group, group_id, df, dset, data_f,
                        tuffy=False, iden='0'):
        exact = self.config_obj.exact
        r_df = self.gen_obj.rel_df_from_df(df, group_id, exact=exact)
        g_df = self._get_group_df(r_df, group_id)
        r_df = r_df[r_df[group_id].isin(g_df[group_id])]

        if tuffy:
            self._write_tuffy_predicates(dset, r_df, relation, group_id,
                                         data_f)
        else:
            self._write_psl_predicates(dset, r_df, g_df, relation, group,
                                       group_id, data_f, iden=iden)
        return r_df

    # private
    def _get_group_df(self, r_df, group_id):
        g_df = r_df.groupby(group_id).size().reset_index()
        g_df.columns = [group_id, 'size']
        g_df = g_df[g_df['size'] > 1]
        return g_df

    def _write_psl_predicates(self, dset, r_df, g_df, relation, group,
                              group_id, data_f, iden='0'):
        r_df.to_csv(data_f + dset + '_' + relation + '_' + iden + '.tsv',
                    sep='\t', columns=['com_id', group_id], index=None,
                    header=None)
        g_df.to_csv(data_f + dset + '_' + group + '_' + iden + '.tsv',
                    sep='\t', columns=[group_id], index=None, header=None)

    def _write_tuffy_predicates(self, dset, r_df, relation, group_id, data_f):
        rel = relation.capitalize()

        with open(data_f + dset + '_evidence.txt', 'a') as ev:
            ev.write('\n')
            for index, row in r_df.iterrows():
                com_id = str(int(row.com_id))
                g_id = str(row[group_id])
                ev.write(rel + '(' + com_id + ', ' + g_id + ')\n')
