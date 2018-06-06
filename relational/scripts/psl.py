"""
This module handles all operations to run the relational model using psl.
"""
import os
import numpy as np
import pandas as pd


class PSL:

    def __init__(self, config_obj, conns_obj, draw_obj, pred_builder_obj,
                 util_obj):
        self.config_obj = config_obj
        self.conns_obj = conns_obj
        self.draw_obj = draw_obj
        self.pred_builder_obj = pred_builder_obj
        self.util_obj = util_obj

    # public
    def clear_data(self, data_f):
        self.util_obj.out('clearing out old data...')
        os.system('rm ' + data_f + '*.tsv')
        os.system('rm ' + data_f + '*.txt')
        os.system('rm ' + data_f + 'db/*.db')

    def clear_preds(self, rel_d):
        self.util_obj.out('clearing out old preds...')
        path = rel_d + 'psl_preds_' + self.config_obj.fold + '.csv'
        os.system('rm -f %s' % path)

    def compile(self, psl_f):
        t1 = self.util_obj.out('compiling reasoning engine...')

        mvn_compile = 'mvn compile -q'
        mvn_build = 'mvn dependency:build-classpath '
        mvn_build += '-Dmdep.outputFile=classpath.out -q'

        self.util_obj.pushd(psl_f)
        os.system(mvn_compile)
        os.system(mvn_build)
        self.util_obj.popd()
        self.util_obj.time(t1)

    def infer(self, df, psl_d, psl_f, rel_d, max_size=500000):
        fold = self.config_obj.fold
        relations = self.config_obj.relations

        # df['ind_pred'] = 1 - df['ind_pred']  # TEMP

        g, ccs = self.conns_obj.find_subgraphs(df, relations, max_size)
        subgraphs = self.conns_obj.consolidate(ccs, max_size)

        for i, (ids, hubs, rels, edges) in enumerate(subgraphs):
            _id = i + int(fold)
            sg_df = df[df['com_id'].isin(ids)]
            self._gen_predicates(sg_df, 'test', psl_d, _id)
            self._network_size(psl_d, _id, dset='test')

            t1 = self.util_obj.out('reasoning over sg_%d...' % i)
            self._run(psl_f, _id, action='Infer')
            self.util_obj.time(t1)
        self._combine_predictions(len(subgraphs), rel_d)

        # stats_df = self._collect_connected_components_stats(ccs, df, rel_d)

        # if self.config_obj.has_display:
        #     preds_df = pd.read_csv(rel_d + 'psl_preds_' + fold + '.csv')
        #     new_df = df.merge(preds_df, how='left')
        #     self.draw_obj.draw_graphs(new_df, g, ccs, relations,
        #                              dir='graphs/', col='psl_pred')

    def train(self, df, psl_d, psl_f):
        # df['ind_pred'] = 1 - df['ind_pred']  # TEMP

        self._gen_predicates(df, 'val', psl_d)
        self._gen_model(psl_d)
        self._network_size(psl_d, dset='val')

        t1 = self.util_obj.out('training...')
        self._run(psl_f, action='Train')
        self.util_obj.time(t1)

    # private
    def _combine_predictions(self, num_subgraphs, rel_d):
        fold = self.config_obj.fold
        dfs = []

        for i in range(num_subgraphs):
            s_id = str(i + int(fold))
            df = pd.read_csv(rel_d + 'psl_preds_' + s_id + '.csv')
            dfs.append(df)
        df = pd.concat(dfs)

        # df['psl_pred'] = 1 - df['psl_pred']  # TEMP

        df.to_csv(rel_d + 'psl_preds_' + fold + '.csv', index=None)

    def _gen_model(self, data_f):
        rules = []

        rules.extend(self._priors())
        for relation, group, group_id in self.config_obj.relations:
            rules.extend(self._map_relation_to_rules(relation, group))
        self._write_model(rules, data_f)

    def _gen_predicates(self, df, dset, rel_data_f, iden=None):
        r_dfs = []
        s_iden = self.config_obj.fold if iden is None else str(iden)

        self.pred_builder_obj.build_comments(df, dset, rel_data_f, iden=s_iden)
        for relation, group, group_id in self.config_obj.relations:
            r_df = self.pred_builder_obj.build_relations(relation, group,
                                                         group_id,
                                                         df, dset, rel_data_f,
                                                         iden=s_iden)
            r_dfs.append(r_df)
        return r_dfs

    def _map_relation_to_rules(self, relation, group, wgt=1.0, sq=True):
        rule1 = str(wgt) + ': '
        rule2 = str(wgt) + ': '

        atom1 = relation + '(Com, ' + group.capitalize() + ')'
        atom2 = 'spmy' + group + '(' + group.capitalize() + ')'
        atom3 = 'spam(Com)'

        rule1 += atom1 + ' & ' + atom2 + ' -> ' + atom3
        rule2 += atom1 + ' & ' + atom3 + ' -> ' + atom2

        if sq:
            rule1 += ' ^2'
            rule2 += ' ^2'
        return [rule1, rule2]

    def _network_size(self, data_f, iden=None, dset='val'):
        s_iden = self.config_obj.fold if iden is None else str(iden)
        relations = self.config_obj.relations
        all_nodes, all_edges = 0, 0

        self.util_obj.out('%s network:' % dset)
        fn_m = data_f + dset + '_' + s_iden + '.tsv'
        msg_nodes = self.util_obj.file_len(fn_m)
        self.util_obj.out('-> msg nodes: %d' % msg_nodes)
        all_nodes += msg_nodes

        for relation, group, group_id in relations:
            fn_r = data_f + dset + '_' + relation + '_' + s_iden + '.tsv'
            fn_g = data_f + dset + '_' + group + '_' + s_iden + '.tsv'
            edges = self.util_obj.file_len(fn_r)
            hubs = self.util_obj.file_len(fn_g)
            t = (relation, hubs, edges)
            self.util_obj.out('-> %s nodes: %d, edges: %d' % t)

            all_edges += edges
            all_nodes += hubs

        t = (all_nodes, all_edges)
        self.util_obj.out('-> all nodes: %d, all edges: %d' % t)
        return all_edges

    def _run(self, psl_f, iden=None, action='Infer'):
        s_iden = self.config_obj.fold if iden is None else str(iden)
        fold = self.config_obj.fold
        domain = self.config_obj.domain

        arg_list = [fold, s_iden, domain]
        execute = 'java -Xmx60g -cp ./target/classes:`cat classpath.out` '
        execute += 'spam.' + action + ' ' + ' '.join(arg_list)

        self.util_obj.pushd(psl_f)
        os.system(execute)
        self.util_obj.popd()

    def _priors(self, wgt=1.0, sq=True):
        neg_prior = str(wgt) + ': ~spam(Com)'
        pos_prior = str(wgt) + ': indpred(Com) -> spam(Com)'

        if sq:
            neg_prior += ' ^2'
            pos_prior += ' ^2'
        return [neg_prior, pos_prior]

    def _write_model(self, rules, data_f):
        fold = self.config_obj.fold

        with open(data_f + 'rules_' + fold + '.txt', 'w') as w:
            for rule in rules:
                w.write(rule + '\n')

    def _collect_connected_components_stats(self, ccs, df, rel_d):
        fold = self.config_obj.fold
        t1 = self.util_obj.out('collecting connected components stats...')

        df_cols = ['size', 'same', 'mean', 'median', 'std', 'max', 'min']
        df_rows = []
        ccs = [x for x in ccs if x[3] > 0]  # filter out no edge subgraphs

        for msg_nodes, hub_nodes, relations, edges in ccs:
            qf = df[df['com_id'].isin(msg_nodes)]
            ip = qf['ind_pred']

            size = len(msg_nodes)
            mean = np.mean(ip)
            median = np.median(ip)
            same = 1 if np.allclose(ip, ip[::-1], atol=1e-4) \
                and np.isclose(mean, median, atol=1e-8) else 0
            std = np.std(ip)
            mx = np.max(ip)
            mn = np.min(ip)
            row = [size, same, mean, median, std, mx, mn]

            label_col = 'label' if 'label' in list(qf) else \
                'is_attributed' if 'is_attributed' in list(qf) else None

            if label_col is not None:
                il = qf['label']
                lab_mean = np.mean(il)
                lab_diff = np.mean(np.abs(np.subtract(ip, il)))
                row.append(lab_mean)
                row.append(lab_diff)

            df_rows.append(row)
        self.util_obj.time(t1)

        if len(df_rows[0]) > 7:
            df_cols += ['lab_mean', 'lab_diff']

        sg_dir = rel_d + '../subgraphs/'
        self.util_obj.create_dirs(sg_dir)
        fname = sg_dir + 'sg_stats_%s.csv' % fold

        if os.path.exists(fname):
            old_df = pd.read_csv(fname)
            new_df = pd.DataFrame(df_rows, columns=df_cols)
            df = pd.concat([old_df, new_df])
        else:
            df = pd.DataFrame(df_rows, columns=df_cols)

        df.sort_values('size').to_csv(fname, index=None)
        return df
