"""
Module with high a high level API to run any part of the app.
"""


class App:

    # public
    def __init__(self, config_obj, data_obj, independent_obj, relational_obj,
                 analysis_obj, util_obj):
        self.config_obj = config_obj
        self.data_obj = data_obj
        self.independent_obj = independent_obj
        self.relational_obj = relational_obj
        self.analysis_obj = analysis_obj
        self.util_obj = util_obj

    def run(self, modified=False, stacking=0, engine='all',
            start=0, end=1000, fold=0, data='both', sim_dir=None,
            clf='lr', alter_user_ids=False, super_train=True,
            domain='twitter', separate_relations=True, train_size=0.7,
            val_size=0.15, relations=['intext'], evaluation='cc',
            param_search='single', tune_size=0.15, featuresets=['all'],
            approx=False, stack_splits=[], val_split=0.0, epsilons=[],
            analyze_subgraphs=False):

        # validate args
        self.config_obj.set_options(domain=domain, start=start, end=end,
                                    train_size=train_size, val_size=val_size,
                                    clf=clf, engine=engine,
                                    fold=fold, relations=relations,
                                    stacking=stacking, data=data,
                                    separate_relations=separate_relations,
                                    alter_user_ids=alter_user_ids,
                                    super_train=super_train, modified=modified,
                                    evaluation=evaluation, tune_size=tune_size,
                                    param_search=param_search,
                                    featuresets=featuresets, approx=approx,
                                    stack_splits=stack_splits,
                                    epsilons=epsilons,
                                    analyze_subgraphs=analyze_subgraphs)

        relations = self.config_obj.relations
        exact = not approx

        # get data
        if evaluation == 'cc':
            coms_df = self.data_obj.get_data(domain=domain, start=start,
                                             end=end, evaluation=evaluation)
            dfs = self.data_obj.split_data(coms_df, train_size=train_size,
                                           val_size=val_size,
                                           val_split=val_split)
            dfs = self.data_obj.get_rel_ids(dfs, domain, relations,
                                            sim_dir=sim_dir, exact=exact)
            # coms_df = self.data_obj.sep_data(coms_df, relations=relations,
            #                                  domain=domain, data=data)
        elif evaluation == 'tt':
            train_df, test_df = self.data_obj.get_data(domain=domain,
                                                       start=start, end=end,
                                                       evaluation=evaluation)
            train_df = self.data_obj.get_rel_ids(train_df, domain, relations,
                                                 sim_dir=sim_dir, exact=exact)
            test_df = self.data_obj.get_rel_ids(test_df, domain, relations,
                                                sim_dir=sim_dir, exact=exact)
            dfs = self.data_obj.split_data(train_df, train_size=train_size,
                                           val_size=val_size,
                                           val_split=val_split)
            dfs['test'] = test_df

        d = self._run_models(dfs, stacking=stacking, engine=engine, data=data,
                             evaluation=evaluation)
        return d

    # private
    def _run_models(self, dfs, stacking=0, engine='all', data='both',
                    evaluation='cc'):
        score_dict = None

        self._print_datasets(dfs)

        self.util_obj.out('\nINDEPENDENT...')
        val_df, test_df = self.independent_obj.main(dfs)

        if data in ['rel', 'both'] and engine in ['psl', 'all']:
            self.util_obj.out('\nPSL...')
            self.relational_obj.compile_reasoning_engine()
            self.relational_obj.main(val_df, test_df, engine='psl')

        if data in ['rel', 'both'] and engine in ['mrf', 'all']:
            self.util_obj.out('\nMRF...')
            self.relational_obj.main(val_df, test_df, engine='mrf')

        if evaluation == 'cc':
            score_dict = self.analysis_obj.evaluate(test_df)
            self.util_obj.out()
            for model, scores in score_dict.items():
                self.util_obj.out('%s: %s' % (model, str(scores)))
            self.util_obj.out()

        return score_dict

    def _print_datasets(self, data):
        train_df, val_df, test_df = data['train'], data['val'], data['test']

        spam, total = len(train_df[train_df['label'] == 1]), len(train_df)
        percentage = round(self.util_obj.div0(spam, total) * 100, 1)
        s = '\ntrain size: ' + str(len(train_df)) + ', '
        s += 'spam: ' + str(spam) + ' (' + str(percentage) + '%)'
        self.util_obj.out(s)

        if val_df is not None:
            spam, total = len(val_df[val_df['label'] == 1]), len(val_df)
            percentage = round(self.util_obj.div0(spam, total) * 100, 1)
            s = 'val size: ' + str(len(val_df)) + ', '
            s += 'spam: ' + str(spam) + ' (' + str(percentage) + '%)'
            self.util_obj.out(s)

        total = len(test_df)
        s = 'test size: ' + str(len(test_df))
        if 'label' in list(test_df):
            spam = len(test_df[test_df['label'] == 1])
            percentage = round(self.util_obj.div0(spam, total) * 100, 1)
            s += ', spam: ' + str(spam) + ' (' + str(percentage) + '%)'
        self.util_obj.out(s)
