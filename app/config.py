"""
Class that maintains the state of the app.
"""


class Config:

    def __init__(self):
        self.app_dir = None  # absolute path to app package.
        self.ind_dir = None  # absolute path to independent package.
        self.rel_dir = None  # absolute path to relational package.
        self.ana_dir = None  # absolute path to analysis package.
        self.fold = None  # experiment identifier.
        self.domain = None  # domain to model.
        self.start = None   # line number to start reading data.
        self.end = None   # line number to read data until.
        self.train_size = None  # amount of data to train independent mdoels.
        self.val_size = None  # amount of data to train relational models.
        self.featuresets = ['all']  # features to use for classification.
        self.classifier = 'lr'  # independent classifier.
        self.relations = None  # relations to exploit.
        self.has_display = False  # has display if True, otherwise does not.
        self.alter_user_ids = False  # make all user ids in test set unique.
        self.super_train = True  # use train and val for training.
        self.evaluation = 'cc'  # cross-compare (cc) or train-test (tt).
        self.param_search = 'single'  # amount of hyper-parameters to search.
        self.tune_size = 0.15  # percentage of training data for tuning.
        self.engine = 'all'  # reasoning engine for collective classification.
        self.stacking = 0  # rounds to compute pseudo-relatonal features.
        self.data = 'both'  # controls which type of data to use.
        self.separate_relations = False  # disjoin training and test sets.
        self.exact = True  # exact matches in relations.
        self.stack_splits = []  # stack sizes, len must be equal to stacking.
        self.epsilons = {}  # epsilon values per relation.
        self.analyze_subgraphs = True  # analyze subgraphs in the test set.

    # public
    def set_display(self, has_display):
        """If True, then application is run on a console.
        has_display: boolean indiciating if on a console."""
        self.has_display = has_display

    def set_directories(self, app_dir, ind_dir, rel_dir, ana_dir):
        """Sets absolute path directories in the config object.
        condig_dir: absolute path to the config package.
        ind_dir: absolute path to the independent package.
        rel_dir: absolute path to the relational package.
        ana_dir: absolute path to the analysis package."""
        self.app_dir = app_dir
        self.ind_dir = ind_dir
        self.rel_dir = rel_dir
        self.ana_dir = ana_dir

    def set_options(self, domain='twitter', start=0, end=1000,
                    train_size=0.7, val_size=0.15, ngrams=True, clf='lr',
                    engine='all', fold=0, relations=['intext'], stacking=0,
                    separate_relations=False, data='both',
                    alter_user_ids=False, super_train=False, modified=False,
                    evaluation='cc', param_search='single', tune_size=0.15,
                    featuresets='all', approx=False, stack_splits=[],
                    epsilons=[], analyze_subgraphs=True):

        # validate args
        assert isinstance(ngrams, bool)
        assert isinstance(separate_relations, bool)
        assert isinstance(alter_user_ids, bool)
        assert isinstance(super_train, bool)
        assert isinstance(modified, bool)
        assert isinstance(approx, bool)
        assert isinstance(analyze_subgraphs, bool)
        assert stacking >= 0
        assert evaluation in ['cc', 'tt']
        assert param_search in ['single', 'low', 'med', 'high']
        assert tune_size >= 0
        assert engine in self._available_engines()
        assert domain in self._available_domains()
        assert data in ['ind', 'rel', 'both']
        assert train_size + val_size < 1.0 if eval == 'cc' else True
        assert train_size + val_size == 1.0 if eval == 'tt' else True
        assert start < end
        assert clf in ['lr', 'rf', 'xgb', 'lgb']
        assert set(relations).issubset(self._available_relations()[domain])
        assert len(stack_splits) == stacking if len(stack_splits) > 0 else True
        assert len(epsilons) == len(relations) if len(epsilons) > 0 else True
        assert val_size > 0 if engine in ['psl', 'all'] else True
        for fset in featuresets:
            assert fset in self._available_featuresets()

        stack_splits = [float(split) for split in stack_splits]
        epsilons = [float(epsilon) for epsilon in epsilons]

        d = {'domain': domain, 'start': start, 'end': end,
             'train_size': train_size, 'val_size': val_size, 'ngrams': ngrams,
             'classifier': clf, 'engine': engine, 'fold': fold,
             'relations': relations, 'separate_relations': separate_relations,
             'data': data, 'alter_user_ids': alter_user_ids,
             'super_train': super_train, 'modified': modified,
             'stacking': stacking, 'evaluation': evaluation,
             'param_search': param_search, 'tune_size': tune_size,
             'featuresets': featuresets, 'approx': approx,
             'stack_splits': stack_splits, 'epsilons': epsilons,
             'analyze_subgraphs': analyze_subgraphs}

        self._populate_config(d)
        print(self)
        return self

    # private
    def _available_domains(self):
        return ['soundcloud', 'youtube', 'twitter', 'toxic', 'twitter2',
                'ifwe', 'yelp_hotel', 'yelp_restaurant', 'adclicks', 'russia']

    def _available_featuresets(self):
        return ['graph', 'ngrams', 'content', 'sequential', 'pseudo',
                'aggregate', 'all']

    def _available_relations(self):
        relations = {}
        relations['soundcloud'] = ['haspost', 'hastext', 'hastrack', 'hashash',
                                   'hasment', 'haslink', 'hasusrtrack']
        relations['youtube'] = ['haspost', 'hastext', 'hasment', 'hasvideo',
                                'hashash']
        relations['twitter'] = ['haspost', 'hastext', 'hashash', 'hasment',
                                'haslink', 'hasusrhash', 'hasusrment',
                                'hasusrlink', 'hasusrtext', 'hasusrrt',
                                'hasusrhashment', 'hashashment',
                                'hashashmentlink', 'hasrthash',
                                'hasusrrthash', 'hasunicode', 'hasunicodecnt']
        relations['twitter2'] = ['haspost', 'hastext', 'hashash', 'hasment',
                                 'haslink', 'hasusrhash', 'hasusrment',
                                 'hasusrlink', 'hasusrtext', 'hasusrrt',
                                 'hasusrhashment', 'hashashment',
                                 'hashashmentlink', 'hasrthash',
                                 'hasusrrthash']
        relations['russia'] = ['haspost', 'hastext', 'hashash', 'hasment',
                               'haslink']
        relations['toxic'] = ['hastext', 'haslink']
        relations['ifwe'] = ['hasr0', 'hasr1', 'hasr2', 'hasr3', 'hasr4',
                             'hasr5', 'hasr6', 'hasr7', 'hassex', 'hasage',
                             'hastimepassed']
        relations['yelp_hotel'] = ['haspost', 'hastext', 'hashotel']
        relations['yelp_restaurant'] = ['haspost', 'hastext', 'hasrest']
        relations['adclicks'] = ['hasip', 'haschannel', 'hasapp', 'hasos',
                                 'hasdevice', 'hasusrapp', 'hasusrad',
                                 'hasusrhour', 'hasusrmin', 'hasusrsec']
        return relations

    def _available_groups(self):
        groups = {'haspost': 'post', 'hastext': 'text', 'hastrack': 'track',
                  'hashash': 'hash', 'hasment': 'ment', 'hasvideo': 'video',
                  'hashour': 'hour', 'haslink': 'link', 'hashotel': 'hotel',
                  'hasrest': 'rest', 'hasr0': 'r0', 'hasr1': 'r1',
                  'hasr2': 'r2', 'hasr3': 'r3', 'hasr4': 'r4', 'hasr5': 'r5',
                  'hasr6': 'r6',
                  'hasr7': 'r7', 'hassex': 'sex', 'hasage': 'age',
                  'hastimepassed': 'timepassed',
                  'hasip': 'ip', 'haschannel': 'channel', 'hasapp': 'app',
                  'hasos': 'os', 'hasdevice': 'device',
                  'hasusrapp': 'usrapp', 'hasusrad': 'usrad',
                  'hasusrhash': 'usrhash', 'hasusrment': 'usrment',
                  'hasusrlink': 'usrlink', 'hasusrtext': 'usrtext',
                  'hasusrhour': 'usrhour', 'hasusrmin': 'usrmin',
                  'hasusrsec': 'usrsec', 'hasusrrt': 'usrrt',
                  'hasusrhashment': 'usrhashment', 'hashashment': 'hashment',
                  'hashashmentlink': 'hashmentlink', 'hasrthash': 'rthash',
                  'hasusrrthash': 'usrrthash', 'hasunicode': 'unicode',
                  'hasunicodecnt': 'unicodecnt', 'hasusrtrack': 'usrtrack'}
        return groups

    def _available_ids(self):
        ids = {'haspost': 'post_gid', 'hastext': 'text_gid',
               'hastrack': 'track_gid',
               'hashash': 'hash_gid', 'hasment': 'ment_gid',
               'hasvideo': 'video_gid', 'hashour': 'hour_gid',
               'haslink': 'link_gid',
               'hashotel': 'hotel_gid', 'hasrest': 'rest_gid',
               'hasr0': 'r0_gid',
               'hasr1': 'r1_gid', 'hasr2': 'r2_gid', 'hasr3': 'r3_gid',
               'hasr4': 'r4_gid', 'hasr5': 'r5_gid', 'hasr6': 'r6_gid',
               'hasr7': 'r7_gid', 'hassex': 'sex_gid', 'hasage': 'age_gid',
               'hastimepassed': 'timepassed_gid', 'hasip': 'ip_gid',
               'haschannel': 'channel_gid', 'hasapp': 'app_gid',
               'hasos': 'os_gid', 'hasdevice': 'device_gid',
               'hasusrapp': 'usrapp_gid', 'hasusrad': 'usrad_gid',
               'hasusrhash': 'usrhash_gid', 'hasusrment': 'usrment_gid',
               'hasusrlink': 'usrlink_gid', 'hasusrtext': 'usrtext_gid',
               'hasusrhour': 'usrhour_gid', 'hasusrmin': 'usrmin_gid',
               'hasusrsec': 'usrsec_gid', 'hasusrrt': 'usrrt_gid',
               'hasusrhashment': 'usrhashment_gid',
               'hashashment': 'hashment_gid', 'hasusrtrack': 'usrtrack_gid',
               'hashashmentlink': 'hashmentlink_gid',
               'hasrthash': 'rthash_gid', 'hasusrrthash': 'usrrthash_gid',
               'hasunicode': 'unicode_gid', 'hasunicodecnt': 'unicodecnt_gid'}
        return ids

    def _groups_for_relations(self, relations):
        available_groups = self._available_groups()
        groups = [available_groups[relation] for relation in relations]
        return groups

    def _ids_for_relations(self, relations):
        available_ids = self._available_ids()
        ids = [available_ids[relation] for relation in relations]
        return ids

    def _available_engines(self):
        return ['psl', 'tuffy', 'mrf', 'all', None]

    def _populate_config(self, config):
        relations = config['relations']
        epl = config['epsilons']

        groups = self._groups_for_relations(relations)
        ids = self._ids_for_relations(relations)

        self.relations = list(zip(relations, groups, ids))
        self.domain = str(config['domain'])
        self.start = int(config['start'])
        self.end = int(config['end'])
        self.train_size = float(config['train_size'])
        self.val_size = float(config['val_size'])
        self.classifier = str(config['classifier'])
        self.fold = str(config['fold'])
        self.engine = str(config['engine'])
        self.stacking = int(config['stacking'])
        self.evaluation = str(config['evaluation'])
        self.param_search = str(config['param_search'])
        self.tune_size = float(config['tune_size'])
        self.featuresets = config['featuresets']
        self.data = str(config['data'])
        self.exact = bool(not config['approx'])
        self.stack_splits = config['stack_splits']
        self.super_train = config['super_train']
        self.epsilons = dict(zip(relations, epl)) if len(epl) > 0 else {}
        self.analyze_subgraphs = config['analyze_subgraphs']

    def __str__(self):
        relations = [r[0] for r in self.relations]

        s = '\nDomain: ' + str(self.domain) + '\n'
        s += 'Data range: ' + str(self.start) + ' to ' + str(self.end) + '\n'
        s += 'Training size: ' + str(self.train_size) + '\n'
        s += 'Validation size: ' + str(self.val_size) + '\n'
        s += 'Independent classifier: ' + str(self.classifier) + '\n'
        s += 'Stacks: ' + str(self.stacking) + '\n'
        s += 'Fold: ' + str(self.fold) + '\n'
        s += 'Relations: ' + str(relations) + '\n'
        s += 'Engine: ' + str(self.engine) + '\n'
        s += 'Evaluation: ' + str(self.evaluation) + '\n'
        s += 'Param search: ' + str(self.param_search) + '\n'
        s += 'Tuning size: ' + str(self.tune_size) + '\n'
        s += 'Featuresets: ' + str(self.featuresets) + '\n'
        s += 'Data: ' + str(self.data) + '\n'
        s += 'Exact matches: ' + str(self.exact) + '\n'
        s += 'Stack splits: ' + str(self.stack_splits) + '\n'
        s += 'Super train: ' + str(self.super_train) + '\n'
        s += 'Epsilons: ' + str(self.epsilons) + '\n'
        s += 'Analyze subgraphs: ' + str(self.analyze_subgraphs)
        return s
