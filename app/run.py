import os
import sys
import argparse
import warnings
import pandas as pd
from app.config import Config
from app.data import Data
from app.app import App
from independent.scripts.independent import Independent
from independent.scripts.classification import Classification
from independent.scripts.features import Features
from relational.scripts.comments import Comments
from relational.scripts.generator import Generator
from relational.scripts.pred_builder import PredicateBuilder
from relational.scripts.psl import PSL
from relational.scripts.relational import Relational
from relational.scripts.tuffy import Tuffy
from relational.scripts.mrf import MRF
from analysis.analysis import Analysis
from analysis.connections import Connections
from analysis.draw import Draw
from analysis.label import Label
from analysis.purity import Purity
from analysis.evaluation import Evaluation
from analysis.interpretability import Interpretability
from analysis.util import Util
from experiments.ablation_exp import Ablation_Experiment
from experiments.learning_exp import Learning_Experiment
from experiments.relations_exp import Relations_Experiment
from experiments.stacking_exp import Stacking_Experiment
from experiments.subsets_exp import Subsets_Experiment
from experiments.ultimate_exp import Ultimate_Experiment


def directories(this_dir):
    app_dir = this_dir + '/app/'
    ind_dir = this_dir + '/independent/'
    rel_dir = this_dir + '/relational/'
    ana_dir = this_dir + '/analysis/'
    return app_dir, ind_dir, rel_dir, ana_dir


def init_dependencies():
    config_obj = Config()
    util_obj = Util()

    draw_obj = Draw(util_obj)
    connections_obj = Connections(util_obj)
    generator_obj = Generator(util_obj)
    data_obj = Data(generator_obj, util_obj)

    features_obj = Features(config_obj, util_obj)
    classify_obj = Classification(config_obj, features_obj, util_obj)
    independent_obj = Independent(config_obj, classify_obj, util_obj)

    comments_obj = Comments(config_obj, util_obj)
    pred_builder_obj = PredicateBuilder(config_obj, comments_obj,
                                        generator_obj, util_obj)
    psl_obj = PSL(config_obj, connections_obj, draw_obj, pred_builder_obj,
                  util_obj)
    tuffy_obj = Tuffy(config_obj, pred_builder_obj, util_obj)
    mrf_obj = MRF(config_obj, connections_obj, draw_obj, generator_obj,
                  util_obj)
    relational_obj = Relational(connections_obj, config_obj, psl_obj,
                                tuffy_obj, mrf_obj, util_obj)

    label_obj = Label(config_obj, generator_obj, util_obj)
    purity_obj = Purity(config_obj, generator_obj, util_obj)
    evaluate_obj = Evaluation(config_obj, generator_obj, connections_obj,
                              util_obj)
    interpret_obj = Interpretability(config_obj, connections_obj,
                                     generator_obj, pred_builder_obj, util_obj)
    analysis_obj = Analysis(config_obj, label_obj, purity_obj, evaluate_obj,
                            interpret_obj, util_obj)

    app_obj = App(config_obj, data_obj, independent_obj, relational_obj,
                  analysis_obj, util_obj)
    return config_obj, app_obj, util_obj


def global_settings(config_obj):
    pd.options.mode.chained_assignment = None
    warnings.filterwarnings('ignore', module='numpy')
    warnings.filterwarnings('ignore', module='sklearn')
    warnings.filterwarnings('ignore', module='scipy')
    warnings.filterwarnings('ignore', module='matplotlib')
    warnings.filterwarnings('ignore', module='networkx')
    if os.isatty(sys.stdin.fileno()):
        rows, columns = os.popen('stty size', 'r').read().split()
        pd.set_option('display.width', int(columns))
        pd.set_option('display.max_columns', 100)
        config_obj.set_display(True)


def add_args():
    description = 'Spam detection for online social networks'
    parser = argparse.ArgumentParser(description=description, prog='run')

    # high level args
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--run', '-r', action='store_true',
                       help='Run detection engine, default: %(default)s')
    group.add_argument('--ablation', action='store_true',
                       help='Run ablation, default: %(default)s')
    group.add_argument('--learning', action='store_true',
                       help='Run learning curves, default: %(default)s')
    group.add_argument('--relations', action='store_true',
                       help='Run relations, default: %(default)s')
    group.add_argument('--stacking', action='store_true',
                       help='Run stacking, default: %(default)s')
    group.add_argument('--subsets', action='store_true',
                       help='Run subsets, default: %(default)s')
    group.add_argument('--ultimate', action='store_true',
                       help='Run ultimate, default: %(default)s')

    # general args that overlap among different APIs
    parser.add_argument('-d', default='twitter', metavar='DOMAIN',
                        help='social network, default: %(default)s')
    parser.add_argument('-s', default=0, metavar='START', type=int,
                        help='data range start, default: %(default)s')
    parser.add_argument('-e', default=1000, metavar='END', type=int,
                        help='data range end, default: %(default)s')
    parser.add_argument('-f', default=0, metavar='FOLD', type=int,
                        help='experiment identifier, default: %(default)s')
    parser.add_argument('--engine', default=None,
                        help='relational framework, default: %(default)s')
    parser.add_argument('--clf', default='lr', metavar='CLF',
                        help='classifier, default: %(default)s')
    parser.add_argument('--stacks', default=0, type=int,
                        help='number of stacks, default: %(default)s')
    parser.add_argument('--stack_splits', nargs='*', metavar='PERCENT',
                        help='size of stacks, default: %(default)s')
    parser.add_argument('--data', default='both',
                        help='rel, ind, or both, default: %(default)s')
    parser.add_argument('--train_size', default=0.8, metavar='PERCENT',
                        type=float, help='train size, default: %(default)s')
    parser.add_argument('--val_size', default=0.1, metavar='PERCENT',
                        type=float, help='val size, default: %(default)s')
    parser.add_argument('--tune_size', default=0.2, metavar='PERCENT',
                        type=float, help='tuning size, default: %(default)s')
    parser.add_argument('--param_search', default='single', metavar='LEVEL',
                        help='parameter search, default: %(default)s')
    parser.add_argument('--no_sep_rels', action='store_true',
                        help='do not break relations, default: %(default)s')
    parser.add_argument('--eval', default='cc', metavar='SCHEMA',
                        help='type of testing, default: %(default)s')
    parser.add_argument('--rels', nargs='*', metavar='REL',
                        help='relations to exploit, default: %(default)s')
    parser.add_argument('--sim_dir', default=None, metavar='DIR',
                        help='similarities directory, default: %(default)s')
    parser.add_argument('--approx', action='store_true',
                        help='use approx similarity, default: %(default)s')
    parser.add_argument('--super_train', action='store_true',
                        help='train includes val data, default: %(default)s')
    parser.add_argument('--epsilons', nargs='*', metavar='EP',
                        help='epsilons per relation, default: %(default)s')
    parser.add_argument('--no_analyze_subgraphs', action='store_true',
                        help='do not analyze subgraphs, default: %(default)s')

    # experiment specific args
    parser.add_argument('--train_start', default=0, metavar='NUM', type=int,
                        help='start of training data, default: %(default)s')
    parser.add_argument('--train_end', default=100, metavar='NUM', type=int,
                        help='end of training data, default: %(default)s')
    parser.add_argument('--test_start', default=200, metavar='NUM', type=int,
                        help='start of testing data, default: %(default)s')
    parser.add_argument('--test_end', default=400, metavar='NUM', type=int,
                        help='end of testing data, default: %(default)s')
    parser.add_argument('--learn_sizes', nargs='*', metavar='SIZE',
                        help='list of learning sizes, default: %(default)s')
    parser.add_argument('--feat_sets', nargs='*', metavar='FEATS',
                        help='list of featuresets, default: %(default)s')
    parser.add_argument('--clfs', nargs='*', metavar='CLF',
                        help='list of classifiers, default: %(default)s')
    parser.add_argument('--start_stack', default=0, metavar='NUM', type=int,
                        help='beginning stack number, default: %(default)s')
    parser.add_argument('--end_stack', default=4, metavar='NUM', type=int,
                        help='ending stack number, default: %(default)s')
    parser.add_argument('--metric', default='aupr', metavar='METRIC',
                        help='performance measurement, default: %(default)s')
    parser.add_argument('--num_sets', default=100, metavar='SUBSETS', type=int,
                        help='number of subsets, default: %(default)s')
    parser.add_argument('--sub_size', default=-1, metavar='SIZE', type=int,
                        help='subset size, default: %(default)s')
    parser.add_argument('--sim_dirs', nargs='*', metavar='DIR',
                        help='list of similarity dirs, default: %(default)s')
    parser.add_argument('--start_on', default=0, metavar='NUM', type=int,
                        help='subset to start on, default: %(default)s')
    parser.add_argument('--train_pts', default=-1, metavar='NUM', type=int,
                        help='points to train on, default: %(default)s')
    parser.add_argument('--test_pts', default=-1, metavar='NUM', type=int,
                        help='points to test on, default: %(default)s')
    return parser


def parse_args(parser):
    p = {}
    a = parser.parse_args()

    p['domain'] = a.d
    p['start'] = a.s
    p['end'] = a.e
    p['engine'] = a.engine
    p['fold'] = a.f
    p['clf'] = a.clf
    p['stacks'] = a.stacks
    p['stack_splits'] = a.stack_splits if a.stack_splits is not None else []
    p['data'] = a.data
    p['train_size'] = a.train_size
    p['val_size'] = a.val_size
    p['tune_size'] = a.tune_size
    p['param_search'] = a.param_search
    p['separate_relations'] = True if a.no_sep_rels is False else False
    p['eval'] = a.eval
    p['relations'] = a.rels if a.rels is not None else []
    p['sim_dir'] = a.sim_dir
    p['learn_sizes'] = a.learn_sizes if a.learn_sizes is not None else []
    p['feat_sets'] = a.feat_sets if a.feat_sets is not None else ['all']
    p['clfs'] = a.clfs if a.clfs is not None else ['lr']
    p['start_stack'] = a.start_stack
    p['end_stack'] = a.end_stack
    p['metric'] = a.metric
    p['subsets'] = a.num_sets
    p['subset_size'] = a.sub_size
    p['sim_dirs'] = [None] + a.sim_dirs if a.sim_dirs is not None else []
    p['approx'] = a.approx
    p['super_train'] = a.super_train
    p['epsilons'] = a.epsilons if a.epsilons is not None else []
    p['start_on'] = a.start_on
    p['train_start'] = a.train_start
    p['train_end'] = a.train_end
    p['test_start'] = a.test_start
    p['test_end'] = a.test_end
    p['train_pts'] = a.train_pts
    p['test_pts'] = a.test_pts
    p['analyze_subgraphs'] = not a.no_analyze_subgraphs

    return a, p


def main():
    parser = add_args()
    args, p = parse_args(parser)

    this_dir = os.path.abspath(os.getcwd())
    app_dir, ind_dir, rel_dir, ana_dir = directories(this_dir)
    config_obj, app_obj, util_obj = init_dependencies()

    global_settings(config_obj)
    config_obj.set_directories(app_dir, ind_dir, rel_dir, ana_dir)

    if args.run:
        app_obj.run(domain=p['domain'], start=p['start'], end=p['end'],
                    engine=p['engine'], clf=p['clf'],
                    stacking=p['stacks'], data=p['data'],
                    train_size=p['train_size'], val_size=p['val_size'],
                    relations=p['relations'], sim_dir=p['sim_dir'],
                    separate_relations=p['separate_relations'],
                    evaluation=p['eval'], param_search=p['param_search'],
                    tune_size=p['tune_size'], fold=p['fold'],
                    featuresets=p['feat_sets'], approx=p['approx'],
                    stack_splits=p['stack_splits'], epsilons=p['epsilons'])

    elif args.ablation:
        le = Ablation_Experiment(config_obj, app_obj, util_obj)
        le.run_experiment(start=p['start'], end=p['end'], domain=p['domain'],
                          featuresets=p['feat_sets'], fold=p['fold'],
                          clf=p['clf'], train_size=p['train_size'],
                          relations=p['relations'],
                          analyze_subgraphs=p['analyze_subgraphs'],
                          param_search=p['param_search'])

    elif args.learning:
        le = Learning_Experiment(config_obj, app_obj, util_obj)
        le.run_experiment(test_start=p['test_start'], test_end=p['test_end'],
                          train_start=p['train_start'],
                          train_end=p['train_end'], engine=p['engine'],
                          learn_sizes=p['learn_sizes'], domain=p['domain'],
                          fold=p['fold'], clf=p['clf'], sim_dir=p['sim_dir'],
                          super_train=p['super_train'],
                          relations=p['relations'])

    elif args.relations:
        le = Relations_Experiment(config_obj, app_obj, util_obj)
        le.run_experiment(start=p['start'], end=p['end'], domain=p['domain'],
                          relations=p['relations'], fold=p['fold'],
                          clf=p['clf'], train_size=p['train_size'],
                          val_size=p['val_size'], engine=p['engine'],
                          subsets=p['subsets'], subset_size=p['subset_size'],
                          sim_dirs=p['sim_dirs'])

    elif args.stacking:
        se = Stacking_Experiment(config_obj, app_obj, util_obj)
        se.run_experiment(domain=p['domain'], start=p['start'], end=p['end'],
                          clf=p['clf'], train_size=p['train_size'],
                          start_stack=p['start_stack'], fold=p['fold'],
                          end_stack=p['end_stack'], relations=p['relations'],
                          subsets=p['subsets'], subset_size=p['subset_size'],
                          sim_dir=p['sim_dir'])

    elif args.subsets:
        se = Subsets_Experiment(config_obj, app_obj, util_obj)
        se.run_experiment(domain=p['domain'], start=p['start'], end=p['end'],
                          subsets=p['subsets'], data=p['data'], fold=p['fold'],
                          engine=p['engine'], train_size=p['train_size'],
                          val_size=p['val_size'], relations=p['relations'],
                          clf=p['clf'], featuresets=p['feat_sets'],
                          stacking=p['stacks'], sim_dir=p['sim_dir'],
                          param_search=p['param_search'],
                          subset_size=p['subset_size'],
                          train_pts=p['train_pts'], test_pts=p['test_pts'],
                          start_on=p['start_on'], epsilons=p['epsilons'])

    elif args.ultimate:
        ue = Ultimate_Experiment(config_obj, app_obj, util_obj)
        ue.run_experiment(domain=p['domain'], start=p['start'], end=p['end'],
                          clf=p['clf'], train_size=p['train_size'],
                          start_stack=p['start_stack'], fold=p['fold'],
                          end_stack=p['end_stack'],
                          relationsets=p['relations'], engine=p['engine'],
                          val_size=p['val_size'], subsets=p['subsets'],
                          param_search=p['param_search'],
                          tune_size=p['tune_size'], sim_dir=p['sim_dir'],
                          subset_size=p['subset_size'])
