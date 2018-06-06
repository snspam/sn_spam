import os
import argparse
import util as ut
import numpy as np
import random as ran
import pandas as pd
from generator import Generator
from connections import Connections
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from scipy.stats import ttest_rel


def compute_big_aupr(start_fold=0, ref_start_fold=-1, num_folds=5,
                     domain='twitter', models=['ind'], in_dir='', gids=[]):
    ind_data_dir = 'independent/data/' + domain + '/'

    lines = {'ind': 'b-', 'mrf': 'g--', 'psl': 'm-.', 'mean': 'r:',
             'median': 'c:', 'max': 'y:'}
    inds, mrfs, psls, approxs, refs = [], [], [], [], []
    preds = []

    gen_obj = Generator()
    relations = _relations_for_gids(gids)

    for model in models:
        preds.append(model + '_pred')
    if 'approx' in models:
        models.remove('approx')
        models.extend(['mean', 'median', 'max'])
        preds.extend(['mean_pred', 'median_pred', 'max_pred'])
    preds = list(zip(models, preds))

    t1 = ut.out('reading true labels...', 0)
    full_df = pd.read_csv(ind_data_dir + 'comments.csv')
    lbl_df = full_df[['com_id', 'label']]
    ut.time(t1)

    s = '%s: reading model preds from fold %d to %d:'
    ut.out(s % (domain, start_fold, start_fold + num_folds - 1), 1)

    newline = 1 if 'approx' in models else 0

    d = {}
    for i, fold in enumerate(range(start_fold, start_fold + num_folds)):
        ut.out('\nreading preds for fold %d...' % i, newline)
        f_dict = {}

        if ref_start_fold > -1:
            ndx = ref_start_fold + i
            fname = in_dir + 'test_' + str(ndx) + '_preds.csv'
            assert os.path.exists(fname)
            refs.append(pd.read_csv(fname))

        if 'ind' in models:
            fname = in_dir + 'test_' + str(fold) + '_preds.csv'
            assert os.path.exists(fname)
            ind_df = pd.read_csv(fname)
            inds.append(ind_df)
            ind_lbl_df = full_df.merge(ind_df, on='com_id')
            t1 = ut.out('generating group ids...')
            for gid in gids:
                ind_lbl_df = gen_obj.gen_group_id(ind_lbl_df, gid)
            ut.time(t1)
            m_dict = _metrics(ind_lbl_df)
            a_dict = _analyze(ind_lbl_df, relations=relations, col='ind_pred')
            s_dict = _spread(ind_lbl_df, col='ind_pred', relations=relations)
            f_dict.update(a_dict)
            f_dict.update(s_dict)
            f_dict.update(m_dict)

            if 'mean' in models:
                temp_df = full_df.merge(ind_df)

                t1 = ut.out('generating group ids...')
                for gid in gids:
                    temp_df = gen_obj.gen_group_id(temp_df, gid)
                ut.time(t1)

                approx_df = _approximations(temp_df, relations)
                approxs.append(approx_df)

        if 'mrf' in models:
            fname = in_dir + 'mrf_preds_' + str(fold) + '.csv'
            assert os.path.exists(fname)
            mrf_df = pd.read_csv(fname)
            mrfs.append(mrf_df)
            mrf_lbl_df = lbl_df.merge(mrf_df)
            m_dict = _metrics(mrf_lbl_df, col='mrf_pred', model='mrf')
            f_dict.update(m_dict)

        if 'psl' in models:
            fname = in_dir + 'psl_preds_' + str(fold) + '.csv'
            assert os.path.exists(fname)
            psl_df = pd.read_csv(fname)
            psls.append(psl_df)
            psl_lbl_df = lbl_df.merge(psl_df)
            m_dict = _metrics(psl_lbl_df, col='psl_pred', model='psl')
            f_dict.update(m_dict)

        d[i] = f_dict
        print(d)

    dicts = [d[i] for i in range(len(d))]
    stats_df = pd.DataFrame(dicts)
    stats_df = stats_df.reset_index()\
                       .rename(columns={'index': 'test_set'})
    stats_df.to_csv('tw_full_0stk.csv', index=None)

    t1 = ut.out('concatenating test set predictions...')
    df = full_df[['com_id', 'label']]

    if 'ind' in models:
        ind_df = pd.concat(inds)
        df = df.merge(ind_df)

        if 'mean' in models:
            approx_df = pd.concat(approxs)
            assert set(ind_df['com_id']) == set(approx_df['com_id'])
            df = df.merge(approx_df)

    if ref_start_fold > -1:
        ref_df = pd.concat(refs)
        ref_df = full_df[['com_id', 'label']].merge(ref_df)
        ref_df = ref_df[['com_id', 'ind_pred']]
        ref_df = ref_df.rename(columns={'ind_pred': 'ref_pred'})
        assert set(ind_df['com_id']) == set(ref_df['com_id'])
        df = df.merge(ref_df)

    if 'mrf' in models:
        mrf_df = pd.concat(mrfs)
        assert set(ind_df['com_id']) == set(mrf_df['com_id'])
        df = df.merge(mrf_df)

    if 'psl' in models:
        psl_df = pd.concat(psls)
        assert set(ind_df['com_id']) == set(psl_df['com_id'])
        df = df.merge(psl_df)
    ut.time(t1)

    t1 = ut.out('applying noise to predictions...')
    noise = 0.000025
    perturb = lambda x: max(0.0, min(1.0, x + ran.uniform(-noise, noise)))

    if 'ind' in models:
        df['ind_pred'] = df['ind_pred'].apply(perturb)

        if 'mean' in models:
            df['mean_pred'] = df['mean_pred'].apply(perturb)
            df['median_pred'] = df['median_pred'].apply(perturb)
            df['max_pred'] = df['max_pred'].apply(perturb)

    if 'mrf' in models:
        df['mrf_pred'] = df['mrf_pred'].apply(perturb)

    if 'psl' in models:
        df['psl_pred'] = df['psl_pred'].apply(perturb)
    ut.time(t1)

    # compute reference aupr and auroc
    ref_label, ref_pred = df['label'], df['ref_pred']
    ref_aupr = average_precision_score(ref_label, ref_pred)
    ref_auroc = roc_auc_score(ref_label, ref_pred)
    ref_p, ref_r, ref_t = precision_recall_curve(ref_label, ref_pred)
    ref_fpr, ref_tpr, ref_t2 = roc_curve(ref_label, ref_pred)
    ut.out('%s aupr: %.4f, auroc: %.4f' % ('reference', ref_aupr, ref_auroc))

    ut.plot_pr_curve('ref', ref_p, ref_r, ref_aupr, domain=domain,
                     line='k-', show_legend=True)
    ut.plot_roc_curve('ref', ref_tpr, ref_fpr, ref_auroc, domain=domain,
                      line='k-', show_legend=True)

    auroc_pval, aupr_pval = 0, 0
    # compute combined test set curves
    for i, (model, pred) in enumerate(preds):
        aupr = average_precision_score(df['label'], df[pred])
        auroc = roc_auc_score(df['label'], df[pred])
        p, r, _ = precision_recall_curve(df['label'], df[pred])
        fpr, tpr, _ = roc_curve(df['label'], df[pred])
        # aupr_pval, auroc_pval = _significance(df, pred)
        t = (model, aupr, aupr_pval, auroc, auroc_pval)
        ut.out('%s aupr: %.4f (%.4f), auroc: %.4f (%.4f)' % t)

        save = True if i == len(preds) - 1 else False
        ut.plot_pr_curve(model, p, r, aupr, domain=domain,
                         line=lines[model], show_legend=True)
        ut.plot_roc_curve(model, tpr, fpr, auroc, save=save, domain=domain,
                          line=lines[model], show_legend=True)
    ut.out()


def _relations_for_gids(gids):
    relations = []
    for gid in gids:
        group = gid.replace('_gid', '')
        rel = 'has' + group
        relations.append((rel, group, gid))
    return relations


def _significance(df, pred, samples=20):
    ref_auprs, pred_auprs = [], []
    ref_aurocs, pred_aurocs = [], []
    lc, rc = 'label', 'ref_pred'

    t1 = ut.out('computing aupr and auroc significance levels...')

    for i in range(samples):
        s_df = df.sample(frac=0.5, replace=True)
        ref_auprs.append(average_precision_score(s_df[lc], s_df[rc]))
        ref_aurocs.append(roc_auc_score(s_df[lc], s_df[rc]))
        pred_auprs.append(average_precision_score(s_df[lc], s_df[pred]))
        pred_aurocs.append(roc_auc_score(s_df[lc], s_df[pred]))

    auprs = np.subtract(ref_auprs, pred_auprs)
    aurocs = np.subtract(ref_aurocs, pred_aurocs)
    zeros = np.zeros(len(auprs))
    t1, aupr_pval = ttest_rel(auprs, zeros)
    t2, auroc_pval = ttest_rel(aurocs, zeros)
    ut.time(t1)

    return aupr_pval, auroc_pval


def _metrics(df, col='ind_pred', model='ind'):
    m_dict = {}
    m_dict[model + '_aupr'] = average_precision_score(df['label'], df[col])
    m_dict[model + '_auroc'] = roc_auc_score(df['label'], df[col])
    return m_dict


def _analyze(df, col, samples=100, relations=[]):
    gids = [r[2] for r in relations]

    if len(relations) == 0:
        return {}

    t1 = ut.out('computing messages missed most often...')

    p, r, ts = precision_recall_curve(df['label'], df[col])
    aupr = average_precision_score(df['label'], df[col])
    mp = 1.0 - aupr

    corrects = []
    step = int(len(ts) / 100) if len(ts) > 100 else 1
    for i in range(0, len(ts), step):
        t = ts[i]
        df['pred'] = np.where(df[col] > t, 1, 0)
        correct = df['pred'] == df['label']
        corrects.append(correct.apply(int))

    total_corrects = [sum(x) for x in zip(*corrects)]
    df['correct'] = total_corrects

    # extract bottom x% data
    df = df.sort_values('correct', ascending=False)
    ndx = len(df) - int(len(df) * mp)
    qf1, qf2 = df[ndx:], df[:ndx]
    # dfs = df[df['label'] == 1]
    qf1s = qf1[qf1['label'] == 1]  # low performers
    qf1o = qf1[qf1['label'] == 0]  # low performers
    qf2s = qf2[qf2['label'] == 1]  # high performers
    qf2o = qf2[qf2['label'] == 0]  # high performers
    ut.time(t1)

    # ut.out('spam in bot %.2f%%: %d' % (mp * 100, len(qf1s)))
    # ut.out('ham in bot %.2f%%: %d' % (mp * 100, len(qf1o)))

    t1 = ut.out('computing messages with a relation...')
    r1s, r1sf = _msgs_with_rel(qf1s, gids, mp, 'bot', 'spam')
    r1o, r1of = _msgs_with_rel(qf1o, gids, mp, 'bot', 'ham')
    r2s, r2sf = _msgs_with_rel(qf2s, gids, mp, 'top', 'spam')
    r2o, r2of = _msgs_with_rel(qf2o, gids, mp, 'top', 'ham')
    ut.time(t1)

    # ut.out()

    t1 = ut.out('computing messages with an outside relation...')
    rr1sof = _rm_in_sect(df, qf1s, qf2, gids, mp, r1s, 'bot', 'spam')
    rr1oof = _rm_in_sect(df, qf1o, qf2, gids, mp, r1o, 'bot', 'ham')
    rr2sof = _rm_in_sect(df, qf2s, qf1, gids, mp, r2s, 'top', 'spam')
    rr2oof = _rm_in_sect(df, qf2o, qf1, gids, mp, r2o, 'top', 'ham')
    # rr1sif = self._rm_in_sect(df, qf1s, qf1, gids, mp, r1s, 'bot', 'spam',
    #                           'inside')
    # rr1oif = self._rm_in_sect(df, qf1o, qf1, gids, mp, r1o, 'bot', 'ham',
    #                           'inside')

    sd = {}
    sd['bot_spam_rels'] = round(r1sf, 4)
    sd['bot_ham_rels'] = round(r1of, 4)
    sd['top_spam_rels'] = round(r2sf, 4)
    sd['top_ham_rels'] = round(r2of, 4)
    sd['bot_spam_rels_out'] = round(rr1sof, 4)
    sd['bot_ham_rels_out'] = round(rr1oof, 4)
    sd['top_spam_rels_out'] = round(rr2sof, 4)
    sd['top_ham_rels_out'] = round(rr2oof, 4)
    # sd['bot_spam_rels_in'] = rr1sif
    # sd['bot_ham_rels_in'] = rr1oif

    ut.time(t1)
    return sd


def _msgs_with_rel(df, gids, miss_pct, sect='bot', lbl='spam'):
    n = len(df[(df[gids] != -1).any(axis=1)])
    frac = ut.div0(n, len(df))
    # sect_pct = 1 - miss_pct if sect == 'top' else miss_pct
    # t = (sect, sect_pct * 100, lbl, frac * 100)
    # ut.out('%s %.2f%% %s w/ relations: %.2f%%' % t)
    return n, frac


def _rm_in_sect(df, tgt_df, cmp_df, gids, miss_pct, num_rel_msgs,
                sect='bot', lbl='spam', boundary='outside'):
    fraction = set()
    tgt_msgs = set(tgt_df['com_id'])
    cmp_msgs = set(cmp_df['com_id'])

    for gid in gids:
        tgt_gids = {x for x in tgt_df[gid] if x != -1}

        for temp_gid, qf in df.groupby(gid):
            if temp_gid in tgt_gids:
                grp_msgs = set(qf['com_id'])
                other = grp_msgs.intersection(cmp_msgs)
                if len(other) > 0:
                    fraction.update(grp_msgs.intersection(tgt_msgs))

    n = ut.div0(len(fraction), num_rel_msgs)
    # sect_pct = 1 - miss_pct if sect == 'top' else miss_pct
    # t = (sect, sect_pct * 100, lbl, boundary, sect, n * 100)
    # self.util_obj.out('%s %.2f%% %s w/ rels %s %s: %.2f%%' % t)
    return n


def _spread(df, col='ind_pred', relations=[]):
    """This'll give some post-hoc test-set analysis, when running this,
    keep track of the test sets that improved using relational modeling,
    then average those test set statistics together to compare to the test
    sets that did not improve."""
    t1 = ut.out('computing subgraph statistics...')
    con_obj = Connections()

    gids = [r[2] for r in relations]
    g, sgs = con_obj.find_subgraphs(df, relations, verbose=False)
    spread_dict = {}

    sg_list = []
    for i, sg in enumerate(sgs):
        if sg[3] > 0:  # num edges > 0
            sg_list.extend([(x, i) for x in sg[0]])  # give sg_id

    if len(sg_list) == 0:
        return spread_dict

    sg_df = pd.DataFrame(sg_list, columns=['com_id', 'sg_id'])
    df = df.merge(sg_df, how='left')
    df['sg_id'] = df['sg_id'].fillna(-1).apply(int)

    p, r, ts = precision_recall_curve(df['label'], df[col])
    aupr = average_precision_score(df['label'], df[col])
    mp = 1.0 - aupr

    corrects = []
    step = int(len(ts) / 100) if len(ts) > 100 else 1
    for i in range(0, len(ts), step):
        t = ts[i]
        df['pred'] = np.where(df[col] > t, 1, 0)
        correct = df['pred'] == df['label']
        corrects.append(correct.apply(int))

    total_corrects = [sum(x) for x in zip(*corrects)]
    df['correct'] = total_corrects

    # extract bottom x% data
    df = df.sort_values('correct', ascending=False)
    ndx = len(df) - int(len(df) * mp)
    qfs = df[df['label'] == 1]
    qfo = df[df['label'] == 0]
    qf1, qf2 = df[ndx:], df[:ndx]
    qf1s = qf1[qf1['label'] == 1]  # low performers
    qf1o = qf1[qf1['label'] == 0]  # low performers
    qf2s = qf2[qf2['label'] == 1]  # high performers
    qf2o = qf2[qf2['label'] == 0]  # high performers

    spread_dict['spam_mean'] = round(qfs['ind_pred'].mean(), 4)
    spread_dict['spam_median'] = round(qfs['ind_pred'].median(), 4)
    spread_dict['ham_mean'] = round(qfo['ind_pred'].mean(), 4)
    spread_dict['ham_median'] = round(qfo['ind_pred'].median(), 4)

    for nm, temp_df in [('bot_spam', qf1s), ('bot_ham', qf1o),
                        ('top_spam', qf2s), ('top_ham', qf2o)]:
        wf = temp_df[(temp_df[gids] != -1).any(axis=1)]
        sg_mean = wf.groupby('sg_id')['ind_pred'].mean().reset_index()\
            .rename(columns={'ind_pred': 'sg_mean'})
        sg_std = wf.groupby('sg_id')['ind_pred'].std().reset_index()\
            .rename(columns={'ind_pred': 'sg_std'})
        sg_median = wf.groupby('sg_id')['ind_pred'].median().reset_index()\
            .rename(columns={'ind_pred': 'sg_median'})
        sg_min = wf.groupby('sg_id')['ind_pred'].min().reset_index()\
            .rename(columns={'ind_pred': 'sg_min'})
        sg_max = wf.groupby('sg_id')['ind_pred'].max().reset_index()\
            .rename(columns={'ind_pred': 'sg_max'})
        wf = wf.merge(sg_mean).merge(sg_std).merge(sg_median)\
            .merge(sg_min).merge(sg_max)
        wf['sg_spread'] = wf['sg_max'] - wf['sg_min']

        spread_dict[nm + '_sg_mean'] = round(np.mean(wf['sg_mean']), 4)
        spread_dict[nm + '_sg_std'] = round(np.mean(wf['sg_std']), 4)
        spread_dict[nm + '_sg_median'] = round(np.mean(wf['sg_median']), 4)
        spread_dict[nm + '_sg_min'] = round(np.mean(wf['sg_min']), 4)
        spread_dict[nm + '_sg_max'] = round(np.mean(wf['sg_max']), 4)
        spread_dict[nm + '_sg_spread'] = round(np.mean(wf['sg_spread']), 4)

    ut.time(t1)
    return spread_dict


def _approximations(df, relations=[]):
    t1 = ut.out('approximating relational with mean, max, median...')
    df = df.copy()

    con_obj = Connections()

    g, sgs = con_obj.find_subgraphs(df, relations, verbose=False)
    approx_dict = {}

    sg_list = []
    for i, sg in enumerate(sgs):
        if sg[3] > 0:  # num edges > 0
            sg_list.extend([(x, i) for x in sg[0]])  # give sg_id

    if len(sg_list) == 0:
        return approx_dict

    sg_df = pd.DataFrame(sg_list, columns=['com_id', 'sg_id'])
    df = df.merge(sg_df, how='left')
    df['sg_id'] = df['sg_id'].fillna(-1).apply(int)

    sg_mean = df.groupby('sg_id')['ind_pred'].mean().reset_index()\
        .rename(columns={'ind_pred': 'sg_mean_pred'})
    sg_median = df.groupby('sg_id')['ind_pred'].median().reset_index()\
        .rename(columns={'ind_pred': 'sg_median_pred'})
    sg_max = df.groupby('sg_id')['ind_pred'].max().reset_index()\
        .rename(columns={'ind_pred': 'sg_max_pred'})
    df = df.merge(sg_mean).merge(sg_median).merge(sg_max)

    filler = lambda x, c: x['ind_pred'] if x['sg_id'] == -1 else x[c]
    for col in ['sg_mean_pred', 'sg_median_pred', 'sg_max_pred']:
        cols = ['ind_pred', col, 'sg_id']
        df[col] = df[cols].apply(filler, axis=1, args=(col,))

    ut.time(t1)

    return df


if __name__ == '__main__':
    description = 'Script to merge and compute subset predictions scores'
    parser = argparse.ArgumentParser(description=description, prog='big_aupr')

    parser.add_argument('-d', metavar='DOMAIN',
                        help='domain, default: %(default)s')
    parser.add_argument('--start_fold', metavar='NUM', type=int,
                        help='first subset, default: %(default)s')
    parser.add_argument('--ref_start_fold', metavar='NUM', type=int, default=-1,
                        help='references first subset, default: %(default)s')
    parser.add_argument('--num_folds', metavar='NUM', type=int,
                        help='number of subsets, default: %(default)s')
    parser.add_argument('--models', nargs='*', metavar='MODEL',
                        help='list of models, default: %(default)s')
    parser.add_argument('--in_dir', metavar='DIR', default='',
                        help='predictions directory, default: %(default)s')
    parser.add_argument('--gids', nargs='*', metavar='GID',
                        help='list of gids, default: %(default)s')

    args = parser.parse_args()
    domain = args.d
    start = args.start_fold
    ref_start = args.ref_start_fold
    folds = args.num_folds
    models = args.models if args.models is not None else ['ind']
    in_dir = args.in_dir
    gids = args.gids if args.gids is not None else []

    compute_big_aupr(start_fold=start, ref_start_fold=ref_start,
                     num_folds=folds, domain=domain, models=models,
                     in_dir=in_dir, gids=gids)
