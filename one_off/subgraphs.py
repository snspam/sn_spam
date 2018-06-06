import argparse
import util as ut
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from generator import Generator


def do_log_scale_binning(dummy_df, base=2):
    size = dummy_df['size']
    max_size = max(size)

    stop = 0
    for j in range(100):
        if base ** j >= max_size:
            stop = j
            break

    cuts = np.logspace(0, stop, stop + 1, base=base, dtype=int)
    splits = np.split(size, cuts)

    bins = []
    for bin_ndx, split_arr in enumerate(splits):
        bins.extend(len(split_arr) * [bin_ndx])
    dummy_df['q'] = bins

    ll = list(dummy_df)
    l = list(dummy_df)
    l.remove('q')
    l.remove('size')

    gs = dummy_df.groupby('q')['size'].max().reset_index()

    if 'cnt_rto' in l:
        l.remove('cnt_rto')
        gr = dummy_df.groupby('q')['cnt_rto'].sum().reset_index()
        ga = dummy_df.groupby('q')[l].mean().reset_index()
        gf = gs.merge(gr).merge(ga)
        assert np.isclose(gf['cnt_rto'].sum(), 1.0, atol=1e-04)
    else:
        ga = dummy_df.groupby('q')[l].mean().reset_index()
        gf = gs.merge(ga)

    gf = gf[ll]
    return gf


def compute_stats_per_size(gf, p_spam, p_ham, label='both',
                           total_pts=0, single_cnt=0):
    g2 = gf.groupby('size')
    cnt = g2.size().reset_index().rename(columns={0: 'cnt'})
    slc = g2['same_label'].sum().reset_index()\
        .rename(columns={'same_label': 'same_label_cnt'})
    mean_label2 = g2['mean_label'].mean().reset_index()
    sf = cnt.merge(slc).merge(mean_label2)

    if label == 'both':
        assert total_pts > 0
        assert single_cnt > 0
        sf['cnt_rto'] = (sf['cnt'] * sf['size']) / total_pts
        sf['same_label_rto'] = sf['same_label_cnt'] / sf['cnt']

    if label in ['both', 'spam']:
        sf = sf[sf['size'] != single_cnt]

    if label == 'both':
        # compute E[same_label_rto | spam/ham ratios]
        prob_all_same = lambda x: np.prod(x * [p_ham]) + np.prod(x * [p_spam])
        sf['e_sme_lbl_rto'] = sf['size'].apply(prob_all_same)
    elif label == 'spam':
        sf['e_mean_lbl_sme_lbl'] = p_spam
    elif label == 'ham':
        sf['e_mean_lbl_not_sme_lbl'] = sf['size'].apply(lambda x: p_spam ** x)
    return sf


def single_relational(in_dir='', out_dir='', gids=['text_gid'], pts=100000,
                      start=0, dom=''):
    gen = Generator()

    ut.out('gids: %s' % str(gids), 0)

    t1 = ut.out('reading data...')
    df = pd.read_csv(in_dir + 'comments.csv', skiprows=range(1, start),
                     nrows=pts)
    pts = len(df)
    ut.time(t1)

    colors = {'twitter': 'blue', 'youtube': 'red', 'soundcloud': 'orange'}

    # basic statistics
    p_spam = df.label.sum() / len(df)
    p_ham = 1 - p_spam
    ut.out('spam pct: %.2f' % p_spam)

    for gid in gids:
        t1 = ut.out('generating %s...' % gid)
        df = gen.gen_group_id(df, gid)
        ut.time(t1)

    for gid in gids:
        t1 = ut.out('grouping by %s...' % gid)
        g1 = df.groupby(gid)
        ut.time(t1)

        t1 = ut.out('computing stats per group...')
        size = g1.size().reset_index().rename(columns={0: 'size'})
        sum_label = g1['label'].sum().reset_index()\
            .rename(columns={'label': 'sum_label'})
        mean_label = g1['label'].mean().reset_index()\
            .rename(columns={'label': 'mean_label'})
        gf = size.merge(sum_label).merge(mean_label)

        single_cnt = gf[gf[gid] == -1]['size'].values[0]

        same_label = lambda x: 1 if x['mean_label'] in [1.0, 0.0] else 0
        gf['same_label'] = gf.apply(same_label, axis=1)
        gfs = gf[(gf['same_label'] == 1) | (gf[gid] == -1)]
        gfo = gf[(gf['same_label'] == 0) & (gf[gid] != -1)]
        ut.time(t1)

        t1 = ut.out('computing stats per size...')
        sf = compute_stats_per_size(gf, p_spam, p_ham, label='both',
                                    total_pts=pts, single_cnt=single_cnt)
        sfs = compute_stats_per_size(gfs, p_spam, p_ham, label='spam',
                                     single_cnt=single_cnt)
        sfo = compute_stats_per_size(gfo, p_spam, p_ham, label='ham')

        # compute single node row
        # sfs_df = sfs_df[sfs_df['size'] != single_cnt]
        extract = ['size', 'mean_label']
        vs = gfs[gfs[gid] == -1][extract].values[0]
        row = [(1, single_cnt, single_cnt, vs[1], p_spam)]
        one_line = pd.DataFrame(row, columns=list(sfs))
        sfs = pd.concat([one_line, sfs])
        sfs = sfs.rename(columns={'mean_label': 'mean_lbl_sme_lbl'})
        sfo = sfo.rename(columns={'mean_label': 'mean_lbl_not_sme_lbl'})

        # compute single node row
        v = gf[gf[gid] == -1][['size', 'mean_label']].values[0]
        row = [(1, v[0], v[0], v[1], v[0] / pts, 1, 1)]
        cols = list(sf)
        one_line = pd.DataFrame(row, columns=cols)
        sf = pd.concat([one_line, sf])

        # keep top X% of affected nodes
        pct = 100
        total = sf.cnt.sum()
        for i in range(1, len(sf)):
            if sf[:i].cnt.sum() / total >= pct / float(100):
                sf = sf[:i]
                break
        ut.time(t1)

        t1 = ut.out('plotting...')
        cols = ['cnt_rto', 'same_label_rto',
                'mean_lbl_sme_lbl', 'mean_lbl_not_sme_lbl']
        ncols = len(cols)

        subtitle_list = ['(a)', '(b)', '(c)', '(d)']
        xlabel_list = ['% all messages',
                       '% groups containing same label',
                       'mean label over same label groups',
                       'mean label over mixed label groups']
        subtitles = dict(list(zip(cols, subtitle_list)))
        xlabels = dict(list(zip(cols, xlabel_list)))
        fontsize = 24

        # nrows = 2
        # ncols = int(ncols / nrows)
        # ncols += 1 if ncols % nrows != 0 else 0

        # fig, axs = plt.subplots(nrows, ncols, figsize=(15, 15))
        fig, axs = plt.subplots(1, 4, figsize=(27, 7))
        # fig, axs = plt.subplots(1, 4)
        axs = axs.flatten()
        for i, col in enumerate(cols):
            if col == 'mean_lbl_sme_lbl':
                dummy_df = sfs
            elif col == 'mean_lbl_not_sme_lbl':
                dummy_df = sfo
            else:
                dummy_df = sf

            if len(dummy_df) > 0:
                gf = do_log_scale_binning(dummy_df)
                gf.plot.barh('size', col, ax=axs[i], title=subtitles[col],
                             legend=False, fontsize=fontsize)
                if col == 'same_label_rto':
                    gf.plot.barh('size', 'e_sme_lbl_rto', ax=axs[i],
                                 title=subtitles[col], legend=False,
                                 fontsize=fontsize, alpha=0.5, color='red',
                                 hatch='/')
                elif col in ['mean_lbl_sme_lbl', 'mean_lbl_not_sme_lbl']:
                    axs[i].axvline(p_spam, color='k', linestyle='--')

                axs[i].set_ylabel('group size', fontsize=fontsize)
                axs[i].set_xlabel(xlabels[col], fontsize=fontsize)
                axs[i].set_title(subtitles[col], fontsize=fontsize - 2)

                xt = axs[i].get_xticks()
                tl = len(str(xt[1]))
                if (tl >= 4) or (tl == 3 and len(xt) >= 9):
                    axs[i].set_xticks(axs[i].get_xticks()[::2])

        rel = gid.replace('_gid', '')
        # t = (dom, pts, p_spam * 100, rel)
        title = '%s: spam: %.2f%%, relation: %s' % (dom, p_spam * 100, rel)
        # title = '%s: %d data points, spam: %.2f%%, relation: %s' % t
        fig.tight_layout()
        fig.suptitle(title, y=1.08, fontsize=fontsize)
        fig.savefig(out_dir + 'sg_%s.pdf' % str(gid), format='pdf',
                    bbox_inches='tight')
        plt.close('all')
        ut.time(t1)

        sf.to_csv(out_dir + 'sg_%s.csv' % str(gid), index=None)

    spam_rto = df.label.sum() / len(df)
    ut.out('spam ratio: %.2f' % spam_rto)

    if len(gids) > 1:
        rel_nodes = 0
        g = df.groupby(gids).size().reset_index().rename(columns={0: 'size'})
        for gid in gids:
            g = g[g[gid] != -1]
            rel_nodes += len(df[df[gid] != -1])

        spam_rto = df.label.sum() / len(df)
        overlap_rto = g.size.sum() / rel_nodes
        ut.out('overlap ratio: %.2f' % overlap_rto)
    ut.out()

if __name__ == '__main__':
    pd.set_option('display.width', 181)

    description = 'Script to analyze connected components'
    parser = argparse.ArgumentParser(description=description, prog='subgraphs')

    parser.add_argument('-d', metavar='DOMAIN',
                        help='domain, default: %(default)s')
    parser.add_argument('-s', metavar='NUM', type=int, default=0,
                        help='start, default: %(default)s')
    parser.add_argument('-n', metavar='NUM', type=int, default=100000000,
                        help='nrows, default: %(default)s')
    parser.add_argument('--gids', nargs='*', metavar='GID',
                        help='list of gids, default: %(default)s')
    args = parser.parse_args()

    domain = args.d
    start = args.s
    nrows = args.n
    gids = args.gids if args.gids is not None else []

    in_dir = 'independent/data/' + domain + '/'
    out_dir = 'relational/output/' + domain + '/subgraphs/'
    ut.makedirs(out_dir)

    single_relational(in_dir, out_dir, gids=gids, pts=nrows, start=start,
                      dom=domain)


# lower rank score is better
def rank_score(x, totham):
    label, spam_sum, ham_sum = x['label'], x['spam_sum'], x['ham_sum']
    result = 0
    if label == 0:
        result = spam_sum
    elif label == 1:
        result = totham - ham_sum
    return result
