import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlb
import pandas as pd
import util as ut
from scipy.stats import norm


def plot_distributions(df, feats=[]):

    dfs = df[df['label'] == 1]
    dfh = df[df['label'] == 0]

    for feat in feats:
        ut.out('plotting distribution for: %s\n' % feat, 0)
        f, ax = plt.subplots(1, 1)
        ns, bs, ps = ax.hist(dfs[feat], normed=1, color='r', alpha=0.69)
        nh, bh, ph = ax.hist(dfh[feat], normed=1, color='b', alpha=0.69)
        ms, ss = norm.fit(dfs[feat])
        mh, sh = norm.fit(dfh[feat])
        ys = mlb.normpdf(bs, ms, ss)
        yh = mlb.normpdf(bh, mh, sh)
        ax.plot(bs, ys, 'r--')
        ax.plot(bh, yh, 'b--')
        f.savefig(feat + '.pdf', format='pdf', bbox_inches='tight')
        plt.clf()


def compute_features(df):
    df.text = df.text.fillna('')
    df['len'] = df.text.str.len()
    return df

if __name__ == '__main__':
    domain = 'soundcloud'
    nrows = 1000000
    path = 'independent/data/%s/comments.csv' % domain
    df = pd.read_csv(path, nrows=nrows)

    feats = ['polarity', 'subjectivity', 'len']
    df = compute_features(df)
    plot_distributions(df, feats=feats)
