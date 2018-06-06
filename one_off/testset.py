import os
import argparse
import util as ut
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def _rank_score(x, totham, spamsum_col, hamsum_col):
    label, spam_sum, ham_sum = x['label'], x['spam_sum'], x['ham_sum']
    return spam_sum if label == 0 else totham - ham_sum

# read in test predictions
df = pd.read_csv('')
ip = pd.read_csv('')  # ind_preds
pp = pd.read_csv('')  # psl_preds
qf = df.merge(ip).merge(pp)

# generate relations

# normalize predictions
qf['ind_pred'] = qf['ind_pred'] / qf['ind_pred'].max()
qf['psl_pred'] = qf['psl_pred'] / qf['psl_pred'].max()

# rank predictions
qf['ind_rank'] = qf['ind_pred'].rank(method='first').apply(int)
qf['psl_rank'] = qf['psl_pred'].rank(method='first').apply(int)
qf['inv_lbl'] = qf['label'].apply(lambda x: 1 if x == 0 else 0)

# compute basic statistics about test set
qfh = qf[qf['label'] == 0]
qfs = qf[qf['label'] == 1]
total_ham = len(qfh)
total_spam = len(qfs)

ham_ind_mean = qfh['ind_pred'].mean()
ham_ind_median = qfh['ind_pred'].median()
spam_ind_mean = qfh['ind_pred'].mean()
spam_ind_median = qfh['ind_pred'].median()

ham_psl_mean = qfh['psl_pred'].mean()
ham_psl_median = qfh['psl_pred'].median()
spam_psl_mean = qfh['psl_pred'].mean()
spam_psl_median = qfh['psl_pred'].median()

# independent ranking analysis
qf = qf.sort_values('ind_rank')
qf['ind_spam_sum'] = qf['label'].cumsum()
qf['ind_ham_sum'] = qf['inv_lbl'].cumsum()
qf['ind_rank_score'] = qf.apply(_rank_score, args=(total_ham, 'ind_spam_sum',
                                                   'ind_ham_sum'))

# take bottom x% of lowest rank spam scorers (i.e. highest score ranks)

# how many of these are connected to others in the bottom x%?

# how many connected to other spam messages in the test set, no dbl counting messages if they overlap between relational groups.

# how many of these are connected other ham messages, no dbl counting.

# psl ranking analysis
qf = qf.sort_values('psl_rank')
qf['psl_spam_sum'] = qf['label'].cumsum()
qf['psl_ham_sum'] = qf['inv_lbl'].cumsum()
qf['ind_rank_score'] = qf.apply(_rank_score, args=(total_ham, 'psl_spam_sum',
                                                   'psl_ham_sum'))




binify = lambda x, tx: 0 if x <= tx else 1
missify = lambda x: 1 if x['pred'] == x['label'] else 0
p, r, ts = precision_recall_curve(df['label'], df['ind_pred'])
corrects = []
for i, t in enumerate(ts):
    if i % 50 == 0:
        print(i, t)
    df['pred'] = np.where(df['ind_pred'] > t, 1, 0)
    correct = df['pred'] == df['label']
    corrects.append(correct.apply(int))

total_corrects = [sum(x) for x in zip(*corrects)]
df['correct'] = total_corrects

if __name__ == '__main__':
    description = 'Script to merge and compute subset predictions scores'
    parser = argparse.ArgumentParser(description=description, prog='big_aupr')

    parser.add_argument('-d', metavar='DOMAIN',
                        help='domain, default: %(default)s')
    parser.add_argument('--start_fold', metavar='NUM', type=int,
                        help='first subset, default: %(default)s')
    parser.add_argument('--num_folds', metavar='NUM', type=int,
                        help='number of subsets, default: %(default)s')

    args = parser.parse_args()
