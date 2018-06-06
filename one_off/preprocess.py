import argparse
import util as ut
import numpy as np
import pandas as pd
from textblob import TextBlob


def adclicks(data_dir=''):
    data_dir += 'adclicks/'

    df = pd.read_csv(data_dir + 'test.csv')
    df = df.rename(columns={'click_id': 'com_id'})
    df.to_csv(data_dir + 'test.csv', index=None)

    max_id = df['com_id'].max() + 1

    df = pd.read_csv(data_dir + 'train.csv')
    df = df.rename(columns={'is_attributed': 'label'})
    df['com_id'] = range(max_id, max_id + len(df))
    df.to_csv(data_dir + 'train.csv', index=None)


def russia(data_dir):
    ut.out('reading in data...')
    df = pd.read_csv(data_dir + '2016_election.csv', lineterminator='\n')
    df = df.drop_duplicates('com_id')
    df.to_csv('2016_election.csv', index=None)


def sentiment(data_dir, chunk):
    ut.out('%s, chunk: %d' % (data_dir, chunk), 0)
    polarity = lambda x: TextBlob(x).sentiment.polarity
    subjectivity = lambda x: TextBlob(x).sentiment.subjectivity

    ut.out('reading data...')
    df = pd.read_csv(data_dir + 'comments.csv', lineterminator='\n')
    if chunk >= 0 and chunk <= 8:
        ut.out('splitting data and retrieving chunk %d' % chunk)
        dfs = np.array_split(df, 8)
        df = dfs[chunk]

    df['text'] = df['text'].fillna('')

    ut.out('polarity...')
    df['polarity'] = df['text'].apply(polarity)

    ut.out('subjectivity...')
    df['subjectivity'] = df['text'].apply(subjectivity)

    ut.out('writing...')
    if chunk >= 0 and chunk <= 8:
        df.to_csv(data_dir + 'comments_%d.csv' % chunk, index=None)
    else:
        df.to_csv(data_dir + 'comments_new.csv', index=None)
    ut.out()


def soundcloud(data_dir, chunks):
    dfs = []
    for i in range(chunks):
        df = pd.read_csv(data_dir + 'comments_%d.csv' % i, lineterminator='\n')
        dfs.append(df)
    df = pd.concat(dfs)
    df.to_csv(data_dir + 'new_comments.csv', index=None)


if __name__ == '__main__':
    description = 'Preprocessing'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', default='twitter', metavar='DOMAIN',
                        help='Social Network, default: %(default)s')
    parser.add_argument('--chunk', default=-1, type=int,
                        help='Chunks to combine, default: %(default)s')
    parser.add_argument('--chunks', default=-1, type=int,
                        help='Chunks to combine, default: %(default)s')
    args = parser.parse_args()

    data_dir = 'independent/data/' + args.d + '/'
    sentiment(data_dir, args.chunk)
