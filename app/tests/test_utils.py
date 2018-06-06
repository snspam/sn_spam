"""
Provides test files with convenience methods.
"""
import pandas as pd
from app import config


def sample_config():
    c = config.Config()
    c.start = '0'
    c.end = '1000'
    c.train_size = 0.7
    c.classifier = 'lr'
    c.ngrams = False
    c.fold = '1'
    c.relations = [('intext', 'text', 'text_id'),
            ('posts', 'user', 'user_id')]
    c.domain = 'soundcloud'
    c.model = 'basic'
    c.engine = 'psl'
    c.ind_dir = 'ind/'
    c.rel_dir = 'rel/'
    c.debug = False
    return c


def simple_df():
    l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    df = pd.DataFrame(l)
    return df


def sample_df(num):
    l1, l2 = list(range(num)), list(range(num, num + num))
    l = list(zip(l1, l2))
    df = pd.DataFrame(l)
    df.columns = ['com_id', 'random']
    return df


def sample_df_with_com_id(num):
    l1, l2 = list(range(num)), list(range(num, num + num))
    l = list(zip(l1, l2))
    df = pd.DataFrame(l)
    df.columns = ['com_id', 'random']
    return df


def sample_df_with_user_id(num):
    l1, l2 = list(range(num)), list(range(num, num + num))
    l = list(zip(l1, l2))
    df = pd.DataFrame(l)
    df.columns = ['user_id', 'random']
    return df


def sample_df_with_com_id_and_user_id(num):
    l1, l2 = list(range(num)), list(range(num, num + num))
    l = list(zip(l1, l2))
    df = pd.DataFrame(l)
    df.columns = ['com_id', 'user_id']
    return df


def sample_group_df(l=None):
    if l is None:
        l = [1, 1, 2, 2, 3, 3, 4]
    sample_df = pd.DataFrame(l)
    sample_df.columns = ['com_id']
    return sample_df


def sample_group_user_df():
    l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    sample_df = pd.DataFrame(l)
    sample_df.columns = ['com_id']
    sample_df['user_id'] = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    return sample_df


def sample_text_df():
    l = ['cool', 'cool', 'tool', 'tool', 'm', 'm', 'f', 'f', 'p', 'p']
    sample_df = pd.DataFrame(l)
    sample_df.columns = ['text']
    return sample_df


def sample_relational_df():
    l = [[1, 10, 100, 0], [2, 10, 101, 1], [3, 11, 101, 1],
            [4, 12, 102, 0], [5, 13, 103, 0], [6, 10, 100, 0],
            [7, 14, 104, 1], [8, 14, 105, 0]]
    df = pd.DataFrame(l)
    df.columns = ['com_id', 'user_id', 'text_id', 'label']
    return df


def sample_perturbed_df():
    l = [(100, 0.75, 0.85, 0.65, 0.95, 0.55),
         (101, 0.75, 0.85, 0.65, 0.95, 0.55),
         (102, 0.75, 0.85, 0.65, 0.95, 0.55),
         (103, 0.75, 0.85, 0.65, 0.95, 0.55),
         (104, 0.75, 0.85, 0.65, 0.95, 0.55)]
    df = pd.DataFrame(l)
    df.columns = ['com_id', 'ind_pred', '0', '1', '2', '3']
    return df
