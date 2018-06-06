import re
import pandas as pd
import util as ut
import argparse


def extract(df, target_col='text', info_type='link', out_dir=''):
    ut.makedirs(out_dir)
    df = df[['com_id', target_col]]

    ut.out('target column: %s, info type: %s' % (target_col, info_type))

    if info_type == 'text':
        ut.out('writing info to csv...\n')
        df.to_csv(out_dir + info_type + '.csv', index=None)
        return

    d, i = {}, 0
    regex = _get_regex(info_type)

    df = df[['com_id', target_col]]
    for ndx, com_id, text in df.itertuples():
        i += 1
        if i % 100000 == 0:
            ut.out('(%d/%d)...' % (i, len(df)))

        info = _get_items(text, regex)
        if info != '':
            d[com_id] = info

    if len(d) > 0:
        info_df = pd.DataFrame.from_dict(d, orient='index').reset_index()
        info_df.columns = ['com_id', info_type]
        fname = info_type + '.csv'
        ut.out(str(info_df))
        ut.out('writing info to csv...\n')
        info_df.to_csv(out_dir + fname, index=None)
    else:
        ut.out('No extractions made...')


# private
def _get_regex(info_type='link'):
    d = {}
    d['hashtag'] = re.compile(r'(#\w+)')
    d['mention'] = re.compile(r'(@\w+)')
    d['link'] = re.compile(r'(http[^\s]+)')
    return d[info_type]


def _get_items(text, regex, str_form=True):
    items = regex.findall(str(text))
    result = sorted([x.lower() for x in items])
    if str_form:
        result = ''.join(result)
    return result

if __name__ == '__main__':
    description = 'Extracts elements from strings (links, hashtags, etc.)'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-t', '--target_col', default='text',
                        help='column to extract from, default: %(default)s')
    parser.add_argument('-i', '--info_type', default='hashtag',
                        help='type of element to extract default: %(default)s')
    parser.add_argument('-d', '--domain', default='twitter',
                        help='social network, default: %(default)s')
    parser.add_argument('-n', '--nrows', default=None,
                        help='number of rows to read, default: %(default)s')
    args = parser.parse_args()

    domain = args.domain
    target_col = args.target_col
    info_type = args.info_type
    nrows = int(args.nrows) if args.nrows is not None else None

    data_dir = 'independent/data/' + domain + '/'
    out_dir = 'independent/data/' + domain + '/extractions/'

    ut.out('reading in data...')
    df = pd.read_csv(data_dir + 'comments.csv', nrows=nrows)
    extract(df, target_col=target_col, info_type=info_type, out_dir=out_dir)
