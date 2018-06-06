import re
import sys
import httplib2
import pandas as pd


# public
def lengthen_urls(df, c='text', regex_str=r'(http[^\s]+)', out_dir='',
                  fname='comments.csv'):

    h = httplib2.Http('.cache')
    regex = re.compile(regex_str)

    msgs = list(zip(list(df.index), list(df[c])))

    for i, (n, string) in enumerate(msgs):
        _out('(%d/%d)' % (i, len(msgs)))

        short_urls = regex.findall(string)

        for short_url in short_urls:
            try:
                header = h.request(short_url)[0]
                if 'content-location' in header:
                    long_url = header['content-location']
                    df.at[n, c] = df.at[n, c].replace(short_url, long_url)
                    _out('%s -> %s' % (short_url, long_url))
                else:
                    _out('ERR: %s' % short_url)
            except Exception:
                _out('ERR: %s' % short_url)

    df = df[['com_id', 'text']]
    df.to_csv(out_dir + fname, index=None)


# private
def _out(message=''):
    sys.stdout.write(message + '\n')
    sys.stdout.flush()


if __name__ == '__main__':
    _out('reading in messages...')
    chunk = 3
    df = pd.read_csv('short_' + str(chunk) + '.csv')
    lengthen_urls(df, out_dir='', fname='long_' + str(chunk) + '.csv')
