import pandas as pd


def split(df, num_chunks=80, out_dir='', out_fname='chunk_'):

    chunk_size = int(len(df) / num_chunks)
    chunk_start = 0

    for i in range(num_chunks):
        chunk_df = df[chunk_start: chunk_start + chunk_size]
        chunk_df.to_csv(out_dir + out_fname + str(i) + '.csv', index=None)
        chunk_start += chunk_size


def combine(in_dir='', in_fname='replace_', num_chunks=80, out_dir='',
            out_fname='comments.csv'):
    chunks = []

    for i in range(num_chunks):
        chunks_df = pd.read_csv(in_dir + in_fname + str(i) + '.csv')
        chunks.append(chunks_df)

    df = pd.concat(chunks)
    df.to_csv(out_dir + out_fname, index=None)


if __name__ == '__main__':
    data_dir = '/Volumes/Brophy/data/twitter/'
    df = pd.read_csv(data_dir + 'comments.csv')
    split(df, num_chunks=5, out_dir=data_dir + 'chunks/')
    # combine(in_dir='chunks/', num_chunks=80)
