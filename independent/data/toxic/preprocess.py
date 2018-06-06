import pandas as pd

filename = ''
df = pd.read_csv(filename)


def winning(x):
    result = 0
    if x['toxic'] == 1 or x['severe_toxic'] == 1 or x['obscene'] == 1 or x['threat'] == 1 or x['identity_hate'] == 1 or x['insult'] == 1:
        result = 1
    return result

df['test'] = df.apply(winning, axis=1)
q = df.copy()

l = list(q)
q.remove('id')
q.remove('comment_text')
q.remove('test')

q = q.drop(l, axis=1)
q.columns = ['com_id', 'text', 'label']
q.to_csv('blah.csv', index=None)
