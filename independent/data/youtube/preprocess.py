import pandas as pd

df = pd.read_csv('youtube_comments_20120117.csv', header=None)
df = df.reset_index()
df.columns = ['com_id', 'timestamp', 'vid_id', 'user_id', 'text', 'label']
df['vid_id'] = df['vid_id'].str.replace('video', '').apply(int)
df['user_id'] = df['user_id'].str.replace('user', '').apply(int)
df['label'] = df['label'].apply(lambda x: 0 if x is False else 1)
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.to_csv('comments.csv', index=None)
