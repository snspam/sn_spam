YouTube Dataset
===

### Preprocessing ###

1. Place `youtube_comments_20120117.csv` file here.
2. Run `python3 preprocess.py`.

Now you should be able to use this dataset for any experiment.

---

### Attributes (6): ###

* *com_id*: unique int.
* *timestamp*: time the comment was posted.
* *vid_id*: video the comment was posted on.
* *user_id*: user who posted the comment.
* *text*: content of the comment.
* *label*: 0 - not spam, 1 - spam.

---

### Basic Statistics ###

* *6,431,471* total comments; *481,334* spam comments (7.5%).
* *2,860,264* users; *177,542* spammers (6.2%).
* *6,407* videos; *6,340* spam videos (98.9%).