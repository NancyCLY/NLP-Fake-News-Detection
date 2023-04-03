import torchtext as ttx
import torch
import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
# nltk.download('stopwords')

stops = set(stopwords.words("english"))
max_seq = 200
state = 0

def cleantext(string):
    text = string.lower().split()
    text = " ".join(text)
    text = re.sub(r"http(\S)+", ' ', text)
    text = re.sub(r"www(\S)+", ' ', text)
    text = re.sub(r"&", ' and ', text)
    text = text.replace('&amp', ' ')
    text = re.sub(r"[^0-9a-zA-Z]+", ' ', text)
    text = text.split()
    text = [w for w in text if not w in stops]
    return text


def glove_embedding(tweets, dim=50, max_len=200):
    global state
    state += 1
    if state % 10 == 0:
        print(state)
    if len(tweets) < max_len :
        tweets = tweets + ["<pad>"] * (max_seq - len(tweets))
    vec = ttx.vocab.GloVe(name='6B', dim=dim)
    ret = vec.get_vecs_by_tokens(tweets, lower_case_backup=True).numpy()
    return ret


train_data = pd.read_csv("../data/Constraint_English_Train - Sheet1.csv")
val_data = pd.read_csv("../data/Constraint_English_Val - Sheet1.csv")


train_data['tweet'] = train_data['tweet'].map(lambda x: cleantext(x))
val_data['tweet'] = val_data['tweet'].map(lambda x: cleantext(x))


train_data['token'] = train_data['tweet'].map(lambda x: glove_embedding(x))
val_data['token'] = val_data['tweet'].map(lambda x: glove_embedding(x))

train_data["avg_len"] = train_data["tweets"].map(lambda x: len(x))
val_data["avg_len"] = val_data["tweets"].map(lambda x: len(x))
avg_length = train_data["avg_len"].mean()
train_data["avg_len"] = train_data["avg_len"] - avg_length
val_data["avg_len"] = val_data["avg_len"] - avg_length

train_data.to_pickle("../data/train_len.pkl")
val_data.to_pickle("../data/val_len.pkl")
