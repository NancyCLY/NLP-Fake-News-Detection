import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
from joblib import dump, load
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
import numpy as np
import itertools

nltk.download('stopwords')

stops = set(stopwords.words("english"))


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
    text = " ".join(text)
    return text


train = pd.read_csv("../data/Constraint_English_Train - Sheet1.csv")
val = pd.read_csv("../data/Constraint_English_Val - Sheet1.csv")

train['tweet'] = train['tweet'].map(lambda x: cleantext(x))
val['tweet'] = val['tweet'].map(lambda x: cleantext(x))


def print_metrices(pred, true):
    print(confusion_matrix(true, pred))
    print(classification_report(true, pred, ))
    print("Accuracy : ", accuracy_score(pred, true))
    print("Precison : ", precision_score(pred, true, average='weighted'))
    print("Recall : ", recall_score(pred, true, average='weighted'))
    print("F1 : ", f1_score(pred, true, average='weighted'))



pipeline = Pipeline([
    ('bow', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('c', LinearSVC())
])
fit = pipeline.fit(train['tweet'], train['label'])
print('SVM')
print('val:')
pred = pipeline.predict(val['tweet'])
print_metrices(pred, val['label'])
