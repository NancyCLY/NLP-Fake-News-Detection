import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import itertools
import torch
import re
from nltk.corpus import stopwords
from torch.utils.data.dataset import Dataset
import math
from transformer import Transformer_for_classify, TransformerDataset
from dense import DenseDataset, SimpleDense

stops = set(stopwords.words("english"))
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []
best_acc = 90
epochs = 30
max_seq = 200
tfidf = TfidfVectorizer()

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

def cleantext_connect(string):
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

def normalize(series):
    return (series - series.mean()) / math.sqrt(series.var())

def conf_matrix(mat, output, label):
    output = output.cpu().numpy()
    label = label.cpu().numpy()
    for i in range(len(output)):
        mat[output[i]][label[i]] += 1
    return mat
#
#
# def glove_embedding(tweets, dim=50, max_len=200):
#     if len(tweets) < max_len :
#         tweets = tweets + ["<pad>"] * (max_seq - len(tweets))
#     vec = ttx.vocab.GloVe(name='6B', dim=dim)
#     ret = vec.get_vecs_by_tokens(tweets, lower_case_backup=True)
#     return ret
#
#
# train_data = pd.read_csv("../data/Constraint_English_Train - Sheet1.csv")[:1000]
# val_data = pd.read_csv("../data/Constraint_English_Val - Sheet1.csv")[:200]
#
#
#
# for i in range(len(train_data['tweet'])):
#     if len(train_data['tweet'][i]) > max_seq:
#         train_data.drop(i)
# for i in range(len(val_data['tweet'])):
#     if len(val_data['tweet'][i]) > max_seq:
#         val_data.drop(i)
#
# train_data['token'] = train_data['tweet'].map(lambda x: glove_embedding(x))
# val_data['token'] = val_data['tweet'].map(lambda x: glove_embedding(x))

train_data = pd.read_pickle("../data/train.pkl")
val_data = pd.read_pickle("../data/val.pkl")

train_ori = pd.read_csv("../data/Constraint_English_Train - Sheet1.csv")
val_ori = pd.read_csv("../data/Constraint_English_Val - Sheet1.csv")
# train_ori['tweet'] = train_ori['tweet'].map(lambda x: cleantext(x))
# val_ori['tweet'] = val_ori['tweet'].map(lambda x: cleantext(x))

train_ori["ori_len"] = train_ori["tweet"].map(lambda x: len(x))
val_ori["ori_len"] = val_ori["tweet"].map(lambda x: len(x))



train_data["ori_len"] = normalize(train_ori["ori_len"])
val_data["ori_len"] = normalize(val_ori["ori_len"])
train_data["ori_text"] = train_ori["tweet"].map(lambda x: cleantext_connect(x))
val_data["ori_text"] = val_ori["tweet"].map(lambda x: cleantext_connect(x))

train_data["len"] = train_data["token"].map(lambda x: len(x))
sorted_train = train_data.sort_values(by='len').reset_index()
train_data = sorted_train[:-3]
val_data["len"] = val_data["token"].map(lambda x: len(x))
sorted_val = val_data.sort_values(by='len').reset_index()
val_data = sorted_val[:-3]


def get_feature(tensor, feature_list):
    tensor = tensor.cpu().numpy()
    for batch in tensor:
        feature_list.append(batch)

def test_feature(model, val_loader, criterion, device, epoch):
    model.eval()
    avg_loss = 0
    avg_acc = 0
    feature_list = []
    label_list = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            target = target.squeeze(1)
            get_feature(target, label_list)
            output = model(data)
            get_feature(output, feature_list)
            loss = criterion(output, target)
            avg_loss += loss.item()
            _, preds = output.max(1)
            avg_acc += preds.eq(target).sum().float()
    avg_loss /= len(val_loader)
    avg_acc /= len(val_loader.dataset)
    # print('\nTest set: Batch average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(avg_loss, avg_acc * 100))
    return feature_list, label_list


def svm(train_cat_input, train_label, val_cat_input, val_label):

    def print_metrices(pred, true):
        print(confusion_matrix(true, pred))
        print(classification_report(true, pred, ))
        print("Accuracy : ", accuracy_score(pred, true))
        print("Precison : ", precision_score(pred, true, average='weighted'))
        print("Recall : ", recall_score(pred, true, average='weighted'))
        print("F1 : ", f1_score(pred, true, average='weighted'))

    pipeline = Pipeline([
        ('c', LinearSVC())
    ])
    fit = pipeline.fit(train_cat_input, train_label)
    print('SVM')
    print('val:')
    pred = pipeline.predict(val_cat_input)
    print_metrices(pred, val_label)

def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using', device)

    corpus = pd.concat([train_data['ori_text'], val_data['ori_text']])
    tfidf_corpus = np.array(tfidf.fit_transform(corpus).todense())
    train_tfidf = tfidf_corpus[:6417]
    val_tfidf = tfidf_corpus[6417:]

    t_model = Transformer_for_classify().to(device)
    t_model.load_state_dict(torch.load("../model/transformer.pth"))
    t_train_dataset = TransformerDataset(train_data['token'], train_data['label'], train_data['ori_len'])
    t_val_dataset = TransformerDataset(val_data['token'], val_data['label'], val_data['ori_len'])
    t_train_loader = torch.utils.data.DataLoader(t_train_dataset, batch_size=16, shuffle=False)
    t_val_loader = torch.utils.data.DataLoader(t_val_dataset, batch_size=16, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    t_train_feature, t_train_label = test_feature(t_model, t_train_loader, criterion, device, 1)
    t_train_feature, t_train_label = np.array(t_train_feature), np.array(t_train_label)
    t_val_feature, t_val_label = test_feature(t_model, t_val_loader, criterion, device, 1)
    t_val_feature, t_val_label = np.array(t_val_feature), np.array(t_val_label)

    d_model = SimpleDense(dropout=0.2).to(device)
    d_model.load_state_dict(torch.load("../model/dense.pth"))
    d_train_dataset = DenseDataset(train_tfidf, train_data['label'])
    d_val_dataset = DenseDataset(val_tfidf, val_data['label'])
    d_train_loader = torch.utils.data.DataLoader(d_train_dataset, batch_size=16, shuffle=False)
    d_val_loader = torch.utils.data.DataLoader(d_val_dataset, batch_size=16, shuffle=False)
    d_train_feature, d_train_label = test_feature(d_model, d_train_loader, criterion, device, 1)
    d_train_feature, d_train_label = np.array(d_train_feature), np.array(d_train_label)
    d_val_feature, d_val_label = test_feature(d_model, d_val_loader, criterion, device, 1)
    d_val_feature, d_val_label = np.array(d_val_feature), np.array(d_val_label)

    # train_len = train_data['ori_len'].to_numpy().reshape(-1, 1)
    # val_len = val_data['ori_len'].to_numpy().reshape(-1, 1)

    train_cat = np.concatenate([t_train_feature, d_train_feature], axis=1)
    train_label = np.concatenate([t_train_label], axis=0)
    val_cat = np.concatenate([t_val_feature, d_val_feature], axis=1)
    val_label = np.concatenate([t_val_label], axis=0)


    # svm(t_train_feature, train_label, t_val_feature, val_label)
    svm(train_cat, train_label, val_cat, val_label)

if __name__ == '__main__':
    main()