import torch
import pandas as pd
import re
from nltk.corpus import stopwords
from torch.utils.data.dataset import Dataset
from matplotlib import pyplot as plt
import os
import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

nltk.download('stopwords')
stops = set(stopwords.words("english"))
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []
best_acc = 0
epochs = 30
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


corpus = pd.concat([train_data['ori_text'], val_data['ori_text']])
tfidf_corpus = np.array(tfidf.fit_transform(corpus).todense())
train_tfidf = tfidf_corpus[:6417]
val_tfidf = tfidf_corpus[6417:]


class DenseDataset(Dataset):
    """
    You need to inherit nn.Module and
    overwrite __getitem__ and __len__ methods.
    """

    def __init__(self,
                 len_list, label_list):
        self.label_list = label_list
        self.len_list = len_list

    def __getitem__(self, index):
        length = self.len_list[index]
        length = torch.Tensor(length)
        if self.label_list[index] == "real":
            label = torch.from_numpy(np.array([0])).long()
        elif self.label_list[index] == "fake":
            label = torch.from_numpy(np.array([1])).long()
        else:
            label = torch.from_numpy(np.array([2])).long()
        return length, label

    def __len__(self):
        return len(self.label_list)


class SimpleDense(torch.nn.Module):
    def __init__(self, in_dim=16202, feature_dim=10, out_dim=2, dropout=0):
        super(SimpleDense, self).__init__()
        self.linear1 = torch.nn.Linear(in_dim, feature_dim)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(feature_dim, out_dim)
        self.softmax = torch.nn.Softmax(dim=1)
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, src):
        output = self.linear1(src)
        output = self.drop(output)
        output = self.relu(output)
        output = self.linear2(output)
        output = self.drop(output)
        output = self.softmax(output)
        return output


def train(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    avg_loss = 0
    avg_acc = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        target = target.squeeze(1)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        avg_loss += loss.item()
        optimizer.step()
        _, preds = output.max(1)
        avg_acc += preds.eq(target).sum().float()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    avg_loss /= len(train_loader)
    avg_acc /= len(train_loader.dataset)
    print('\nTrain set: Batch average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(avg_loss, avg_acc * 100))
    train_loss_list.append(avg_loss)
    train_acc_list.append(avg_acc.cpu())


def test(model, val_loader, criterion, device, epoch):
    model.eval()
    avg_loss = 0
    avg_acc = 0
    test_confusion_matrix = np.zeros((2, 2))
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            target = target.squeeze(1)
            output = model(data)
            loss = criterion(output, target)
            avg_loss += loss.item()
            _, preds = output.max(1)
            avg_acc += preds.eq(target).sum().float()
            test_confusion_matrix = conf_matrix(test_confusion_matrix, preds, target)
    avg_loss /= len(val_loader)
    avg_acc /= len(val_loader.dataset)
    print('\nTest set: Batch average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(avg_loss, avg_acc * 100))
    print(test_confusion_matrix)
    test_loss_list.append(avg_loss)
    test_acc_list.append(avg_acc.cpu())
    global best_acc
    if avg_acc > best_acc:
        torch.save(model.state_dict(), "../model/dense.pth")
        best_acc = avg_acc
        print("======Saving model======")


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using', device)
    model = SimpleDense(dropout=0.2).to(device)
    train_dataset = DenseDataset(train_tfidf, train_data['label'])
    val_dataset = DenseDataset(val_tfidf, val_data['label'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        train(model, train_loader, optimizer, criterion, device, epoch)
        test(model, val_loader, criterion, device, epoch)
    print(str(best_acc))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label='Training Loss')
    plt.plot(test_loss_list, label='Test Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_list, label='Training Accuracy')
    plt.plot(test_acc_list, label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.legend()
    if not os.path.exists("../pic"):
        os.mkdir("../pic")
    plt.savefig("../pic/dense.png")

def main_test():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using', device)
    model = SimpleDense(dropout=0.25).to(device)
    model.load_state_dict(torch.load("../model/dense.pth"))
    train_dataset = DenseDataset(train_tfidf, train_data['label'])
    val_dataset = DenseDataset(val_tfidf, val_data['label'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    test(model, val_loader, criterion, device, 0)




if __name__ == '__main__':
    main()
    # main_test()
