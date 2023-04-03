# NLP-Fake-News-Detection
# Background
Fake news detection is still one major concern in post-pandemic era. As the technological barrier is way lower than before, everyone can publish we-media contents. Detecting fake news is therefore much more important currently. 
This gives rise to our incentive to construct and train the model to automatically detect the fake news.
# Dataset
The train set has 6420 entries of news. 
The test set has 2110 entries of news. 
Both train and test set has a distribution of roughly 50% real and 50% fake to avoid bias and overfit. 
# Data Preprocessing
First, we will remove the website information and use regular expression to extract the related words in lower case as data preprocessing before embedding. 
# Embedding
GloVe: combine the advantages of word2vec and count based. 
TF-IDF
# Models
traditional machine learning: SVM
neural network model: Transformer
