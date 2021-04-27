#!/usr/bin/env python
# coding: utf-8



#import libraries
import stanza
import json
import pandas as pd
import numpy as np
import os.path
import re
import string
import nltk
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences



#download stanza and load Armenian treebank
stanza.download('hy')
nlp = stanza.Pipeline('hy')



#open the data
train = pd.read_csv('Train_Emotion.csv', encoding = 'utf-8')
test = pd.read_csv('Test_Emotion.csv', encoding = 'utf-8')



#retrieve the input and target
X_train = train['Text']
y_train = train['Label']
X_test = test['Text']
y_test = test['Label']



#open the lexicon
with open('finalScore_depeche.pickle', 'rb') as f:
    scores = pickle.load(f)



#count the number of unique lemmas
lemmas = scores['Armenian'].unique()



#associate emotion score to each unique lemma
emotion = {}
for i in range(len(lemmas)):
    word = lemmas[i]
    ind = np.where(scores['Armenian'] == word)[0]
    afraid, amused, angry, annoyed, dont_care, happy, inspired, sad = 0, 0, 0, 0, 0, 0, 0, 0
    for j in range(len(ind)):
        afraid += scores['Afraid'][ind[j]]
        amused += scores['Amused'][ind[j]]
        angry += scores['Angry'][ind[j]]
        annoyed += scores['Annoyed'][ind[j]]
        dont_care += scores['Dont_Care'][ind[j]]
        happy += scores['Happy'][ind[j]]
        inspired += scores['Inspired'][ind[j]]
        sad += scores['Sad'][ind[j]]
    afraid = afraid / len(ind)
    amused = amused / len(ind)
    angry = angry / len(ind)
    annoyed = annoyed / len(ind)
    dont_care = dont_care / len(ind)
    happy = happy / len(ind)
    inspired = inspired / len(ind)
    sad = sad / len(ind)
    emotion[word] = np.array([afraid, amused, angry, annoyed, dont_care, happy, inspired, sad])



#get the Armenian stopwords
stopwords = pd.read_csv('stop-words.csv', header = None)
stopwords = list(stopwords[0])



#function to remove emojis from text
def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags = re.UNICODE)
    return emoji_pattern.sub(r'', string)



#function to preprocess text
def preprocess(info):
    strings = []
    for i in info.index:
        info[i] = info[i].lower() #lowercase
        info[i] = re.sub(r'\d+', '', info[i]) #numbers
        info[i] = re.sub(r'[^\w\s]', '', info[i]) #punctuation
        info[i] = info[i].strip() #whitespace
        info[i] = remove_emoji(info[i]) #emojis
        info[i] = re.sub(r'\s?[a-zA-Z]+\.?[a-zA-Z]*', '', info[i]) #english
        tokenized = info[i].split()
        filtered = [w for w in tokenized if not w in stopwords]
        lemmatized = [nlp(w).to_dict()[0][0]['lemma'] for w in filtered]
        strings.append(lemmatized)
    info = pd.Series(strings, index = info.index)
    return info



#preprocess the texts
X_train = preprocess(X_train)
X_test = preprocess(X_test)



#count maximum number of words in X_train
count = []
for i in X_train.index:
    count.append(len(X_train[i]))
maximum = max(count)


#open Armenian glove, change entries to lemmas, and concatenate the emotion scores
words = []
embedding_index = {}

with open("glove.hy.300.txt", 'r', encoding = 'utf8') as f:
    for line in f:
        values = line.split()
        modified = re.sub(r'[^\w\s]', '', values[0])
        if modified != '':
            word = nlp(modified).to_dict()[0][0]['lemma']
        else:
            word = values[0]
        words.append(word)
        coefs = np.asarray(values[1:], 'float32')
        embedding_index[word] = coefs



#save the dictionary as pickle
with open('Embedding_dict_2.pickle', 'wb') as f:
    pickle.dump(embedding_index, f)

with open('Words_2.pickle', 'wb') as f:
    pickle.dump(words, f)



#transform the data into a matrix to use for training
t = Tokenizer()
t.fit_on_texts(words)
vocab_size = len(t.word_index) + 1
encoded_docs1 = t.texts_to_sequences(X_train)
encoded_X_train = pad_sequences(encoded_docs1, maxlen = maximum, padding = 'post')
encoded_docs2 = t.texts_to_sequences(X_test)
encoded_X_test = pad_sequences(encoded_docs2, maxlen = maximum, padding = 'post')



total, count = 0, 0
embedding_matrix = np.zeros((vocab_size, 58))
for word, i in t.word_index.items():
    total += 1
    embedding_vector = embedding_index.get(word)
    emotion_vector = emotion.get(word)
    if embedding_vector is not None and len(embedding_vector) == 50:
        embedding_matrix[i][:50] = embedding_vector
    if emotion_vector is not None:
        count += 1
        embedding_matrix[i][50:] = emotion_vector



training_matrix = np.zeros((len(X_train), maximum, 58))
testing_matrix = np.zeros((len(X_test), maximum, 58))

for i in range(len(X_train)):
    for j in range(maximum):
        training_matrix[i][j] = embedding_matrix[encoded_X_train[i][j]]
train = training_matrix.reshape(len(X_train), maximum * 58)

for i in range(len(X_test)):
    for j in range(maximum):
        testing_matrix[i][j] = embedding_matrix[encoded_X_test[i][j]]
test = testing_matrix.reshape(len(X_test), maximum * 58)



#functions for hyperparameter tuning, training and testing without and with cross validation
def search(clf, parameters):
    classifier = RandomizedSearchCV(clf, parameters, cv = 5)
    clf_search = classifier.fit(train, list(y_train))
    return clf_search.best_estimator_



def train_predict(clf):
    clf.fit(train, list(y_train))
    y = clf.predict(test)
    acc = accuracy_score(list(y_test), y)
    f1 = f1_score(list(y_test), y, average = 'weighted')
    rec = recall_score(list(y_test), y, average = 'weighted')
    pre = precision_score(list(y_test), y, average = 'weighted')
    return clf, y, acc, f1, rec, pre



training = pd.DataFrame(train, index = y_train.index)



def train_predict_cv(clf):
    kf = StratifiedKFold(10)
    for m, _ in kf.split(training, y_train):
        clf.fit(training.loc[training.index.intersection(m)], y_train.loc[y_train.index.intersection(m)])
    y = clf.predict(test)
    acc = accuracy_score(y_test, y)
    f1 = f1_score(y_test, y, average = 'weighted')
    rec = recall_score(y_test, y, average = 'weighted')
    pre = precision_score(y_test, y, average = 'weighted')
    return clf, y, acc, f1, rec, pre



sv_parameters = {'tol': [0.0001, 0.001, 0.01, 0.1, 1],
                 'C': np.logspace(-4, 4, 20),
                 'kernel': ['poly', 'rbf', 'sigmoid']}
sv = SVC()
sv_best = search(sv, sv_parameters)
print(sv_best)



sv_trained, sv_y, sv_accuracy, sv_f1, sv_recall, sv_precision = train_predict(sv_best)
print('Accuracy:', sv_accuracy)
print('F-measure:', sv_f1)
print('Recall:', sv_recall)
print('Precision', sv_precision)



sv_trained, sv_y, sv_accuracy, sv_f1, sv_recall, sv_precision = train_predict_cv(sv_best)
print('Accuracy:', sv_accuracy)
print('F-measure:', sv_f1)
print('Recall:', sv_recall)
print('Precision', sv_precision)

