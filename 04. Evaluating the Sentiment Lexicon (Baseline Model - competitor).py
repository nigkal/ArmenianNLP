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
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score



#download stanza and load Armenian treebank
stanza.download('hy')
nlp = stanza.Pipeline('hy')



#open the data
data = pd.read_csv('Data.csv', encoding = 'utf-8')
data.head()



#retrieve the input and target
X = data['Text']
y = data['Label']



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
X = preprocess(X)



#open the lexicons
positives = pd.read_csv('positive_words_hy.txt', header = None)
negatives = pd.read_csv('negative_words_hy.txt', header = None)



positives = positives.rename(columns = {0: 'Lemma'})
negatives = negatives.rename(columns = {0: 'Lemma'})



#give positive score of 1 to positive words and negative score of 1 to negative scores
pos_positive = []
pos_negative = []
for i in range(len(positives)):
    pos_positive.append(1)
    pos_negative.append(0)
positives['Positive'] = pos_positive
positives['Negative'] = pos_negative

neg_positive = []
neg_negative = []
for i in range(len(negatives)):
    neg_positive.append(0)
    neg_negative.append(1)
negatives['Positive'] = neg_positive
negatives['Negative'] = neg_negative



scores = pd.concat([positives, negatives], ignore_index = True)



#function to transform the text into a vector of size 3: [Positive, Negative, Objective]
def transform(x):
    total = 0
    count = 0
    score = np.zeros((len(x), 2))
    for i in x.index:
        s = np.zeros((len(x[i]), 2))
        for j in range(len(x[i])):
            total += 1
            if x[i][j] in scores['Lemma'].values:
                count += 1
                ind = np.where(scores['Lemma'] == x[i][j])[0]
                pos, neg = 0, 0
                for k in range(len(ind)):
                    pos += scores['Positive'][ind[k]]
                    neg += scores['Negative'][ind[k]]
                s[j][0] = pos / len(ind)
                s[j][1] = neg / len(ind)
        t = s.sum(axis = 0)
        h = list(x.index).index(i)
        score[h] = t
    return [score, count, total]



#transform the input data
transformed = transform(X)
text = pd.DataFrame(transformed[0], index = X.index)
text.columns = ['Positive', 'Negative']



#success rate
transformed[1] / transformed[2] * 100



#change the target values to (0, 1) to reflect (negative, positive)
real = np.zeros(len(y))
for i in range(len(y)):
    if y[i] == 'positive':
        real[i] = 1
    else:
        real[i] = 0
real = pd.Series(real, index = y.index)



#baseline model to evaluate the lexicon
results = []
for i in range(len(text)):
    result = text['Positive'][i] - text['Negative'][i]
    if result > 0:
        results.append(1)
    elif result <= 0:
        results.append(0)
results = pd.Series(results, index = y.index)



#check performance
acc = accuracy_score(real, results)
f1 = f1_score(real, results)
rec = recall_score(real, results)
pre = precision_score(real, results)
print('Accuracy:', acc)
print('F-measure:', f1)
print('Recall:', rec)
print('Precision:', pre)

