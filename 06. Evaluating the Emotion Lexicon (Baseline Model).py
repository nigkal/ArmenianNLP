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
part_one = pd.read_csv('Train_Emotion.csv', encoding = 'utf-8')
part_two = pd.read_csv('Test_Emotion.csv', encoding = 'utf-8')
data = pd.concat([part_one, part_two], ignore_index = True)



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



#open the lexicon
with open('finalScore_depeche.pickle', 'rb') as f:
    scores = pickle.load(f)



#function to transform the text into a vector of size 3: [Positive, Negative, Objective]
def transform(x):
    total = 0
    count = 0
    score = np.zeros((len(x), 8))
    for i in x.index:
        s = np.zeros((len(x[i]), 8))
        for j in range(len(x[i])):
            total += 1
            if x[i][j] in scores['Armenian'].values:
                count += 1
                ind = np.where(scores['Armenian'] == x[i][j])[0]
                afraid, amused, angry, annoyed, dont_care, happy, inspired, sad = 0, 0, 0, 0, 0, 0, 0, 0
                for k in range(len(ind)):
                    afraid += scores['Afraid'][ind[k]]
                    amused += scores['Amused'][ind[k]]
                    angry += scores['Angry'][ind[k]]
                    annoyed += scores['Annoyed'][ind[k]]
                    dont_care += scores['Dont_Care'][ind[k]]
                    happy += scores['Happy'][ind[k]]
                    inspired += scores['Inspired'][ind[k]]
                    sad += scores['Sad'][ind[k]]
                s[j][0] = afraid / len(ind)
                s[j][1] = amused / len(ind)
                s[j][2] = angry / len(ind)
                s[j][3] = annoyed / len(ind)
                s[j][4] = dont_care / len(ind)
                s[j][5] = happy / len(ind)
                s[j][6] = inspired / len(ind)
                s[j][7] = sad / len(ind)
        t = s.sum(axis = 0)
        h = list(x.index).index(i)
        score[h] = t
    return [score, count, total]



#transform the input data
transformed = transform(X)
text = pd.DataFrame(transformed[0], index = X.index)
text.columns = ['Afraid', 'Amused', 'Angry', 'Annoyed', 'Dont_Care', 'Happy', 'Inspired', 'Sad']



#baseline model to evaluate the lexicon
results = []
for i in range(len(text)):
    output = list(text[['Afraid', 'Amused', 'Angry', 'Annoyed', 'Dont_Care', 'Happy', 'Inspired', 'Sad']].loc[0].values)
    result = output.index(max(output))
    new = sorted(output)
    while result == 1 or result == 3 or result == 4:
        new = new[:-1]
        result = output.index(max(new))
    if result == 0:
        results.append('fear')
    elif result == 2:
        results.append('anger')
    elif result == 5:
        results.append('joy')
    elif result == 6:
        results.append('surprise')
    elif result == 7:
        results.append('sadness')
results = pd.Series(results, index = y.index)



#check performance
acc = accuracy_score(y, results)
f1 = f1_score(y, results, average = 'weighted')
rec = recall_score(y, results, average = 'weighted')
pre = precision_score(y, results, average = 'weighted')
print('Accuracy:', acc)
print('F-measure:', f1)
print('Recall:', rec)
print('Precision:', pre)


