#!/usr/bin/env python
# coding: utf-8



#import libraries
import stanza
import pickle
import pandas as pd
import numpy as np



#download stanza and load Armenian treebank
stanza.download('hy')
nlp = stanza.Pipeline('hy')



#open Armenian words and their English translations
armenian = pd.read_csv('Armenian.txt', encoding = 'utf-8', header = None, delimiter = '\t')
english = pd.read_csv('English.txt', encoding = 'utf-8', header = None, delimiter = '\t')

#select the first column of each
armenian = armenian[0]
english = english[0]



#remove more than one-word Armenian entries
labels = []
for i in range(len(armenian)):
    if len(armenian[i].split()) > 1:
        labels.append(i)
armenian = armenian.drop(labels)
armenian = armenian.reset_index(drop = True)
english = english.drop(labels)
english = english.reset_index(drop = True)



#change to lowercase and remove whitespace
for i in range(len(armenian)):
    armenian[i] = armenian[i].lower().strip()
    english[i] = english[i].lower().strip()
    
POS = []
    
#lemmatize armenian words and add POS tag
for i in range(len(armenian)):
    arm = nlp(armenian[i]).to_dict()
    armenian[i] = arm[0][0]['lemma']
    pos = arm[0][0]['upos']
    if pos == 'ADJ':
        POS.append('a')
    elif pos == 'ADV':
        POS.append('r')
    elif pos == 'NOUN':
        POS.append('n')
    elif pos == 'VERB':
        POS.append('v')
    else:
        POS.append('n/a')

#create the dataframe
columns = ['Armenian', 'English', 'POS']
translation = pd.DataFrame(columns = columns)
translation['Armenian'] = armenian
translation['English'] = english
translation['POS'] = POS



for i in range(len(translation)):
    if ',' in translation['English'][i]:
        translation['English'][i] = translation['English'][i].replace(',', ' ')
translation.head()



#read SentiWordNet
swn = pd.read_csv('SentiWordNet_3.0.0.txt', sep = '\t', encoding = 'utf8', dtype = {'ID':str})
swn.head()



for i in range(len(swn)):
    terms = swn['SynsetTerms'][i].split()
    s = ''
    for j in range(len(terms)):
        s = s + terms[j][:-2] + ' '
    swn['SynsetTerms'][i] = s[:-1]



#maximum length of words in the English translations
lengths = []
for i in range(len(translation)):
    lengths.append(len(translation['English'][i].split()))
max_length = max(lengths)



#create the necessary columns
columns = ['HWN_Offset', 'SWN_Offset', 'Armenian', 'English', 'Positive', 'Negative', 'Objective']
lexicon = pd.DataFrame(columns = columns)



def retrieve_scores(index):
    pos = swn['PosScore'][index]
    neg = swn['NegScore'][index]
    num = swn['ID'][index]
    return [np.array([pos, neg]), num]



#find the matches by first going through the words in synsets and then through the glosses
new = []
for i in range(len(translation)):
    words = translation['English'][i].split(',')
    n = len(words)
    found = False
    for j in range(n):
        if words[j] in swn['SynsetTerms'].values:
            ind = np.where(swn['SynsetTerms'].values == words[j])[0]
            for k in range(len(ind)):
                if swn['POS'][ind[k]] == translation['POS'][i]:
                    scores = retrieve_scores(ind[k])
                    found = True
                    new.append({'SWN_Offset': scores[1], 'Armenian': translation['Armenian'][i], 
                                'English': translation['English'][i], 'Positive': scores[0][0], 'Negative': scores[0][1], 
                                'Objective': 1 - np.sum(scores[0])})
    p = n
    while p > 0 and found == False:
        if ' '.join(words[:p]) in swn['Gloss'].values:
            ind = np.where(swn['Gloss'].values == ' '.join(words[:p]))[0]
            for j in range(len(ind)):
                if swn['POS'][ind[j]] == translation['POS'][i]:
                    scores = retrieve_scores(ind[j])
                    found = True
                    new.append({'SWN_Offset': scores[1], 'Armenian': translation['Armenian'][i], 
                                'English': translation['English'][i], 'Positive': scores[0][0], 'Negative': scores[0][1], 
                                'Objective': 1 - np.sum(scores[0])})
        p = p - 1
    if found == False:
        scores = [[np.nan, np.nan], np.nan]
        new.append({'SWN_Offset': scores[1], 'Armenian': translation['Armenian'][i], 'English': translation['English'][i], 
                    'Positive': scores[0][0], 'Negative': scores[0][1], 'Objective': 1 - np.sum(scores[0])})
lexicon = lexicon.append(new, ignore_index = True)



#check the percentage of nan entries in the lexicon dataframe
lexicon['Positive'].isna().sum() / len(lexicon) * 100



#drop the rows with nan scores, reset its index and drop the old one
lexicon = lexicon.dropna(subset = ['Positive', 'Negative', 'Objective'])
lexicon = lexicon.reset_index()
lexicon = lexicon.drop('index', axis = 1)



#save the dataframe as a pickle file
with open('translationScores.pickle', 'wb') as f:
    pickle.dump(lexicon, f)
    
#save the dataframe as a text file (for those interested to read the results)
with open('translationScores.txt', 'w', encoding = 'utf-8') as f:
    f.write(lexicon.to_string())

