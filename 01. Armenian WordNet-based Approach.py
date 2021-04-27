#!/usr/bin/env python
# coding: utf-8


#import libraries
import re
import pickle
import numpy as np
import pandas as pd



#read the files necessary: English wordnet, Armenian wordnet, SentiWordNet
ewn = pd.read_csv('wn-wikt-eng.tab', sep = '\t', encoding = 'utf8')
hwn = pd.read_csv('wn-wikt-hye.tab', sep = '\t', encoding = 'utf8')
swn = pd.read_csv('SentiWordNet_3.0.0.txt', sep = '\t', encoding = 'utf8', dtype = {'ID':'str'})



#change swn IDs to match English wordnet ID format
swn['Number'] = swn['ID']
for i in range(len(swn)):
    swn['Number'][i] = swn['Number'][i] + '-' + swn['POS'][i]



#add the three columns: English words, Positive, Negative, Objective scores to hwn
hwn['English'] = np.nan
hwn['Positive'] = np.nan
hwn['Negative'] = np.nan
hwn['Objective'] = np.nan

#change column names and drop some unneeded columns
hwn = hwn.rename(columns = {'# Wiktionary':'ID', 'http://wiktionary.org/':'Armenian'})
hwn = hwn.drop(['hye', 'CC BY-SA'], axis = 1)



#get the scores from swn
for i in range(len(swn)):
    score = hwn.loc[hwn['ID'] == (swn['Number'][i])]
    if score.empty == False:
        index = list(score.index)
        for ind in index:
            hwn['English'][ind] = swn['SynsetTerms'][i]
            hwn['Positive'][ind] = swn['PosScore'][i]
            hwn['Negative'][ind] = swn['NegScore'][i]
            hwn['Objective'][ind] = 1 - (swn['PosScore'][i] + swn['NegScore'][i])



#check if any of the entries has no score
indexes = hwn['Positive'].index[hwn['Positive'].apply(np.isnan)]
print(indexes)



#rename columns of hwn
hwn = hwn.rename(columns = {'ID':'HWN_Offset'})
hwn['SWN_Offset'] = hwn['HWN_Offset']
for i in range(len(hwn)):
    hwn['SWN_Offset'][i] = hwn['SWN_Offset'][i][:-2]
hwn = hwn[['HWN_Offset', 'SWN_Offset', 'Armenian', 'English', 'Positive', 'Negative', 'Objective']]



#change how the words appear in the English column of hwn (to be more presentable)
for i in range(len(hwn)):
    s = hwn['English'][i]
    s = s.replace('#', '')
    s = re.sub(r'\d+', '', s)
    s = s.replace(' ', ',')
    hwn['English'][i] = s



#save the dataframe as a pickle file
with open('hwn.pickle', 'wb') as f:
    pickle.dump(hwn, f)
    
#save the dataframe as a text file (for those interested to read the results)
with open('hwn.txt', 'w', encoding = 'utf-8') as f:
    f.write(hwn.to_string())

