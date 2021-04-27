#!/usr/bin/env python
# coding: utf-8



#import libraries
import pandas as pd
import numpy as np
import pickle



#open ArSEL and sentiment lexicon
arsel = pd.read_csv('ArSEL.txt', encoding = 'utf-8', delimiter = ';', header = 0, dtype = str)

with open('finalScores.pickle', 'rb') as f:
    scores = pickle.load(f)



#fix the AFRAID column
arsel = arsel.rename(columns = {'Confidence###AFRAID':'AFRAID'})
for i in range(len(arsel)):
    arsel['AFRAID'][i] = arsel['AFRAID'][i].split('###')[1]



#create the needed columns
scores['Afraid'] = np.zeros(len(scores))
scores['Amused'] = np.zeros(len(scores))
scores['Angry'] = np.zeros(len(scores))
scores['Annoyed'] = np.zeros(len(scores))
scores['Dont_Care'] = np.zeros(len(scores))
scores['Happy'] = np.zeros(len(scores))
scores['Inspired'] = np.zeros(len(scores))
scores['Sad'] = np.zeros(len(scores))



#make offset resemble those in ArSEL
for i in range(len(scores)):
    l = scores['SWN_Offset'][i]
    if len(l) != 0 and len(l) < 8 and type(l) == str:
        if len(l) == 1:
            l = '0000000' + l
        elif len(l) == 2:
            l = '000000' + l
        elif len(l) == 3:
            l = '00000' + l
        elif len(l) == 4:
            l = '0000' + l
        elif len(l) == 5:
            l = '000' + l
        elif len(l) == 6:
            l = '00' + l
        else:
            l = '0' + l
    scores['SWN_Offset'][i] = l



#get the emotion scores from ArSEL
for i in range(len(scores)):
    print(i)
    l = scores['SWN_Offset'][i]
    index = arsel.index[arsel['EWN_OFFSET'] == l].tolist()
    if len(index) != 0:
        scores['Afraid'][i] = arsel['AFRAID'][index[0]]
        scores['Amused'][i] = arsel['AMUSED'][index[0]]
        scores['Angry'][i] = arsel['ANGRY'][index[0]]
        scores['Annoyed'][i] = arsel['ANNOYED'][index[0]]
        scores['Dont_Care'][i] = arsel['DONT_CARE'][index[0]]
        scores['Happy'][i] = arsel['HAPPY'][index[0]]
        scores['Inspired'][i] = arsel['INSPIRED'][index[0]]
        scores['Sad'][i] = arsel['SAD'][index[0]]


#save the lexicon
with open('finalScore_depeche.pickle', 'wb') as f:
    pickle.dump(scores, f)


#save the lexicon as a text file (for those interested to read the results)
with open('finalScore_depech.txt', 'w', encoding = 'utf-8') as f:
    f.write(scores.to_string())
