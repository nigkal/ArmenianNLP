#!/usr/bin/env python
# coding: utf-8



#import libraries
import pickle
import numpy as np
import pandas as pd



#load the files needed
with open('translationScores_new.pickle', 'rb') as f:
    translation = pickle.load(f)

with open('hwn.pickle', 'rb') as f:
    hwn = pickle.load(f)



#concatenate hwn translation dataframe
new = pd.concat([hwn, translation], ignore_index = True)
new = new.fillna('') #fill any NaNs



#save the dataframe as a pickle file
with open('finalScores.pickle', 'wb') as f:
    pickle.dump(new, f)
    
#save the dataframe as a text file (for those interested to read the results)
with open('finalScores.txt', 'w', encoding = 'utf-8') as f:
    f.write(new.to_string())

