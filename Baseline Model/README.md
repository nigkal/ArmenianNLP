# Baseline Model
The sentiment and emotion lexicons are evaluated using a baseline model. These models take the sentiment or emotion scores of the tokens in the text from the lexicons and add the ones under the same category. For sentiment analysis, the negative score is subtracted from the positive score, and the sentiment is predicted based on the result. For emotion recognition, the emotion with the highest score is used to predict the emotion of the text.

Use the following function to extract the sentiments from the sentiment lexicon and create the sentiment vector of each sentence in the data set.

```python
def transform(x):
    score = np.zeros((len(x), 3))
    for i in x.index:
        s = np.zeros((len(x[i]), 3))
        for j in range(len(x[i])):
            if x[i][j] in scores['Armenian'].values:
                ind = np.where(scores['Armenian'] == x[i][j])[0]
                pos, neg, obj = 0, 0, 0
                for k in range(len(ind)):
                    pos += scores['Positive'][ind[k]]
                    neg += scores['Negative'][ind[k]]
                    obj += scores['Objective'][ind[k]]
                s[j][0] = pos / len(ind)
                s[j][1] = neg / len(ind)
                s[j][2] = obj / len(ind)
        t = s.sum(axis = 0)
        h = list(x.index).index(i)
        score[h] = t
    return score
```

Similar approach may be used to get the emotion scores of each sentence in the data set.

```python
def transform(x):
    score = np.zeros((len(x), 5))
    for i in x.index:
        s = np.zeros((len(x[i]), 5))
        for j in range(len(x[i])):
            if x[i][j] in scores['Armenian'].values:
                ind = np.where(scores['Armenian'] == x[i][j])[0]
                afraid, angry, happy, inspired, sad = 0, 0, 0, 0, 0, 0, 0, 0
                for k in range(len(ind)):
                    afraid += scores['Afraid'][ind[k]]
                    angry += scores['Angry'][ind[k]]
                    happy += scores['Happy'][ind[k]]
                    inspired += scores['Inspired'][ind[k]]
                    sad += scores['Sad'][ind[k]]
                s[j][0] = afraid / len(ind)
                s[j][1] = angry / len(ind)
                s[j][2] = happy / len(ind)
                s[j][3] = inspired / len(ind)
                s[j][4] = sad / len(ind)
        t = s.sum(axis = 0)
        h = list(x.index).index(i)
        score[h] = t
    return score
```
