# Dict-BERT Model
For this model, you'll need to use the contribution of each word in the sentiment or emotion lexicon. You can use the contribution files in the folder and make changes to the contribution of each word without changing the words (which are collected from the lexicons). If you're using the same data set, then you don't need to make changes to the files. Here is the code used to count and calculate the contribution of the sentiment or emotion words in our data set.

```python
#after placing the entire text data into a string
#you can use the separate emotions for emotion recognition instead of positive and negative lists
positive_counts = []
negative_counts = []

for i in range(len(positive_words)):
    positive_counts.append(string.count(positive_words[i]))

for i in range(len(negative_words)):
    negative_counts.append(string.count(negative_words[i]))
    
positive_contributions = []
negative_contributions = []

for i in range(len(positive_counts)):
    x = positive_counts[i]
    positive_contributions.append(1 / (1 + np.exp(-x)))

for i in range(len(negative_counts)):
    x = negative_counts[i]
    negative_contributions.append(1 / (1 + np.exp(-x)))
```

Afterwards, you need to calculate the probability ratio and set a threshold. Retrieve all the indexes in your testing data set that show a probability ratio below the set threshold. Use the following functions to calculate the positive/negative sentiment score. You can amend it to perform similar calculations for emotion.

``` python
def found_something(ind, words, window, adverb):
    a = ind - 1
    found_a = False
    count_a = 0
    while a >= 0 and found_a == False and count_a <= window:
        if words[a] in adverb:
            found_a = True
        a = a - 1
        count_a = count_a + 1
    b = ind + 1
    found_b = False
    count_b = 0
    while b < len(data[i]) and found_b == False and count_b <= window:
        if words[b] in adverb:
            found_b = True
        b = b + 1
        count_b = count_b + 1
    return found_a, found_b

def sentiment_score(data, positive_contribution, negative_contribution, positive_adverb, negative_adverb):
  positive_sentiment = []
  negative_sentiment = []

  for i in low_index:
      words = data[i]
      pos_words = []
      neg_words = []
      pos_scores = np.array([])
      neg_scores = np.array([])

      for j in range(len(words)):
          if words[j] in positive_contribution['Word'].values:
              pos_ind = np.where(positive_contribution['Word'] == words[j])[0]
              pos_cont = positive_contribution['Contribution'][pos_ind]
              pos_words.append([words[j], pos_cont])
          elif words[j] in negative_contribution['Word'].values:
              neg_ind = np.where(negative_contribution['Word'] == words[j])[0]
              neg_cont = negative_contribution['Contribution'][neg_ind]
              neg_words.append([words[j], neg_cont])

      for k in range(len(pos_words)):
          word = pos_words[k][0]
          c = pos_words[k][1]
          ind = data[i].index(word)
          pos_found = found_something(ind, words, 4, positive_adverb)
          if pos_found[0] == True or pos_found[1] == True:
              g = 2
          else:
              g = 1
          neg_found = found_something(ind, words, 3, negative_adverb)
          if neg_found[0] == True or neg_found[1] == True:
              f = -1
          else:
              f = 1
          pos_scores = np.append(pos_scores, g * f * c)

      for l in range(len(neg_words)):
          word = neg_words[l][0]
          c = neg_words[l][1]
          ind = data[i].index(word)
          pos_found = found_something(ind, words, 4, positive_adverb)
          if pos_found[0] == True or pos_found[1] == True:
              g = 2
          else:
              g = 1
          neg_found = found_something(ind, words, 3, negative_adverb)
          if neg_found[0] == True or neg_found[1] == True:
              f = -1
          else:
              f = 1
          neg_scores = np.append(neg_scores, g * f * c)

      positive_sentiment.append(np.sum(pos_scores))
      negative_sentiment.append(np.sum(neg_scores))

  return [positive_sentiment, negative_sentiment]
```
