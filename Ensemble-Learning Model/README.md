# Ensemble-Learning Model
For this, you first need to get the Armenian GloVe from [here](https://at.ispras.ru/owncloud/index.php/s/eXNpONfB09TBpgI).

Use this code to read the file, get the embeddings, and create the embedding matrix. The example uses the 300-dimensional GloVe representation. When you use another dimension, please amend the code by replacing all 300's with the corresponding dimension number.

``` python
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

words = []
embedding_index = {}

with open(r'glove.hy.300.txt', 'r', encoding = 'utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        words.append(word)
        coefs = np.asarray(values[1:], 'float32')
        embedding_index[word] = coefs

t = Tokenizer()
t.fit_on_texts(words)
vocab_size = len(t.word_index) + 1
encoded_docs1 = t.texts_to_sequences(X_train)
encoded_X_train = pad_sequences(encoded_docs1, maxlen = maximum, padding = 'post')
encoded_docs2 = t.texts_to_sequences(X_test)
encoded_X_test = pad_sequences(encoded_docs2, maxlen = maximum, padding = 'post')

embedding_matrix = np.zeros((vocab_size, 300))
for word, i in t.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None and len(embedding_vector) == 300:
        embedding_matrix[i][:300] = embedding_vector
```

Next, you need to create both training and testing matrices for the models to train and make predictions. The example below shows how to prepare these matrices using the sentiment lexicon. You can use the same approach for emotion recognition.

``` python
training_matrix_1 = np.zeros((len(X_train), maximum, 300))
training_matrix_2 = np.zeros((len(X_train), maximum, 3)) # for emotion, change this to 5
testing_matrix_1 = np.zeros((len(X_test), maximum, 300))
testing_matrix_2 = np.zeros((len(X_test), maximum, 3)) # for emotion, change this to 5

for i in range(len(X_train)):
    for j in range(maximum):
        training_matrix_1[i][j] = embedding_matrix[encoded_X_train[i][j]]
    for k in range(maximum):
        training_matrix_2[i][k] = train_score[i][k]
train_1 = training_matrix_1.reshape(len(X_train), maximum * 300)
train_2 = training_matrix_2.reshape(len(X_train), maximum * 3) # for emotion, change this to 5
train = np.concatenate((train_1, train_2), axis = 1)

for i in range(len(X_test)):
    for j in range(maximum):
        testing_matrix_1[i][j] = embedding_matrix[encoded_X_test[i][j]]
    for k in range(maximum):
        testing_matrix_2[i][k] = test_score[i][k]
test_1 = testing_matrix_1.reshape(len(X_test), maximum * 300)
test_2 = testing_matrix_2.reshape(len(X_test), maximum * 3) # for emotion, change this to 5
test = np.concatenate((test_1, test_2), axis = 1)
```

Later, support-vector machine (SVM) and logistic regression (LR) are used to build a stacking classifier in an ensemble-learning model. Based on the data sets that we used, here are the hyperparameter results that we got from performing hyperparameter tuning.

### Sentiment Analysis
| | Regularization | Tolerance | Penalty |
| --- | --- | --- | --- |
| SVM | 545.5594781168514 | 0.1 | N/A |
| LR | 29.763514416313132 | 2.121212121212121 | 'none' |

### Emotion Recognition
| | Regularization | Tolerance |
| --- | --- | --- |
| SVM | 3792.690190732246 | 0.0001 |
| LR | 4.281332398719396 | 5.555555555555555 |

After creating these models, use the following functions to train SVM and LR respectively (clf is the classifier with the hyperparameters). 

``` python
training = pd.DataFrame(train, index = y_train.index)

def train_cv_embedding(clf):
    kf = StratifiedKFold(10)
    for m, _ in kf.split(training[range(maximum * 300)], y_train):
        clf.fit(training[range(maximum * 300)].loc[training[range(maximum * 300)].index.intersection(m)], y_train.loc[y_train.index.intersection(m)])
    y = clf.predict(test[:, range(maximum * 300)])
    
def train_cv_sentiment(clf): #for emotion, replace 303 with 305 in all
    kf = StratifiedKFold(10)
    for m, _ in kf.split(training[range(maximum * 300, maximum * 303)], y_train): 
        clf.fit(training[range(maximum * 300, maximum * 303)].loc[training[range(maximum * 300, maximum * 303)].index.intersection(m)], y_train.loc[y_train.index.intersection(m)])
    y = clf.predict(test[:, range(maximum * 300, maximum * 303)])
```

Finally, use this to create, train, and run the stacking classifier.

``` python
from sklearn.ensemble import StackingClassifier

estimators = [('sv', sv_embedding_trained), ('lr', lr_sentiment_trained)]
final = StackingClassifier(estimators)
final.fit(training, y_train)
y_pred = final.predict(test)
```
