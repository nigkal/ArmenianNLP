# ArmenianNLP
This repository is for a project to bring NLP resources for the Armenian language, including sentiment and emotion lexicons and an Armenian BERT.

To access the data used for this project, visit the Data folder. This folder contains all the files used for creating the Armenian sentiment and emotion lexicons. The codes used for this project are in the Jupyter Notebooks folder. This folder contains the notebooks which include all the codes used to implement and run all the processes. The Outputs folder contain the files we collected from the codes, such as the lexicons, training sets, etc.

Details about each folder is found in the corresponding folder.

## Baseline Models
The sentiment and emotion lexicons are evaluated using a baseline model. These models take the sentiment or emotion scores of the tokens in the text from the lexicons and add the ones under the same category. For sentiment analysis, the negative score is subtracted from the positive score, and the sentiment is predicted based on the result. For emotion recognition, the emotion with the highest score is used to predict the emotion of the text.

For access to the full snipet of the code, see [baseline_model.py](baseline_model.py).

## Ensemble-learning Model
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

## BERT Model
The Armenian BERT is trained using the following BERT-base configuration:

``` python
bert_base_config = {
  "attention_probs_dropout_prob": 0.1, 
  "directionality": "bidi", 
  "hidden_act": "gelu", 
  "hidden_dropout_prob": 0.1, 
  "hidden_size": 768, 
  "initializer_range": 0.02, 
  "intermediate_size": 3072, 
  "max_position_embeddings": 512, 
  "num_attention_heads": 12, 
  "num_hidden_layers": 12, 
  "pooler_fc_size": 768, 
  "pooler_num_attention_heads": 12, 
  "pooler_num_fc_layers": 3, 
  "pooler_size_per_head": 128, 
  "pooler_type": "first_token_transform", 
  "type_vocab_size": 2, 
  "vocab_size": 64000
}
```

To evaluate the BERT model, download the model from [here]().

Run the following code to load the BERT model for sentiment and emotion analyses.

``` python
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

#for sentiment
model = TFBertForSequenceClassification.from_pretrained('./bert_model', from_pt = True)
tokenizer = BertTokenizer.from_pretrained('./bert_model')

#for emotion
model = TFBertForSequenceClassification.from_pretrained('./bert_model', from_pt = True, num_labels = 5)
tokenizer = BertTokenizer.from_pretrained('./bert_model')
```

To transform your data into BERT-readable format, run the following code.

```python
def convert_data_to_examples(train, test, DATA_COLUMN, LABEL_COLUMN): 
    train_InputExamples = train.apply(lambda x: InputExample(guid=None, #globally unique ID for bookkeeping, unused in this case
                                                          text_a = x[DATA_COLUMN], 
                                                          text_b = None,
                                                          label = x[LABEL_COLUMN]), axis = 1)

    validation_InputExamples = test.apply(lambda x: InputExample(guid=None, #globally unique ID for bookkeeping, unused in this case
                                                          text_a = x[DATA_COLUMN], 
                                                          text_b = None,
                                                          label = x[LABEL_COLUMN]), axis = 1)
  
    return train_InputExamples, validation_InputExamples
  
def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    features = [] #will hold InputFeatures to be converted later

    for e in examples:
        input_dict = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens = True,
            max_length = max_length, #truncates if len(s) > max_length
            return_token_type_ids = True,
            return_attention_mask = True,
            pad_to_max_length = True, #pads to the right by default
            truncation = True
        )

        input_ids, token_type_ids, attention_mask = (input_dict['input_ids'],
            input_dict['token_type_ids'], input_dict['attention_mask'])

        features.append(
            InputFeatures(
                input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids, label = e.label
            )
        )

    def gen():
        for f in features:
            yield (
                {
                    'input_ids': f.input_ids,
                    'attention_mask': f.attention_mask,
                    'token_type_ids': f.token_type_ids,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({'input_ids': tf.int32, 'attention_mask': tf.int32, 'token_type_ids': tf.int32}, tf.int64),
        (
            {
                'input_ids': tf.TensorShape([None]),
                'attention_mask': tf.TensorShape([None]),
                'token_type_ids': tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )


DATA_COLUMN = #column of the text data
LABEL_COLUMN = #column of the label data
```

Fine-tuning can be performed using the following:

```python
train_InputExamples, validation_InputExamples = convert_data_to_examples(train, dev, DATA_COLUMN, LABEL_COLUMN)

train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
train_data = train_data.shuffle(100).batch(32).repeat(2)

validation_data = convert_examples_to_tf_dataset(list(validation_InputExamples), tokenizer)
validation_data = validation_data.batch(32)

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 3e-5), 
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), 
              metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

model.fit(train_data, epochs = 1, validation_data = validation_data, verbose = 2)
```

Finally, use the following code for prediction and evaluation.

```python
pred_sentences = #column of the testing text data
tf_batch = tokenizer(list(pred_sentences), max_length = 128, padding = True, truncation = True, return_tensors = 'tf')
tf_outputs = model(tf_batch)
tf_predictions = tf.nn.softmax(tf_outputs[0], axis = -1)
labels = #labels used for sentiment or emotion
label = tf.argmax(tf_predictions, axis = 1)
label = label.numpy()
predictions = pd.Series(label, index = test.index)
```

## Dict-BERT Model
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

Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: http://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
