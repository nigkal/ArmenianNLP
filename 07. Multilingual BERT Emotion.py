#!/usr/bin/env python
# coding: utf-8



#import libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score



#get the model
model = TFBertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels = 5)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')



#read the data
t = pd.read_csv('Train_Emotion.csv', encoding = 'utf-8')
test = pd.read_csv('Test_Emotion.csv', encoding = 'utf-8')



#create a polarity column and change the targets accordingly
def change_label(data):
    data['Polarity'] = np.nan
    for i in range(len(data)):
        if data['Label'][i] == 'anger':
            data['Polarity'][i] = 0
        elif data['Label'][i] == 'fear':
            data['Polarity'][i] = 1
        elif data['Label'][i] == 'joy':
            data['Polarity'][i] = 2
        elif data['Label'][i] == 'sadness':
            data['Polarity'][i] = 3
        else:
            data['Polarity'][i] = 4
    data = data.drop('Label', axis = 1)
    return data



t = change_label(t)
test = change_label(test)



#split data into train, validate, test
train, dev = train_test_split(t, test_size = 0.1)



#define the functions
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


DATA_COLUMN = 'Text'
LABEL_COLUMN = 'Polarity'



#train the model on the dataset
train_InputExamples, validation_InputExamples = convert_data_to_examples(train, dev, DATA_COLUMN, LABEL_COLUMN)

train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
train_data = train_data.shuffle(100).batch(32).repeat(2)

validation_data = convert_examples_to_tf_dataset(list(validation_InputExamples), tokenizer)
validation_data = validation_data.batch(32)

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 3e-5, epsilon = 1e-08, clipnorm = 1.0), 
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), 
              metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

model.fit(train_data, epochs = 10, validation_data = validation_data, verbose = 2)



#predict
pred_sentences = test['Text']
tf_batch = tokenizer(list(pred_sentences), max_length = 128, padding = True, truncation = True, return_tensors = 'tf')
tf_outputs = model(tf_batch)
tf_predictions = tf.nn.softmax(tf_outputs[0], axis = -1)
labels = [0, 1, 2, 3, 4]
label = tf.argmax(tf_predictions, axis = 1)
label = label.numpy()
predictions = pd.Series(label, index = test.index)
print('Accuracy:', accuracy_score(test['Polarity'], predictions))
print('F-measure:', f1_score(test['Polarity'], predictions, average = 'weighted'))
print('Recall:', recall_score(test['Polarity'], predictions, average = 'weighted'))
print('Precision:', precision_score(test['Polarity'], predictions, average = 'weighted'))

