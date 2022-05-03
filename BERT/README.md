# BERT Model
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

To evaluate the BERT model, download the model from [here](https://mailaub-my.sharepoint.com/:f:/g/personal/nhk19_mail_aub_edu/Eurzo_S3iZtDgkfbAK_FvYQBMoVDhdBH7oSzR-0NRAsjjw?e=bDWgDm).

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

To transform your data into BERT-readable format, run the following functions.

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
