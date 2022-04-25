# Jupyter Notebooks
The following files are found in this folder. You can download and run the codes in these files. Note that you need to have the specific files from the Data
and the Output folders to run each notebook when needed.

Here are the descriptions of the files (with their outputs, if any) in the Jupyter Notebooks folder:
1. **01. Armenian WordNet-based Approach.ipynb:** This links the Armenian wordnet to the English SentiWordNet. The output is the first part of the Armenian sentiment lexicon.
2. **02. English Translation-based Approach.ipynb:** This links the Armenian-English dictionary to the English SentiWordNet. The output is the second part of the Armenian sentiment lexicon.
3. **03. Combining the Two Approaches.ipynb:** This combines the first and second part of the Armenian sentiment lexicon. The output is the entire Armenian sentiment lexicon.
4. **03b. Emotion Lexicon.ipynb:** This links the Armenian sentiment lexicon to EmoWordNet. The output is the Armenian emotion lexicon.
5. **04. Evaluating the Lexicon EMOTION ENSEMBLE.ipynb:** This runs emotion recognition (using an ensemble-learning model) on the emotion data set using the Armenian emotion lexicon.
6. **04. Evaluating the Lexicon EMOTION.ipynb:** This runs emotion recognition (using a baseline model) on the emotion data set using the Armenian emotion lexicon.
7. **04. Evaluating the Lexicon SENTIMENT COMPETITION.ipynb:** This runs sentiment analysis (using a baseline model) on the sentiment data set using the multilingual sentiment lexicon.
8. **04. Evaluating the Lexicon SENTIMENT ENSEMBLE COMPETITION.ipynb:** This runs sentiment analysis (using an ensemble-learning model) on the sentiment data set using the multilingual sentiment lexicon.
9. **04. Evaluating the Lexicon SENTIMENT ENSEMBLE.ipynb:** This runs sentiment analysis (using an ensemble-learning model) on the sentiment data set using the Armenian sentiment lexicon.
10. **04. Evaluating the Lexicon SENTIMENT.ipynb:** This runs sentiment analysis (using a baseline model) on the sentiment data set using the Armenian sentiment lexicon.
11. **05. Monolingual BERT Emotions PROBABILITIES.ipynb:** This runs emotion recognition on the emotion data set using Armenian BERT and an adaptive emotion dictionary.
12. **05. Monolingual BERT Emotions.ipynb:** This runs emotion recognition on the emotion data set using Armenian BERT.
13. **05. Monolingual BERT Sentiment PROBABILITIES.ipynb:** This runs sentiment analysis on the sentiment data set using Armenian BERT and an adaptive sentiment dictioanry.
14. **05. Monolingual BERT Sentiment.ipynb:** This runs sentiment analysis on the sentiment data set using Armenian BERT.
15. **05. Multilingual BERT Emotions.ipynb:** This runs emotion recognition on the emotion data set using Multilingual BERT.
16. **05. Multilingual BERT Sentiment.ipynb:** This runs sentiment analysis on the sentiment data set using Multilingual BERT.
17. **BERT.ipynb:** This is used to create the Armenian BERT using the Armenian corpus.
18. **Splitting Up the Emotion Lexicon.ipynb:** This creates the adaptive emotion dictionary. The outputs are the files that are linked to different emotions.
19. **Splitting Up the Sentiment Lexicon.ipynb:** This creates the adaptive sentiment dictionary. The outputs are the files that are linked to different sentiments, along with the degree adverbs and negative words.

The following libraries are needed to run the codes:
1. Bert
2. Json
3. Keras
4. Numpy
5. Pandas
6. Pickle
7. Re
8. Stanza
9. Scikit-learn
10. Tensorflow
11. Transformers
