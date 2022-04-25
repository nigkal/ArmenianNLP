The following files are found in this folder. Note that anywhere, specific file is read in the notebooks that are not part of the outputs (our products),
the files can be linked to this folder.

Here are the descriptions of the files in the Data folder:
1. **Armenian:** This contains the Armenian lemmas in the Armenian-English dictionary.
2. **EmotionTest:** This contains the testing data used for emotion recognition. It has been translated to Armenian. The original data set is SemEval-2018 Task 1.
3. **EmotionTrain:** This contains the training data used for emotion recognition. It has been translated to Armenian. The original data set is SemEval-2018 Task 1.
4. **EmoWordNet:** This is the emotion lexicon in English, which is used to get the emotions of the Armenian lemmas for the Armenian emotion lexicon.
5. **English:** This contains the English translations in the Armenian-English dictionary.
6. **negative_words_hy:** This contains the negative words of the multilingual sentiment lexicon.
7. **positive_words_hy:** This contains the positive words of the multilingual sentiment lexicon.
8. **Sentiment_TestB:** This contains the testing data used for sentiment analysis. It has been translated to Armenian. The original data set is SemEval-2017 Task 4.
9. **Sentiment_TrainB:** This contains the training data used for sentiment analysis. It has been translated to Armenian. The original data set is SemEval-2017 Task 4.
10. **SentiWordNet_3.0.0:** This is the sentiment lexicon in English, which is used to get the sentiments of the Armenian lemmas for the Armenian sentiment lexicon.
11. **stop-words:** This contains the stopwords for the Armenian language.
12. **wn-wikt-eng.tab:** This is the English WordNet.
13. **wn-wikt-hye-tab:** This is the Armenian WordNet.

Below, you can find the files to use based on the task you are performing:
1. For creating the Armenian sentiment lexicon: Armenian, English, SentiWordNet_3.0.0, wn-wikt-eng.tab, wn-wikt-hye-tab
2. For creating the Armenian emotion lexicon: EmoWordNet
3. For sentiment analysis: Sentiment_TestB, Sentiment_TrainB, stop-words(, negative_words_hy, positive_words_hy)
4. For emotion recognition: EmotionTest, EmotionTrain, stop-words
