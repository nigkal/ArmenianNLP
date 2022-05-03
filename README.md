# ArmenianNLP
This repository is for a project to bring NLP resources for the Armenian language, including sentiment and emotion lexicons and an Armenian BERT.

The folders in this repository are divided into the different models used for sentiment analysis and emotion recognition. The models are: baseline model, ensemble-learning model, BERT model, and Dict-BERT model. Each folder in this repository includes the following:
- Jupyter notebooks where each evaluation is done with an the entire code.
- Data folder that includes the files needed while evaluating the lexicons and the BERT models.
- An explanation of how to do the evaluations on your own should you want to evaluate the lexicons and the models on your own.

Below, you will find the results we got doing sentiment analysis and emotion recognition on our lexicons and models.

## Sentiment Analysis
| | Baseline Model | SVM | LR | Ensemble-Learning Model | ArmBERT | Dict-BERT |
| --- | --- | --- | --- | --- | --- | --- |
| **Accuracy** | 0.698314 | 0.775510 | 0.775510 | 0.771960 | 0.825199 | 0.827861 |
| **Recall** | 0.795847 | 0.955017 | 0.982698 | 0.967704 | 0.946943 | 0.953863 |
| **Precision** | 0.808909 | 0.794625 | 0.781651 | 0.785580 | 0.844650 | 0.843017 |
| **F-measure** | 0.802325 | 0.867469 | 0.870720 | 0.867183 | 0.892876 | 0.895021 |

## Emotion Recognition
| | Baseline Model | SVM | LR | Ensemble-Learning Model | ArmBERT | Dict-BERT |
| --- | --- | --- | --- | --- | --- | --- |
| **Accuracy** | 0.149819 | 0.590252 | 0.521660 | 0.635379 | 0.774368 | 0.770758 |
| **Recall** | 0.254257 | 0.381862 | 0.212244 | 0.373244 | 0.774368 | 0.587237 | 
| **Precision** | 0.277453 | 0.495213 | 0.173497 | 0.410971 | 0.771354 | 0.554299 |
| **F-measure** | 0.138642 | 0.403657 | 0.176465 | 0.376579 | 0.772258 | 0.568813 |


Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: http://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
