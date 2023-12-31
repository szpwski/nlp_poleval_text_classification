# Harmful vs Non-harmful Text Classification

Presentation with the approach and achieved results can be seen in [here](https://szpwski.github.io/nlp_poleval_text_classification/presentations/presentation_eng.pdf).

### Problem

Cyberbullying, phishing, harmful and hurtful content are a growing and disturbing trend in social media and online communications. This phenomenon poses serious challenges as customers are at risk of losing data and financial resources.

### Solution

Detecting and responding to these threats by classifying harmful content is a potential solution to protect customers.  For text classification tasks, natural language processing (NLP) combined with machine learning (ML) plays a key role.

### Methods and tools 

In order to create an effective tool, typical NLP techniques such as n-grams and TF-IDF, and machine learning models such as SVM, NB and HerBERT were used. The implementation of these solutions was carried out in the freely available Python programming language.

### Data

The tool was created based on a common dataset from the evaluation campaign for NLP tools PolEval 2019. The dataset includes neutral and harmful content written by users (tweets). These include cyberbullying, hate speech and related phenomena. Dataset comes from the [6th task of PolEval 2019](http://2019.poleval.pl/index.php/tasks/task6).

### Results

| Model                | ACC   | Precision  | Recall   | F1   |
|----------------------|-------|-------|-------|-------|
| **EDA + TF-IDF + NB**    | **0.793** | **0.339** | **0.575** | **0.427** |
| SMOTE + TF-IDF + NB  | 0.791 | 0.338 | 0.582 | 0.427 |
| EDA + TF-IDF + SVM   | 0.842 | 0.3   | 0.134 | 0.186 |
| SMOTE + TF-IDF + SVM | 0.871 | 0.692 | 0.067 | 0.122 |
| **HerBERT**             | **0.888** | **0.634** | **0.388** | **0.481** |

### Next steps

- check other techniques for data augmentation
- seek out different models 
- try out ensemble learning
- perform more advanced feature engineering and incorporate those features into the model