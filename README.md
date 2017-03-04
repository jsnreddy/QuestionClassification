# QuestionClassification

## Introduction
The objective here is to classify questions into 5 classes - 
what, who, when, affirmation, unknown.
We are using around 1.5k labelled data to train the models. 

## Prerequisites
Please go through the requirements.txt and other_requirements.txt for prerequisites to run the code.

## Installation
* Clone the repository
* run pip install -r requirements.txt
* Follow steps in other_requirements.txt to get all other dependencies.

## Scripts
### appConfig.py
The configurations need to be set in this file. The configurations include - 
###### What kind of feature extraction.
    a. bow - Bag of Words(with N_GRAM_MIN and N_GRAM_MAX limits)
    b. gwv - Google Word2Vec(using the average vector of words in text)
###### Which classifier to train.
    a. rf - Random Forest
    b. lsvc - Linear Support Vector Classifier
    c. gnb - Gaussian Naive Bayes
    d. lr - Logistic Regression(One vs All)
    e. knn - K-Nearest Neighbours
    f. neuralN - Neural Network
###### Parameters to be set in feature extraction, training(eg. ngrams, knn)
    a. N_GRAM_MIN and N_GRAM_MAX - for bag of words features
    b. N_NEAREST_NEIGHBOURS - for knn classifier
    c. DATA_SPLIT - for splitting labelled data between train and test
    d. custom_stop_words - for cleaning text
    e. CROSS_VALIDATION_FOLDS - number of splits in training data for cross validation
###### Filepaths to data, models and output files
###### Fields in LabelledData.txt

### MainController.py 
This is the main script to run training and classfication. Once the feature extraction and classification mode is configured in the appConfig.py, we can run this script to see the results.

### FeatureExtractor.py
This script contains functions to preprocess the data and extract features.

### Learner.py
Training data is processed, and model is trained and saved using this class.

### Classifier.py
Testing data is processed, saved model is loaded and predictions are generated. This class also writes the predictions into a file in output folder and the results numbers - accuracy, precision, recall and confusion matrix into another file in output folder.

### WitClassifier.py
This script is using Wit.ai API to train a model with all the labelled data. This class can be run independently and output can be seen for individual cases.

### trainDoc2VecModel.py
This class is to train our own custom doc2vec model. The model is being trained on ~6k reviews(very less to make a good model) and model is saved into models folder as a binary.

## Running Tests
* After selecting the feature extraction mode and classifier mode in the appConfig, run the MainController.py to run the whole pipeline - read data, preprocess, train, cross-validate and test.
* Run WitClassifier.py independently giving the intended example in the code. This produces results from a model which is trained from the data in LabelledData.txt (1483 tagged questions).
* To build a custom Doc2Vec model, provide the dataFile and run the trainDoc2VecModel.y script.The script defaults to 'TestData.txt' in 'data' folder which has ~6k questions. This would generate a model and save in 'models' folder.(This model can be used for feature extraction once the bug fix is done regarding loading the Doc2Vec model).

## Results
Comparative Analysis of all classifiers used with both Bag of words and Word2Vec features is available at - https://docs.google.com/spreadsheets/d/1PVVe6NhjzcN0LJyeg7uj1dE-Kc0-p3r9pDfflyG37KI/edit#gid=0

## Improvements to be done
* The present implementation of using the average wordvector ( average of the wordvectors of all words in the preprocessed text ) is not optimal. This can be improved to use a better heuristic. Training a custom word2vec model for this domain could give better results. Doc2Vec would be a further improvement over this implementation.
* The present doc2vec module is not completely ready, testing is not possible right now. There is a bug in loading the model for feature extraction which throwing an exception due to file encoding, which needs to be fixed. Further improvement on this would be to build the model on a bigger dataset to give better results.
* Usage of sense2vec to extract features. This allows us to leverage syntactic features like POS tag of the words in the sentence also in generating the features representing the question.
* The model parameters used to train can be tweaked further to get better results ( Eg: no.of hidden layers in neural network, learning rate, kernel in SVM, number of estimators in random forest).
* WordNet would allow us to incorporate lexical features by using the synsets.
* ConceptNet would provide a meaning of the words, helping us to incorporate semantic features.
* Further improvement would be using dependency parse of the sentences combined with POS tags to extract the features. This allows us to capture the relation between the words in the sentence without just considering them as independent occurrences.
