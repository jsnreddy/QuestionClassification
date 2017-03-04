# QuestionClassification

##Introduction
The objective here is to classify questions into 5 classes - 
what, who, when, affirmation, unknown.
We are using around 1.5k labelled data to train the models. 

##Prerequisites
Please go through the requirements.txt and other_requirements.txt for prerequisites to run the code.

##Installation
* Clone the repository
* run pip install -r requirements.txt
* Follow steps in other_requirements.txt to get all other dependencies.

### appConfig.py
The configurations need to be set in this file. The configurations include - 
1. What kind of feature extraction.
2. Which classifier to train.
3. Parameters to be set in feature extraction, training(eg. ngrams, knn)

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

## Improvements to be done
* The present implementation of using the average wordvector(average of all wordvectors of words in the preprocessed text) is not optimal. This can be improved to use a better heuristic. Doc2Vec would be an improvement over this implementation.
* The present doc2vec module is not completely ready, testing is not possible right now. It can also be built on a bigger dataset to give better results.
* Usage of sense2vec to extract features. This allows us to leverage syntactic features like POS tag of the words in the sentence also in generating the features representing the question.
* The model parameters used to train can be tweaked further to get better results(eg, no.of hidden layers in neural network, learning rate, kernel in SVM, number of estimators in random forest).
