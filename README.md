# QuestionClassification

The objective here is to classify questions into 5 classes - 
what, who, when, affirmation, unknown

# appConfig.py
The configurations need to be set in this file. The configurations include - 
1. What kind of feature extraction.
2. Which classifier to train.
3. Parameters to be set in feature extraction, training(eg. ngrams, knn)

# MainController.py 
This is the main script to run training and classfication. Once the feature extraction and classification mode is configured in the appConfig.py, we can run this script to see the results.

# FeatureExtractor.py
This class contains functions to preprocess the data and extract features.

# Learner.py
Training data is processed, and model is trained and saved using this class.

# Classifier.py
Testing data is processed, saved model is loaded and predictions are generated. This class also writes the predictions into a file in output folder and the results numbers - accuracy, precision, recall and confusion matrix into another file in output folder.

# WitClassifier.py
This is using Wit.ai API to train a model with all the labelled data. This class can be run independently and output can be seen for individual cases.

# trainDoc2VecModel.py
This class is to train our own custom doc2vec model. The model is being trained on ~6k reviews(very less to make a good model) and model is saved into models folder as a binary.
