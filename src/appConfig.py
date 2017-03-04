# Folders
MODELS_FOLDER = 'models'
DATA_FOLDER = 'data'
OUTPUT_FOLDER = 'output'

#Data location
DATA_FILE = 'LabelledData.txt'
UNLABELLED_DATA_FILE = 'TestData.txt'

#Fields in LabelledData.txt
ID_FIELD = 'id'
DATA_FIELD = 'question'
LABEL_FIELD = 'category'

#this is for splitting the labelled data into train and test
DATA_SPLIT=0.2

#all the labels in the data
LABELS = ['who', 'what', 'when', 'affirmation', 'unknown']

# bow-bag of words, gwv-google wordvectors,
# doc2vec - custom doc2vec model trained on ~6k questions(http://cogcomp.cs.illinois.edu/Data/QA/QC/train_5500.label)
############ PLEASE DONOT USE doc2vec MODE IN VERSION 1. ##############
FEATURE_EXTRACTION_MODE = 'bow'

# rf-random forests, lsvc - Linear SVC, gnb - Gaussian Naive Bayes, knn - k-nearest neighbours, lr - logistic regression, neuralN - neural network
CLASSIFIER_MODE = 'knn'

#some stop words which can be removed, used in preprocessing
custom_stop_words = ['the', 'a', 'an', 'and', 'of', 'from', 'in']

#arguments for count vectorizer
N_GRAM_MIN=1
N_GRAM_MAX=3

#argument for knn
N_NEAREST_NEIGHBOURS=10

#number of splits in the training data for cross-validation
CROSS_VALIDATION_FOLDS=10

#output file name
OUTPUT_FILE = FEATURE_EXTRACTION_MODE + '_' + CLASSIFIER_MODE + '_stopWordsRemoved'

#location of Google word2vec model binary file
GOOGLE_WORD2VEC_MODEL = MODELS_FOLDER + '/' + 'GoogleNews-vectors-negative300.bin'

#this is the location of the custom trained doc2vec model
DOC2VEC_MODEL= MODELS_FOLDER + '/' + 'doc2vec'+ '.bin'