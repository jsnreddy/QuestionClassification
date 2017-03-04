# Folders
MODELS_FOLDER = 'models'
DATA_FOLDER = 'data'
OUTPUT_FOLDER = 'output'

TRAIN_FILE = 'LabelledData.txt'
TEST_FILE = 'TestData.txt'
ID_FIELD = 'id'
DATA_FIELD = 'question'
LABEL_FIELD = 'category'

DATA_SPLIT=0.2

LABELS = ['who', 'what', 'when', 'affirmation', 'unknown']

# bow-bag of words, gwv-google wordvectors, doc2vec - custom doc2vec model trained on ~6k questions(http://cogcomp.cs.illinois.edu/Data/QA/QC/train_5500.label)
FEATURE_EXTRACTION_MODE = 'gwv'

# rf-random forests, lsvc - Linear SVC, gnb - Gaussian Naive Bayes, knn - k-nearest neighbours, lr - logistic regression, neuralN - neural network
CLASSIFIER_MODE = 'lsvc'

custom_stop_words = ['the', 'a', 'an', 'and', 'of', 'from', 'in']

N_GRAM_MIN=1
N_GRAM_MAX=3
N_NEAREST_NEIGHBOURS=10

OUTPUT_FILE = FEATURE_EXTRACTION_MODE + '_' + CLASSIFIER_MODE + '_stopWordsRemoved'

GOOGLE_WORD2VEC_MODEL = MODELS_FOLDER + '/' + 'GoogleNews-vectors-negative300.bin'

#doc2vec_window5_dmpv, doc2vec_window5_dbow, doc2vec_window15_dbow
DOC2VEC_MODEL= MODELS_FOLDER + '/' + 'doc2vec_window5_dmpv'+ '.bin'