import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))
import appConfig
# import pandas as pd
import re
import numpy as np
import gensim
from sklearn.feature_extraction.text import CountVectorizer


#This function does some cleaning of the inputData.
#1. converts text to lower case
#2. Resolves apostrophes(') when used with question words(who|what|where|when|how)
#3. Removes stop words using the custom_stop_words added in the appConfig.py

def preprocessData(inputData):
    print(__name__ + ' : ' + 'Start preprocessData()')
    # inputData = [re.sub('\\\'s', 'is', q) for q in inputData]
    preprocessedData = []
    for q in inputData:
        q = q.lower()
        match = re.match('^(who|what|where|when|how)(\\s\\\'s)', q)
        if match:
            # print match.group(1)
            preprocessed_question = re.sub('^(who|what|where|when|how)(\\s\\\'s)', match.group(1) + ' is', q)
        else:
            # print 'No Match Found'
            preprocessed_question = q

        # preprocessed_question = re.sub('(`|\?|"|\'|\\\)', "", preprocessed_question)
        words = preprocessed_question.split()
        filtered_words = [w for w in words if w not in appConfig.custom_stop_words]
        filtered_q = " ".join(filtered_words)
        preprocessedData.append(filtered_q)
    # print(__name__ + '\tPreprocessed Data : ' + str(preprocessedData))
    # preprocessedDataFrame = pd.DataFrame(data={'question':preprocessedData})
    # preprocessedDataFrame.to_csv('output/preprocessedData.txt')
    print(__name__ + ' : ' + 'End preprocessData()')
    return preprocessedData

#This function calculates the average of the word-vectors of all the words in the cleaned text
#We check if every word is in the vocab of the word2vec model, if it is not, we don't consider that word.
def getAvgWordVector(wv_model, question):
    words = question.split()
    vector = np.zeros(shape=(1,300))

    nValidWords = 0
    # print(__name__ + 'getting average vector')
    for word in words:
        if word in wv_model.vocab:
            wordvector = wv_model[word]
            nValidWords += 1
            vector = np.add(vector,wordvector)
    # vectors = vectors.toarray()
    # q_vector = np.mean(vectors)
    # q_vector = np.zeros(shape=(1, 300))

    #averaging the word-vector over all the valid words in the text
    q_vector = vector/nValidWords

    return q_vector.tolist()

#This function returns the feature set of the inputData
#There are different modes available here to extract features



def getFeatures(inputData):
    questions = inputData
    # 1. bow - Bag of Words : with N_GRAM_MIN and N_GRAM_MAX limits
    if appConfig.FEATURE_EXTRACTION_MODE == 'bow':
        vectorizer = CountVectorizer(analyzer="word", tokenizer=None, ngram_range=(
        appConfig.N_GRAM_MIN, appConfig.N_GRAM_MAX), binary=True, min_df=1, max_features=125)
        data_features = vectorizer.fit_transform(questions)
        data_features = data_features.toarray()
        print(__name__ + '\tfeature shape : ' + str(data_features.shape))

    # 2. gwv - google word vectors : using the average vector of words in text
    if appConfig.FEATURE_EXTRACTION_MODE == 'gwv':
        wv_model = gensim.models.KeyedVectors.load_word2vec_format(appConfig.GOOGLE_WORD2VEC_MODEL, binary=True)
        print (__name__ + 'Google word2vec model loaded')
        data_features = [getAvgWordVector(wv_model,question)[0] for question in inputData]
        # print (__name__ + 'Data Features : ' + str(data_features))
        # outputNumbersFile = open(appConfig.OUTPUT_FOLDER + "/" + appConfig.CLASSIFIER_MODE + "_wv.txt", 'w')
        # outputNumbersFile.write(str(data_features))
        # outputNumbersFile.close()
        print(__name__ + '\tfeature shape : ' + str(len(data_features)))

    #3. doc2vec - Doc2Vec : custom trained or pretrained doc2vec model returning a single vector representing the whole document
    #This can be used only after running trainDoc2VecModel.py if using a custom trained model
    ############ PLEASE DONOT USE THIS MODE IN VERSION 1. ##############
    # if appConfig.FEATURE_EXTRACTION_MODE == 'doc2vec':
    #     doc2vec_model = gensim.models.KeyedVectors.load_word2vec_format(appConfig.DOC2VEC_MODEL, binary=True)
    #     print (__name__ + 'Custom doc2vec model loaded')
    #     data_features = [doc2vec_model[question] for question in inputData]
    #     print(__name__ + '\tfeature shape : ' + str(len(data_features)))

    return data_features
