import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))
import appConfig
# import pandas as pd
import re
import numpy as np
import gensim
from sklearn.feature_extraction.text import CountVectorizer


def preprocessData(inputData):
    print(__name__ + ' : ' + 'Start preprocessData()')
    # inputData = [re.sub('\\\'s', 'is', q) for q in inputData]
    preprocessedData = []
    for q in inputData:
        q = q.lower()
        match = re.match('^(who|what|where)(\\s\\\'s)', q)
        if match:
            # print match.group(1)
            preprocessed_question = re.sub('^(who|what|where)(\\s\\\'s)', match.group(1) + ' is', q)
        else:
            # print 'No Match Found'
            preprocessed_question = q

        # preprocessed_question = re.sub('(`|\?|"|\'|\\\)', "", preprocessed_question)
        words = preprocessed_question.split()
        filtered_words = [w for w in words if w not in appConfig.custom_stop_words]
        filtered_q = " ".join(filtered_words)
        preprocessedData.append(filtered_q)
    print(__name__ + 'Preprocessed Data : ' + str(preprocessedData))
    # preprocessedDataFrame = pd.DataFrame(data={'question':preprocessedData})
    # preprocessedDataFrame.to_csv('output/preprocessedData.txt')
    print(__name__ + ' : ' + 'End preprocessData()')
    return preprocessedData

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
    q_vector = vector/nValidWords

    return q_vector.tolist()


def getFeatures(inputData):
    questions = inputData
    if appConfig.FEATURE_EXTRACTION_MODE == 'bow':
        vectorizer = CountVectorizer(analyzer="word", tokenizer=None,ngram_range=(appConfig.N_GRAM_MIN,appConfig.N_GRAM_MAX),binary=True, min_df=1, max_features=125)
        data_features = vectorizer.fit_transform(questions)
        data_features = data_features.toarray()
        print(__name__ + '\tfeature shape : ' + str(data_features.shape))

    if appConfig.FEATURE_EXTRACTION_MODE == 'gwv':
        wv_model = gensim.models.KeyedVectors.load_word2vec_format(appConfig.GOOGLE_WORD2VEC_MODEL, binary=True)
        print (__name__ + 'Google word2vec model loaded')
        data_features = [getAvgWordVector(wv_model,question)[0] for question in inputData]
        # print (__name__ + 'Data Features : ' + str(data_features))
        # outputNumbersFile = open(appConfig.OUTPUT_FOLDER + "/" + appConfig.CLASSIFIER_MODE + "_wv.txt", 'w')
        # outputNumbersFile.write(str(data_features))
        # outputNumbersFile.close()
        print(__name__ + '\tfeature shape : ' + str(len(data_features)))

    # if appConfig.FEATURE_EXTRACTION_MODE == 'doc2vec':
    #     doc2vec_model = gensim.models.KeyedVectors.load_word2vec_format(appConfig.DOC2VEC_MODEL, binary=True)
    #     print (__name__ + 'Custom doc2vec model loaded')
    #     data_features = [doc2vec_model[question] for question in inputData]
    #     print(__name__ + '\tfeature shape : ' + str(len(data_features)))

    return data_features
