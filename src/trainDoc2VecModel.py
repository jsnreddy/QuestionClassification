import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))
import gensim.models as g
import codecs

from src import appConfig


# This function takes the unlabelled data and trains the doc2vec model, saves it to binary file in models folder
def train():

    inputFile = appConfig.DATA_FOLDER + "/" + appConfig.UNLABELLED_DATA_FILE

    print 'input data loaded'

    #doc2vec parameters
    vector_size = 300
    window_size = 5
    min_count = 1
    sampling_threshold = 1e-5
    negative_size = 5
    train_epoch = 100
    dm = 1 #0 = dbow; 1 = dmpv
    worker_count = 1 #number of parallel processes

    #loading the inputFile
    docs = g.doc2vec.TaggedLineDocument(inputFile)
    print 'docs loaded'

    #building model
    model = g.Doc2Vec(docs, size=vector_size, window=window_size, min_count=min_count, sample=sampling_threshold, workers=worker_count, hs=0, dm=dm, negative=negative_size, dbow_words=1, dm_concat=1, iter=train_epoch)

    print 'model built'
    #saving model
    # with open(appConfig.MODELS_FOLDER + '/' + 'doc2vec_window5_dmpv' + '.pickle', 'wb') as fid:
        # cPickle.dump(model, fid)
    model.save(appConfig.DOC2VEC_MODEL)
    print 'model saved'

#this is to test the trained doc2vec model
#THIS FUNCTION IS NOT WORKING. THERE IS AN ISSUE IN LOADING THE MODEL DUE TO ENCODING### DONOT USE THIS IN VERSION 1
def test():
    # inference hyper-parameters
    start_alpha = 0.01
    infer_epoch = 1000

    test_docs = appConfig.DATA_FOLDER + '/' + 'doc2vec_test.txt'
    model = appConfig.DOC2VEC_MODEL
    # load model
    m = g.Doc2Vec.load(model)
    #
    test_q = 'What is a gun?'
    #
    test_docs = [x.strip().split() for x in codecs.open(test_docs, "r", "utf-8").readlines()]
    #
    for t in test_docs:
        test_q = " ".join(t)
        # print m[test_q]
        # print m.infer_vector(test_q.lower(), alpha=start_alpha, steps=infer_epoch)
        print m.infer_vector(test_q.lower())

    # with open(appConfig.MODELS_FOLDER + "/" + 'doc2vec_window5_dmpv' + ".model", 'rb') as fid:
    #     model_loaded = cPickle.load(fid)
    # if test_q in model_loaded.vocab:
    #     vector = model_loaded[test_q]
    # print vector


train()
# test()