import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))
import gensim.models as g
import codecs, cPickle

import appConfig

def train():

    inputFile = appConfig.DATA_FOLDER + "/" + appConfig.TEST_FILE

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
    pretrained_embeddings = None

    docs = g.doc2vec.TaggedLineDocument(inputFile)
    print 'docs loaded'
    model = g.Doc2Vec(docs, size=vector_size, window=window_size, min_count=min_count, sample=sampling_threshold, workers=worker_count, hs=0, dm=dm, negative=negative_size, dbow_words=1, dm_concat=1, iter=train_epoch)

    print 'model built'
    #save model
    # with open(appConfig.MODELS_FOLDER + '/' + 'doc2vec_window5_dmpv' + '.pickle', 'wb') as fid:
        # cPickle.dump(model, fid)
    model.save(appConfig.MODELS_FOLDER + '/'+'doc2vec_window5_dmpv.bin')
    print 'model saved'

def test():
    # inference hyper-parameters
    start_alpha = 0.01
    infer_epoch = 1000

    test_docs = appConfig.DATA_FOLDER + '/' + 'doc2vec_test.txt'
    model = appConfig.MODELS_FOLDER + '/'+'doc2vec_window5_dmpv.bin'
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