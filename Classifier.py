import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))
import pandas as pd, cPickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

import appConfig
import FeatureExtractor


class Classifier(object):
    def __init__(self):
        self.dataFolder = appConfig.DATA_FOLDER
        self.modelsFolder = appConfig.MODELS_FOLDER
        self.testFile = appConfig.TEST_FILE
        self.testData = None
        self.testLabels = None
        self.testFeatures = None
        self.result = None

    def printConfusionMatrix(self):

        y_actual = pd.Series(self.testLabels, name='Actual')
        y_prediction = pd.Series(self.result, name='Predicted')
        confusionMatrix = confusion_matrix(y_actual, y_prediction)
        # this is without margin(the total number of cases)
        # confusionMatrix = pd.crosstab(y_actual, y_prediction)
        print 'predicted..\n', y_prediction.value_counts()
        print 'actual..\n', y_actual.value_counts()
        # confusionMatrix = pd.crosstab(y_actual, y_prediction, rownames=['Actual'], colnames=['Predicted'], margins=True)

        print confusionMatrix
        # confusionMatrix.to_csv(appConfig.OUTPUT_FOLDER + "/" + appConfig.CLASSIFIER_MODE + "_numbers.txt", sep='\t')

        accuracy = accuracy_score(y_actual, y_prediction, normalize=True)
        precision, recall, f_score, support = precision_recall_fscore_support(self.testLabels, self.result, average='macro')

        print (__name__ + '\tprecision : ' + str(precision))
        print (__name__ + '\trecall: ' + str(recall))
        print (__name__ + '\tf-score : ' + str(f_score))
        print (__name__ + '\tAccuracy : ' + str(accuracy))
        outputNumbersFile = open(
            appConfig.OUTPUT_FOLDER + "/" + appConfig.OUTPUT_FILE + "_numbers.txt", 'w')
        outputNumbersFile.write('\nPredicted values counts : ' + str(y_prediction.value_counts()))
        outputNumbersFile.write('\nActual values counts : ' + str(y_actual.value_counts()))
        outputNumbersFile.write('\nConfusion Matrix : ' + str(confusionMatrix))
        outputNumbersFile.write('\nAccuracy : ' + str(accuracy))
        outputNumbersFile.write('\nPrecision : ' + str(precision))
        outputNumbersFile.write('\nRecall : ' + str(recall))
        outputNumbersFile.write('\nf-score : ' + str(f_score))
        outputNumbersFile.close()





    def classify(self):

        # if appConfig.CLASSIFIER_MODE == 'rf':
        with open(appConfig.MODELS_FOLDER + "/" +appConfig.OUTPUT_FILE+ ".pickle", 'rb') as fid:
            model_loaded = cPickle.load(fid)

        self.result = model_loaded.predict(self.testFeatures)
        # print str(self.result.shape)
        # print str(self.testLabels.shape)
        # if len(self.result) == len(self.testLabels):
        #     print 'result and testLabels are same length'
        print (__name__ + '\tPrediction Done')
        output = pd.DataFrame(data={appConfig.DATA_FIELD: self.testLabels, appConfig.LABEL_FIELD: self.result})

        output.to_csv(appConfig.OUTPUT_FOLDER + "/" + appConfig.OUTPUT_FILE + "_cases.tsv", sep="\t")
        print (__name__ + '\tOutput written to file')

    def doClassification(self):

        if len(self.testData) > 0:
            print (__name__ + '\tTesting Data Shape : ' + str(self.testData.shape))
            self.testData = FeatureExtractor.preprocessData(self.testData)
            self.testFeatures = FeatureExtractor.getFeatures(self.testData)
            print (__name__ + '\tTest Features extracted')
            self.classify()


if __name__ == '__main__':
    testData = pd.read_csv(appConfig.DATA_FOLDER + '/' + appConfig.TEST_FILE, sep='\t')
    classifierObject = Classifier()
    classifierObject.testData = testData[appConfig.DATA_FIELD]
    classifierObject.testLabels = testData[appConfig.LABEL_FIELD]
    classifierObject.doClassification()
    classifierObject.printConfusionMatrix()
