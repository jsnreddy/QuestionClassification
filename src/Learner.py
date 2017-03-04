import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))
import pandas as pd
import cPickle
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

import appConfig
import FeatureExtractor


class Learner(object):
    def __init__(self):
        self.dataFolder = appConfig.DATA_FOLDER
        self.modelsFolder = appConfig.MODELS_FOLDER
        self.trainingFile = appConfig.TRAIN_FILE
        self.trainingData = None
        self.trainingLabels = None
        self.trainingFeatures = None
        self.model = None

    #This function fit the classifier model(as configured in appConfig.CLASSIFIER_MODE,
    # cross-validates over 10 splits on training data and
    # saves the model to a pickle file in models folder
    # cross validation results are also saved to a file in output folder
    def fitClassifierAndSave(self):
        print (__name__ + '\tFitting classifier')
        if appConfig.CLASSIFIER_MODE == 'rf':
            forest = RandomForestClassifier(n_estimators=20)
            forest = forest.fit(self.trainingFeatures, self.trainingLabels)
            self.model = forest
            print (__name__ + '\tRandom Forest Model Fit')

        if appConfig.CLASSIFIER_MODE == 'lsvc':
            svc = svm.SVC(kernel='linear', C=1.0)
            svc = svc.fit(self.trainingFeatures, self.trainingLabels)
            self.model = svc
            print (__name__ + '\tLinear SVC Model Fit')

        if appConfig.CLASSIFIER_MODE == 'gnb':
            gnb = GaussianNB()
            gnb = gnb.fit(self.trainingFeatures, self.trainingLabels)
            self.model = gnb
            print (__name__ + '\tGaussian Naive Bayes Model Fit')

        if appConfig.CLASSIFIER_MODE == 'knn':
            knn = neighbors.KNeighborsClassifier(n_neighbors=appConfig.N_NEAREST_NEIGHBOURS)
            knn = knn.fit(self.trainingFeatures, self.trainingLabels)
            self.model = knn
            print (__name__ + '\tk-nearest neighbours Model Fit')

        if appConfig.CLASSIFIER_MODE == 'lr':
            lr = LogisticRegression(multi_class='ovr', random_state=42, class_weight='balanced', fit_intercept=False)
            lr = lr.fit(self.trainingFeatures, self.trainingLabels)
            self.model = lr

        if appConfig.CLASSIFIER_MODE == 'neuralN':
            mlp = MLPClassifier()
            mlp = mlp.fit(self.trainingFeatures, self.trainingLabels)
            self.model = mlp

        cvScores = cross_val_score(self.model, self.trainingFeatures, self.trainingLabels, cv=appConfig.CROSS_VALIDATION_FOLDS)
        print(__name__ + '\tCross Validation Scores on Training ..\n' + str(cvScores))
        avgCVScore = float(sum(cvScores))/len(cvScores)
        print(__name__ + '\tAverage Cross Validation Score on Training ..\n' + str(avgCVScore))
        cvScoresFile = open(appConfig.OUTPUT_FOLDER + '/' + appConfig.OUTPUT_FILE + '_cvScore.txt', 'w')
        cvScoresFile.write("Cross Validation Scores : " + str(cvScores))
        cvScoresFile.write("\n\nAverage Cross Validation Score : " + str(avgCVScore))

        with open(appConfig.MODELS_FOLDER + '/' + appConfig.OUTPUT_FILE + '.pickle', 'wb') as fid:
            cPickle.dump(self.model, fid)
        print (__name__ + '\tModel Saved to Pickle file')

    #This is the function doing all the learning
    # gets preprocessed data
    # gets features
    # builds model and saves it
    def doMachineLearning(self):
        if len(self.trainingData) > 0:
            print (__name__ + '\tTraining Data Shape : ' + str(self.trainingData.shape))
            self.trainingData = FeatureExtractor.preprocessData(self.trainingData)
            self.trainingFeatures = FeatureExtractor.getFeatures(self.trainingData)
            self.fitClassifierAndSave()

if __name__ == '__main__':
    trainingData = pd.read_csv(appConfig.DATA_FOLDER + "/" + appConfig.DATA_FILE, sep="\t")
    learnerObject = Learner()
    learnerObject.trainingData = trainingData[appConfig.DATA_FIELD]
    learnerObject.trainingLabels = trainingData[appConfig.LABEL_FIELD]
    learnerObject.doMachineLearning()
