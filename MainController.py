from sklearn.model_selection import train_test_split

from Learner import *
from Classifier import *
from FeatureExtractor import *

inputData = pd.read_csv(appConfig.DATA_FOLDER + "/" + appConfig.TRAIN_FILE, sep="\t")

X = inputData[appConfig.DATA_FIELD]
y = inputData[appConfig.LABEL_FIELD]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=appConfig.DATA_SPLIT, random_state=42)

# print len(X_train)
# print X_test
# print y_test

learnerObject = Learner()
learnerObject.trainingData = X_train
learnerObject.trainingLabels = y_train

learnerObject.doMachineLearning()
learnerObject.fitClassifierAndSave()

classifierObject = Classifier()
classifierObject.testData = X_test
classifierObject.testLabels = y_test
# testdf = pd.DataFrame(data={'label':y_test})
# testdf.to_csv('output/testLabels.txt', sep='\t')
classifierObject.doClassification()
classifierObject.printConfusionMatrix()
