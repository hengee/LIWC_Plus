import os
import numpy as np
from sklearn import svm,tree,preprocessing
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import random

#training = 'training_data.csv'
#training = 'training_data_threshold70.csv' #70%
#training = 'LIWC2015_results_training_data.csv' #7 features from LIWC
#training = 'training_data_allfeatures.csv' #all features from LIWC
training = 'WordNet_15_realtraining.csv'
training_depression_values = 'training_data.csv'

os.chdir(r'E:\NLP Research/sentiment labelled sentences/sentiment labelled sentences/TextFiles')

#dataX = np.loadtxt(training, delimiter=',', usecols=range(1,8)) #70%
#dataX = np.loadtxt(training, delimiter=',', usecols=range(0,7)) #7 features from LIWC
dataX = np.loadtxt(training, delimiter=',') #all features from LIWC
dataY = np.loadtxt(training_depression_values, delimiter=',', usecols=(8,))

X = dataX
y = dataY


clf = tree.DecisionTreeClassifier( random_state= 1 )
#clf = svm.SVC()

#scaler = preprocessing.StandardScaler().fit(X)
#X = scaler.transform(X)
X = preprocessing.scale(X)

clf.fit(X, y) 

#TESTING ---------------------------------------------------------------------------------------------------

#testing = 'testing_data.csv'
#testing = 'testing_data_threshold70.csv' #70%
#testing = 'LIWC2015_results_testing_data.csv' #7 featuresf rom LIWC
#testing = 'testing_data_allfeatures.csv' #all features from lIWC
testing = 'WordNet_15_realtesting.csv'
testing_depression_values = 'testing_data.csv'

#data_testX = np.loadtxt(testing, delimiter=',', usecols=range(1,8)) #70%
#data_testX = np.loadtxt(testing, delimiter=',', usecols=range(0,7)) #7 features from LIWC
data_testX = np.loadtxt(testing, delimiter='\t') #all features from LIWC

data_test_true = np.loadtxt(testing_depression_values, delimiter=',', usecols=(8,))

#data_testX = scaler.transform(data_testX)
data_testX = preprocessing.scale( data_testX )
data_test_predict = clf.predict(data_testX)
print("Threshold of .15")
print(data_test_predict)
print(data_test_true)

acc = accuracy_score(data_test_predict, data_test_true)
print(acc)
