import numpy as np
from sklearn import svm
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

enc = OneHotEncoder()



fileName = 'amazon_cells_labelled.txt'

data = np.loadtxt( fileName, dtype='str', delimiter ='\t', converters={1:lambda x:x.decode()} )

features = data[ :, 0 ]

print (features)
#X = np.random.randint(2, size=10)
#X = [1000, 94]
#y = np.random.randint(2, size=10)

#y = enc.fit(data[ :, 1])
#y = features

#print (X, y)

#clf.fit(features, y)

#print (clf)

'''y = features.target
clf.fit(features, y)
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
clf2.predict(features[0:1])

np.savetxt( 'preprocessed.csv', features, delimiter = ',', fmt='%s' )


feature = np.loadtxt( 'feature.csv', delimiter = ',' )
print(feature)
ground_truth = data[ :, 1 ]

clf = svm.SVC()
clf.fit(feature[ 0 :500, : ], ground_truth[ 0:500 ])


predictions = clf.predict( feature[ 500: 600, : ] )
acc = accuracy_score( ground_truth[500:600], predictions )


wordDic = { 'angry': ['shout', 'depressed'], 'happy': ['good','smile'] } 
'''