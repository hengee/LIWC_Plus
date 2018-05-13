# -*- coding: utf-8 -*-
import os
import re
import numpy as np
#from sklearn import svm
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.metrics import accuracy_score
import gensim, logging 
from gensim.models import Word2Vec
#word2vec

affectFile = 'affect.txt'
angerFile = 'anger.txt'
anxFile = 'anx.txt'
negemoFile = 'negemo.txt'
posemoFile = 'posemo.txt'
sadFile = 'sad.txt'
socialFile = 'social.txt'

fileNames = [affectFile, angerFile, anxFile, negemoFile, posemoFile, sadFile, socialFile]
data = {}
countDict = {}
affectList = {}
angerList = {}
anxList = {}
negemoList = {}
posemoList = {}
sadList = {}
socialList = {}
angerList = {}
affect = 0
anger = 0
anx = 0
negemo = 0
posemo = 0
sad = 0
social = 0

i = 0
for file in fileNames:
    f = open(file)
    words = []

#data = []
    for word in f.read().split():
        words.append(word)
    if (i == 0):
        data['affect'] = words
    elif (i == 1):
        data['anger'] = words
    elif (i == 2):
        data['anx'] = words
    elif (i == 3):
        data['negemo'] = words
    elif (i == 4):
        data['posemo'] = words
    elif (i == 5):
        data['sad'] = words
    elif (i == 6):
        data['social'] = words
    i += 1
    
yelpFile = 'yelp_labelled.txt'

yelpData = np.loadtxt( yelpFile, dtype='str', delimiter ='\t', converters={1:lambda x:x.decode()} )
features = yelpData[:,0]
sent = {}
k = 0
for f in features:
    w = f.split()
    #print(w)
    for s in w:
        s = re.sub(r'[^\w\s]','',s)
        s = s.lower()
        #print("S", s)
        for key, value in data.items():
            #sentFeatures[k] = []
            for a in value:
                #print(a)
                #break
                if s == a:
                    if key == 'affect':
                        if s in affectList:
                            affectList[s] += 1
                        else:
                            affectList[s] = 1
                        affect += 1
                    elif key == 'anger':
                        if s in angerList:
                            angerList[s] += 1
                        else:
                            angerList[s] = 1
                        anger += 1
                    elif key == 'anx':
                        if s in anxList:
                            anxList[s] += 1
                        else:
                            anxList[s] = 1
                        anx += 1
                    elif key == 'negemo':
                        if s in negemoList:
                            negemoList[s] += 1
                        else:
                            negemoList[s] = 1
                        negemo += 1
                    elif key == 'posemo':
                        if s in posemoList:
                            posemoList[s] += 1
                        else:
                            posemoList[s] = 1
                        posemo += 1
                    elif key == 'sad':
                        if s in sadList:
                            sadList[s] += 1
                        else:
                            sadList[s] = 1
                        sad += 1
                    elif key == 'social':
                        if s in socialList:
                            socialList[s] += 1
                        else:
                            socialList[s] = 1
                        social += 1
        #break
    sent[f] = affectList, angerList, anxList, negemoList, posemoList, sadList, socialList
    #print("NEXT: ", sent)
    #Remove these if you want total words
    affectList = {}
    angerList = {}
    anxList = {}
    negemoList = {}
    posemoList = {}
    sadList = {}
    socialList = {}

    #k += 1
    #if k == 5:
    #    break
    #break
#Prints total number of features in each
#print("Affect: ", affectList, "Anger: ", angerList, "Anx: ", anxList, "Negemo: ", negemoList, "Posemo: ", posemoList, "Sad: ", sadList, "Social: ", socialList)
#print(sent)
         


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#sentences = [['first', 'sentence'], ['second', 'sentence']]
# train word2vec on the two sentences
#model = gensim.models.Word2Vec(sentences, min_count=1) 

#model = Word2Vec.load_word2vec_format('/tmp/vectors.bin.gz', binary=True)
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True )
#print("NMODEL: ", model)


print(model.wv.similarity('woman', 'woman'))
word_vectors = model.wv
del model
print("WOP: ", word_vectors.similarity('woman', 'queen'))