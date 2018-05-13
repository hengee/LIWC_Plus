import os
import csv
import re
from itertools import filterfalse
from itertools import chain
import numpy as np
import gensim

filesDict = []
## Convert CSV files into text
if False:
    pathName = "../../dataset/"
    fileNames = os.listdir(pathName)
    print("EFSADA", fileNames)
    paragraph = ''
    for file in fileNames:
        if file.endswith(".csv"):
            n = os.path.splitext(file) # Separates into name and .csv
            with open(n[0] + '.txt', "w") as textFile:
                with open('../../dataset/' + file, "r") as csvFile:
                    filt_f1 = filterfalse(lambda line: line.startswith('\n'), csvFile) # Ignores blank lines
                    reader = csv.reader(filt_f1, delimiter='\t')
                    for row in reader:
                        if(row[2] == None):
                            continue
                        if(row[2] == 'Participant'):
                            paragraph += row[3]
                            paragraph += ' '
                textFile.write(paragraph)
                textFile.close()
                filesDict.append(textFile)
                paragraph = ''

    
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

# Put words from LIWC to its corresponding python dictionary
i = 0
for file in fileNames:
    f = open(file)
    words = []

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

count = 0
countList = {}
emoCountDict = {}    
sent = {}
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True )
#print(model.wv.similarity('woman', 'woman'))
word_vectors = model.wv
del model
fileNames = os.listdir("TextFiles/")
for f in fileNames:     # For file in folder of files
    if f.endswith(".txt"):
        os.chdir(r'E:\NLP Research/sentiment labelled sentences/sentiment labelled sentences/TextFiles')
        #affectList['like'] = 0
        #posemoList['like'] = 0
        n = os.path.splitext(f)
        with open(f) as file:       # Open/closes file
            for line in file:       # For line in the file
                w = line.split()    # Split into words
                for s in w:         # For each word
                    s = re.sub(r'[^\w\s]','',s)
                    s = s.lower()
                    #if f == '302_TRANSCRIPT.txt' and s == 'like':
                    #    print( s )
                    #print("S", s)
                    count += 1                              # Number of words in each paragraph
                    for key, value in data.items():         # Iterate through LIWC data
                        for a in value:                     # Check if word is present in LIWC data
                            if s == a:                      # If it is, go to corresponding list
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
        #if affectList['like']:
        #    affectTemp = affectList['like']
        #if posemoList['like']:
        #    posemoTemp = posemoList['like'] 
        #if posemoList['like'] and affectList['like']:                       
        #    affectList['like'] = affectList['like'] / 21
        #    posemoList['like'] = posemoList['like'] / 21
        #    affect = affect - (affectTemp - affectList['like'])
        #    posemo = posemo - (posemoTemp - posemoList['like'])
            sent[file] = affectList, angerList, anxList, negemoList, posemoList, sadList, socialList
            affectList = {}
            angerList = {}
            anxList = {}
            negemoList = {}
            posemoList = {}
            sadList = {}
            socialList = {}
            numFile = n[0][0] + n[0][1] + n[0][2]           # Participant number
            emoCountDict[numFile] = {'affect': affect, 'anger': anger, 'anx': anx, 'negemo':negemo, 'posemo':posemo, 'sad':sad, 'social':social}
            if numFile == '303':
                print(emoCountDict[numFile])
            print(numFile)
            affect = 0
            anger = 0
            anx = 0
            negemo = 0
            posemo = 0
            sad = 0
            social = 0
            countList[numFile] = count      # Keeps track of total words in each paragraph
            count = 0

tempDict = {}
# Write training data to csv
if False:
    with open('training_data.csv', 'w') as f:
        with open('train_split_Depression_AVEC2017.csv', "r") as trainFile:
            filt_f1 = filterfalse(lambda line: line.startswith('\n'), trainFile) # Ignores blank lines
            reader = csv.reader(filt_f1, delimiter=',')
            for row in reader:
                for key,value in emoCountDict.items():
                    if row[0] == key:
                        tempDict = {k: v / countList[key] for k, v in value.items()}    # Divides nmber of words in each category by total number of words in paragraph
                        f.write('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}\n'.format(key,tempDict['affect'],tempDict['anger'],tempDict['anx'],tempDict['negemo'],tempDict['posemo'],tempDict['sad'],tempDict['social'],row[1],row[2]))        

# Write testing data to csv
if False:
    with open('testing_data.csv', 'w') as f:
        with open('test_split_Depression.csv', "r") as testFile:
            filt_f1 = filterfalse(lambda line: line.startswith('\n'), testFile) # Ignores blank lines
            reader = csv.reader(filt_f1, delimiter=',')
            for row in reader:
                for key,value in emoCountDict.items():
                    if row[0] == key:
                        tempDict = {k: v / countList[key] for k, v in value.items()}    # Divides nmber of words in each category by total number of words in paragraph
                        f.write('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}\n'.format(key,tempDict['affect'],tempDict['anger'],tempDict['anx'],tempDict['negemo'],tempDict['posemo'],tempDict['sad'],tempDict['social'],row[1],row[2]))
#print(emoCountDict[492])
#print(sent['492_TRANSCRIPT.txt'])