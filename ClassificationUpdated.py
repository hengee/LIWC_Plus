import os
import csv
import re
from itertools import filterfalse
from itertools import chain
import numpy as np
import gensim
from gensim.models.keyedvectors import KeyedVectors
from nltk.corpus import wordnet as wn

def csv_to_txt():
    filesDict = []
    ## Convert CSV files into text
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


def LIWCtoDict():
    affectFile = 'affect.txt'
    angerFile = 'anger.txt'
    anxFile = 'anx.txt'
    negemoFile = 'negemo.txt'
    posemoFile = 'posemo.txt'
    sadFile = 'sad.txt'
    socialFile = 'social.txt'
    
    fileNames = [affectFile, angerFile, anxFile, negemoFile, posemoFile, sadFile, socialFile]
    data = {}
    
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
    return data

def loadModel(model):
    #loadedModel = gensim.models.KeyedVectors.load_word2vec_format(model)
    #loadedModel = gensim.models.KeyedVectors.load_word2vec_format(model, binary=True )
    #loadedModel.save("2ndmodel.txt")
    print("SAVED")
    loadedModel = KeyedVectors.load(model)
    print("LOADED")
    #return word_vectors
    return loadedModel

def computeWord2Vec(data, threshold):
    count = 0
    countList = {}
    emoCountDict = {}    
    emotionNames = 'affect', 'anger', 'anx', 'negemo', 'posemo', 'sad', 'social'
    emoCount = {'affect' : 0, 'anger' : 0, 'anx' : 0, 'negemo' : 0, 'posemo' : 0, 'sad' : 0, 'social' : 0}
    emoSimilarityTotal = []
    similarityDict = {}
    similarityValue = 0
    WordCompDict = {}
    fileNames = os.listdir("TextFiles/")
    #model = loadModel("w2vModel.txt")
    parNums = []
    with open('TextFiles/training_testing_nums.txt') as dataNums:
        nums = dataNums.readlines()
        for n in nums:
            n = n.strip()
            parNums.append(n)
    for f in fileNames:     # For file in folder of files
        os.chdir(r'E:\NLP Research/sentiment labelled sentences/sentiment labelled sentences/TextFiles')
        if f.endswith('.txt'):
            n = os.path.splitext(f)
            numFile = n[0][0] + n[0][1] + n[0][2]           # Participant number
            if numFile in parNums:
                with open(f) as paragraph:       # Open/closes file
                    numFileTrack = 0
                    for line in paragraph:       # For line in the file
                        #print('one line')
                        emotionSimDict = {'affect' : {}, 'anger' : {}, 'anx' : {}, 'negemo' : {}, 'posemo' : {}, 'sad' : {}, 'social' : {}}
                        #relatedWordsDict = {'affect' : {}, 'anger' : {}, 'anx' : {}, 'negemo' : {}, 'posemo' : {}, 'sad' : {}, 'social' : {}}
                        wordList = line.split()    # Split into words
                        for words in wordList:         # For each word
                            words = re.sub(r'[^\w\s]','',words)
                            words = words.lower()
                            count += 1                              # Number of words in each paragraph
                            relatedWordsDict = {'affect' : {}, 'anger' : {}, 'anx' : {}, 'negemo' : {}, 'posemo' : {}, 'sad' : {}, 'social' : {}}
                            for emotion, emoWordList in data.items():         # Iterate through LIWC data
                                prevSimValue = ['',0]
                                for LIWCWord in emoWordList:                     # Check if word is present in LIWC data
                                    if words in WordCompDict:
                                        if LIWCWord in WordCompDict[words]:
                                            similarityValue = WordCompDict[words][LIWCWord]
                                        else:
                                            similarityValue = checkSimilarity(words, LIWCWord)
                                            WordCompDict[words][LIWCWord] = similarityValue
                                    else:
                                        similarityValue = checkSimilarity(words, LIWCWord)
                                        try:
                                            WordCompDict[words][LIWCWord] = similarityValue
                                        except:
                                            WordCompDict[words] = {}
                                            WordCompDict[words][LIWCWord] = similarityValue
                                    if prevSimValue[1] == 0:
                                        prevSimValue = [LIWCWord, similarityValue]
                                    else:
                                        if similarityValue > prevSimValue[1] and similarityValue > threshold:
                                            prevSimValue = [LIWCWord, similarityValue]
                                if words not in relatedWordsDict[emotion]:                      #Counting Related Words
                                    if prevSimValue[1] == 0:
                                        continue
                                    if similarityValue > threshold:
                                        relatedWordsDict[emotion][words] = similarityValue
                                    #with open('differencesWordNet15jugyj.csv','a') as newfile:
                                    #    if numFileTrack == 0:
                                    #       newfile.write('{0}\n'.format(numFile))
                                    #    newfile.write('{0},{1},{2},{3}\n'.format(emotion, words, prevSimValue[0], prevSimValue[1]))
                                    #    numFileTrack = 1
                                if prevSimValue[1] > threshold:
                                    if words in emotionSimDict[emotion]:
                                        emoCount[emotion] += 1
                                        #print(prevSimValue[1])
                                        emotionSimDict[emotion][words] += prevSimValue[1]
                                    else:
                                        #print(prevSimValue[1])
                                        emotionSimDict[emotion][words] = prevSimValue[1]
                                        emoCount[emotion] += 1       
                                #print(words)
                    emoCountList = {key : value+0.0001 for key, value in emoCount.items()} #Add 0.0001 to prevent division by zero errors
                    for emo, wordSim in emotionSimDict.items():
                        emoSimilarityTotal.append(sum(wordSim.values()))
                    for simTotal, num, emoNames in zip(emoSimilarityTotal, emoCountList.values(), emotionNames):
                        try:
                            similarityDict[numFile][emoNames] = (simTotal) / count
                        except:
                            similarityDict[numFile] = {}
                            similarityDict[numFile][emoNames] = (simTotal) / count
                    countList[numFile] = count      # Keeps track of total words in each paragraph
                    count = 0
                    emoSimilarityTotal = []
                    print(numFile)
                #if(numFile == '302'):
                    #break

    return countList, emoCountDict, similarityDict

#Check Similarity Function
def checkSimilarity( inputWord, dictWord):
    '''
    try:
        similarityValue = word_vectors.similarity(inputWord, dictWord)
    except:
        similarityValue = 0
    return similarityValue
'''
    textSyn = wn.synsets(inputWord)
    LIWCSyn = wn.synsets(dictWord)
    lenW1 = len(textSyn)
    lenW2 = len(LIWCSyn)
    maxValue = 0
    for i in range(lenW1):
        for j in range(lenW2):
            similarityValue = textSyn[i].path_similarity(LIWCSyn[j])
            try:
                if similarityValue > maxValue:
                    maxValue = similarityValue
            except:
                similarityValue = 0 
    return maxValue

    #textSyn = wn.synsets(inputWord)[0]
    #LIWCSyn = wn.synsets(dictWord)[0]
    #return textSyn.path_similarity(LIWCSyn)


#Write percentages into csv
def outputFile(similarityDict, fileName):
    with open(fileName, 'w') as file:
        for key, value in similarityDict.items():
            file.write('{0},{1},{2},{3},{4},{5},{6},{7}\n'.format(key,value['affect'],value['anger'],value['anx'],value['negemo'],value['posemo'],value['sad'],value['social']))        

# Write training data to csv
def writeTraining(emoCountDict, countList):
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
def writeTesting(emoCountDict, countList):
    with open('testing_data.csv', 'w') as f:
        with open('test_split_Depression.csv', "r") as testFile:
            filt_f1 = filterfalse(lambda line: line.startswith('\n'), testFile) # Ignores blank lines
            reader = csv.reader(filt_f1, delimiter=',')
            for row in reader:
                for key,value in emoCountDict.items():
                    if row[0] == key:
                        tempDict = {k: v / countList[key] for k, v in value.items()}    # Divides number of words in each category by total number of words in paragraph
                        f.write('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}\n'.format(key,tempDict['affect'],tempDict['anger'],tempDict['anx'],tempDict['negemo'],tempDict['posemo'],tempDict['sad'],tempDict['social'],row[1],row[2]))
                        
if __name__ == "__main__":
    threshold = .15
    LIWCDict = LIWCtoDict()
    #model = loadModel("w2vModel.txt")
    #model = loadModel('wiki-news-300d-1M.vec')
    countList, emoCountDict, similarityDict = computeWord2Vec(LIWCDict, threshold)
    outputFile(similarityDict, 'WordNet_15_all.csv')
    
    