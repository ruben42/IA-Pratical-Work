# Spam filter with perceptron algorithm

import collections
import numpy as np

f1 = open("..\sms-spam-collection-dataset\spam.csv", "r")
contents =f1.readlines()
training = []
validating = []
isSpam = []
for i in range(len(contents)):
    if i < 5572:
        training.append(contents[i])
    else:
        validating.append(contents[i])


def createDictionaries(): # returns all email dictionaries and vocab dictionary
    allDict = list() # list of dictionaries for each email
    vocab = collections.Counter() # total dictionary for all words
    for email in training:
        emailDict = dict()
        split = email.split()
        for i in range(len(split)):
            if i == 0:
                if split[i] == "0": # not spam
                    isSpam.append(-1)
                else:
                    isSpam.append(1)
                continue
            if split[i] not in emailDict: #if we have not already seen it before
                vocab[split[i]] += 1
                emailDict[split[i]] = 1
            else: 
                emailDict[split[i]] += 1
        allDict.append(emailDict)
    return allDict, vocab 
    
def getNotIgnoredVocab():
    notIgnoredVocab = set()
    vocab = createDictionaries()[1]
    for word in vocab: 
        if vocab[word] >= 30:
            notIgnoredVocab.add(word)
    return notIgnoredVocab
    
#build vocabulary list
def buildVocabularyVectors():
    #ignore all words that appear in fewer than 30 emails
    allDict = createDictionaries()[0]
    notIgnoredVocab = getNotIgnoredVocab()
    
    #make the dictionaries for each email
    featureVectors = []
    for email in allDict:
        emailVector = []
        for word in notIgnoredVocab:
            if word in email:
                emailVector.append(1)
            else:
                emailVector.append(0)
        # print(emailVector)
        featureVectors.append(np.array(emailVector))

    return notIgnoredVocab, np.array(featureVectors)
        
        

# def emailVector(notIgnoredSet):
    # make dictionary of each word in notIgnoredSet and then check if its in the individual email 

    #for each 

def perceptron_train():
    notIgnoredVocab, featureVectors = buildVocabularyVectors()
    w = np.zeros(len(notIgnoredVocab))
    print(len(featureVectors))
    # keep updating until the weight does not need to be modified
    
    numMistakes = 0
    numUpdates = 0
    numIterations = 0
    while (True):
        numIterations += 1
        numUpdates = 0
        for i in range(len(featureVectors)):
            dot = np.dot(w, featureVectors[i])
            if dot == 0: 
                dot = 1
            # check for sign agreement
            sign = isSpam[i] * dot
            if sign <= 0: 
                numMistakes += 1
                numUpdates += 1
            #modify weight accordingly
                w = np.add(w, np.multiply(isSpam[i],featureVectors[i]))
        if numUpdates == 0:
            break

    # return the number of mistakes, the number of iterations and the weight array
    return w, numMistakes, numIterations

def perceptron_test(w, data):
    print("hiiii")
    dataIsSpam = [] # saves the spam values for the data given as parameter
    dataDict = list()
    for email in data:
        emailDict = dict()
        split = email.split()
        for i in range(len(split)): #iterating over each word
            if i == 0:
                if split[i] == "0": # not spam
                    dataIsSpam.append(-1)
                else:
                    dataIsSpam.append(1)
                continue
            if split[i] not in emailDict: #if we have not already seen it before
                emailDict[split[i]] = 1
            else: 
                emailDict[split[i]] += 1
        dataDict.append(emailDict)
    
    notIgnoredVocab = buildVocabularyVectors()[0]
    featureVectors = []
    print("wop")
    for email in dataDict:
        emailVector = []
        for word in notIgnoredVocab:
            if word in email:
                emailVector.append(1)
            else:
                emailVector.append(0)
        # print(emailVector)
        featureVectors.append(np.array(emailVector))
    
    numMistakes = 0
    numIterations = 0
    print("oh hi")
    for i in range(len(featureVectors)):
        dot = np.dot(w, featureVectors[i])
        if dot == 0: 
            dot = 1
        # check for sign agreement
        sign = isSpam[i] * dot
        if sign <= 0: 
            numMistakes += 1
        numIterations += 1
    print(numMistakes)
    print(numIterations)
    return numMistakes/numIterations
        
    #now i have the feature vectors, the notignoredvocab, and I need w
    
w, numMistakes, numIterations = perceptron_train()
print(w, numMistakes, numIterations)
print(perceptron_test(w, training))
print(perceptron_test(w, validating))