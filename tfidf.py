import pandas as pd
import numpy as np
import sklearn
import math
import nltk
from smart_open import open
import gensim
from gensim.utils import simple_preprocess

from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')

idfV = dict()
idfIndexDict = dict()
pageIds = list()
pageTitles = list()
pageTexts = list()
pageTokens = list()

tfV = [dict()]

tfidfV = [dict()]*len(pageTexts)
pageTfidfLengths = list()


def readCorpus(docCount):
    global pageTitles, tfidfV, pageTfidfLengths
    
    #seperate document fields, tokenize, idf, and create token idf index

    idfV = dict()
    idfIndexDict = dict()
    pageIds = list()
    pageTitles = list()
    pageTexts = list()
    pageTokens = list()
    n = 0

    docCount = int(docCount or -1)

    f = open("lateral.csv", "r")
    for line in f:
        if n==docCount:
            break;

        #seperate title and document
        splitFields = line.split(",", 1)
        pageIds.append(splitFields[0])

        splitText = splitFields[1].split("  ", 1)
        pageTitles.append((splitText[0])[2:])

        text = splitText[1]
        pageTexts.append(text)

        pageTokens.append(list())
        tokens = simple_preprocess(text)
        cleanedTokens = list(filter(lambda x: x not in stop_words, tokens))
        pageTokens[n] = cleanedTokens

        tokensInDoc = set()

        for token in cleanedTokens:
            if token not in tokensInDoc:
                tokensInDoc.add(token)
                if token in idfIndexDict:
                    idfV[idfIndexDict[token]] += 1
                else:
                    idfIndexDict[token] = len(idfIndexDict)
                    idfV[idfIndexDict[token]] = 1

        n += 1

    f.close()

    numDocs = len(pageTokens)
    idfV = dict((k, math.log10(numDocs / v)) for k, v in idfV.items())
    
    
    #tf

    n = 0
    tfV = [dict()]*len(pageTexts)

    for tokens in pageTokens:
        docTf = dict()

        for token in tokens:
            tokenIndex = idfIndexDict[token]
            if tokenIndex in docTf:
                docTf[tokenIndex] += 1
            else:
                docTf[tokenIndex] = 1

        tfV[n] = docTf
        n += 1
        
    
    #tf-idf

    n = 0
    tfidfV = [dict()]*len(pageTexts)
    pageTfidfLengths = list()

    for tf in tfV:
        docTfidf = dict()
        pageTfidfSquare = 0.0

        for tokenIndex in tf:
            tfidf = tf[tokenIndex]*idfV[tokenIndex]
            docTfidf[tokenIndex] = tfidf
            pageTfidfSquare += tfidf**2

        tfidfV[n] = docTfidf
        pageTfidfLengths.append(math.sqrt(pageTfidfSquare))
        n += 1
        
    
    #write tfidf file

    tfidfFile = open('tfidf.txt', 'w')

    columns = "#Page Number#Page Title#TfidfLength#Tfidf\n"

    tfidfFile.write(columns)

    n = 0
    for page in pageTitles:
        line = str(n) + "#" + page + "#" + str(pageTfidfLengths[n]) + "#"

        tfidf = tfidfV[n]
        for tokenIndex in tfidf:
            line += str(tokenIndex) + ":" + str(tfidf[tokenIndex]) + "#"

        line = line[:-1]
        line += "\n"

        tfidfFile.write(line)
        n += 1

    tfidfFile.close()
    


def initFromFile(fileName):
    global pageTitles, tfidfV, pageTfidfLengths
    
    inFile = open(fileName, "r", encoding='utf-8')

    pageIds = list()
    pageTitles = list()
    pageTfidfLengths = list()

    for i, l in enumerate(inFile):
        pass

    tfidfV = [dict()]*i
    
    
    
    inFile.seek(0)
    n = 0

    inFile.readline()
    for line in inFile:
        fields = line.split("#", 3)
        pageTitles.append(fields[1])
        pageTfidfLengths.append(float(fields[2]))

        tfidfFields = fields[3].split("#")

        tfidf = dict()

        for tfidfField in tfidfFields:
            splitField = tfidfField.split(":")
            tfidf[int(splitField[0])] = float(splitField[1])

        tfidfV[n] = tfidf

        n += 1


    inFile.close()



#tf-idf function

def tfidf(rootPageTitle, nTitles):
    if rootPageTitle in pageTitles:
        rootDoc = pageTitles.index(rootPageTitle)
    else:
        print(rootPageTitle, "not in list of WikiPedia pages")
        return list()


    #tf-idf

    rootTfidf = tfidfV[rootDoc]
    rootTfidfLength = pageTfidfLengths[rootDoc]

    simScores = dict()

    n = 0

    for tfidf in tfidfV:
        ab = 0.0

        for tokenIndex in tfidf:
            if tokenIndex in rootTfidf:
                ab += rootTfidf[tokenIndex]*tfidf[tokenIndex]

        simScore = ab / (rootTfidfLength * pageTfidfLengths[n])
        simScores[pageTitles[n]] = simScore

        n += 1


    #sort page titles by similarity score

    sortedTitles = sorted(simScores.items(), key=lambda x: x[1], reverse=True)
    
    return list(sortedTitles[:nTitles])
    


    

