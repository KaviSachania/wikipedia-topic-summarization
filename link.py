import tfidf
import wikipedia
import re
import math
from nltk.tokenize import word_tokenize, sent_tokenize



def init():
    tfidf.initFromFile("tfidf.txt") # use pre-processed tf-idf vectors from file'
    
    
def get_tfidf_doc(docTitle):
    sortedTitles = tfidf.tfidf(docTitle, nTitles=20)

    documents = list()
    for title in sortedTitles:
        wikisearch = wikipedia.WikipediaPage(title[0])
        wikicontent = wikisearch.links
        documents.append(wikicontent)
    return sortedTitles, documents

def dist(docA, docB):
    total = 0.0
    for link in docA:
        if link in docB:
            total += 1
    return total / (math.sqrt(len(docA))*math.sqrt(len(docB)))


def link(docTitle, nTitles):
    titles, docs = get_tfidf_doc(docTitle)
    
    titleDists = dict()
    
    for titleI in range(len(titles)):
        titleDists[titles[titleI][0]] = dist(docs[0], docs[titleI])

    sortedTitles = sorted(titleDists.items(), key=lambda x: x[1], reverse=True)
    
    return list(sortedTitles[:nTitles])