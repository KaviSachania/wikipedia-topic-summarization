import tfidf


#tfidf.readCorpus(10000)

tfidf.initFromFile("tfidf.txt")

sortedTitles = tfidf.tfidf("Cold War", 10)

if sortedTitles is not None:
    for title in sortedTitles:
        print("%.5f"%title[1], title[0])