from pprint import pprint
from Parser import Parser
from textblob import TextBlob as tb
import numpy as np
import util
import glob
import os

class VectorSpace:
    """ A algebraic model for representing text documents as vectors of identifiers.
    A document is represented as a vector. Each dimension of the vector corresponds to a
    separate term. If a term occurs in the document, then the value in the vector is non-zero.
    """

    #Collection of document term vectors
    documentVectors = []

    #Mapping of vector index to keyword
    vectorKeywordIndex=[]

    #Tidies terms
    parser=None



    def __init__(self, documents=[]):
        self.documentVectors=[]
        self.parser = Parser()
        if(len(documents)>0):
            self.build(documents)

    def build(self,documents):
        """ Create the vector space for the passed document strings """
        self.vectorKeywordIndex = self.getVectorKeywordIndex(documents)
        self.documentVectors = [self.makeVector(document) for document in documents]

        #print self.vectorKeywordIndex
        #print self.documentVectors

#Indexing
    def getVectorKeywordIndex(self, documentList):
        """ create the keyword associated to the position of the elements within the document vectors """

        #Mapped documents into a single word string
        vocabularyString = " ".join(documentList)

        vocabularyList = self.parser.tokenise(vocabularyString)
        #Remove common words which have no search value
        vocabularyList = self.parser.removeStopWords(vocabularyList)
        uniqueVocabularyList = util.removeDuplicates(vocabularyList)

        vectorIndex={}
        offset=0
        #Associate a position with the keywords which maps to the dimension on the vector used to represent this word
        for word in uniqueVocabularyList:
            vectorIndex[word]=offset
            offset+=1
        return vectorIndex  #(keyword:position)


    def makeVector(self, wordString):
        """ @pre: unique(vectorIndex) """

        #Initialise vector with 0's
        vector = [0] * len(self.vectorKeywordIndex)
        wordList = self.parser.tokenise(wordString)
        wordList = self.parser.removeStopWords(wordList)
        for word in wordList:
            vector[self.vectorKeywordIndex[word]] += 1; #Use simple Term Count Model
        return vector

#Transfer Queries into a Vector
    def buildQueryVector(self, termList):
        """ convert query string into a term vector """
        query = self.makeVector(" ".join(termList))
        return query


    def related(self,documentId):
        """ find documents that are related to the document indexed by passed Id within the document Vectors"""
        ratings = [util.cosine(self.documentVectors[documentId], documentVector) for documentVector in self.documentVectors]
        #ratings.sort(reverse=True)
        return ratings

#Calculate the Similarity between the Query Vector and the Document Vectors
    def tfconsine(self,searchList,docID):

        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList)

        ratings = [util.cosine(queryVector, documentVector) for documentVector in self.documentVectors]

        dictionary = dict(zip(docID, ratings))

        sorted_doc = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)

        for word, score in sorted_doc[:5]:

            print("\n{}\t{}".format(word, round(score, 6)))

        return

    def tfJaccard(self,searchList,docID):

        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList)

        queryIndex = map(lambda val: ([i for i, x in enumerate(queryVector) if queryVector[i] != 0]), queryVector)[0]
        print queryIndex

        ratings = [self.JaccardScore(queryIndex,documentVector,queryVector) for documentVector in self.documentVectors]

        dictionary = dict(zip(docID, ratings))

        sorted_doc = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)

        for word, score in sorted_doc[:5]:

            print("\n{}\t{}".format(word, round(score, 6)))

        return

    def JaccardScore(self,queryIndex,documentVector,queryVector):

        # intersectionLen = 0

        # for i, (a, b) in enumerate(zip(queryVector, documentVector)):
        #
        #     while (queryVector[i] != 0) and (documentVector[i] != 0):
        #
        #             intersectionLen += 1 ;

        # queryIndex = map(lambda val: ([i for i, x in enumerate(queryVector) if queryVector[i] != 0]), queryVector)[0]
        # print queryIndex


        docIndex = map(lambda val: ([i for i in xrange(len(documentVector)) if documentVector[i] != 0]), documentVector)[0]
        print docIndex

        c = set(queryIndex).intersection(set(docIndex))

        print c

        return  float(len(c)) / (len(queryVector) + len(documentVector) - len(c))
        #





        # c = set(queryIndex).intersection(set(docIndex))
        #
        # print c
        #
        # return float(len(c)) / (len(queryVector) + len(documentVector) - len(c))

    def tfidfcos(self,text):
        for i, blob in enumerate(text):
            # print("Top words in document {}".format(i + 1))
            scores = {word: self.tfidf(word, blob, bloblist) for word in blob.words}

            sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for word, score in sorted_words[:3]:
                print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))




    def idf(word, bloblist):

        return math.log(len(bloblist) / (1 + 2048))

    def tfidf(word, blob, bloblist):
        return self.build(word, blob) * self.idf(word, bloblist)


if __name__ == '__main__':

    # query = raw_input("\nTerm Frequency (TF) Weighting + Cosine Similarity : ")

    # print ("\nDocID\tScore")

    documents = []
    docID = []

    text = []

    path = './documents/*.product'
    files = glob.glob(path)

    for filename in files:

        with open(filename, 'r') as f:

            documents.append(f.read())

            # text.append(tb(f.read()))


        base = os.path.basename(filename)

        docID.append(os.path.splitext(base)[0])

    vectorSpace = VectorSpace(documents)

    # print vectorSpace.vectorKeywordIndex

    # print vectorSpace.documentVectors

    # pprint(vectorSpace.related(1))

    # vectorSpace.tfconsine(query.split(' '),docID)

    # vectorSpace.tfconsine(["drill wood sharp"],docID)

    # print("Term Frequency (TF) Weighting + Jaccard Similarity : ")
    # print ("\nDocID\tScore")

    vectorSpace.tfJaccard(["drill wood sharp"],docID)
    # vectorSpace.tfJaccard(query.split(' '),docID)

    # vectorSpace.tfidfcos(text)


###################################################
