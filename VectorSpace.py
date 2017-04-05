from pprint import pprint
from Parser import Parser
from textblob import TextBlob as tb
from nltk.tag import pos_tag
from numpy import multiply
import numpy as np
import util
import glob
import os
import math

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

    querytfidf = []

    doctfidfs = []

    idfs = []


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

        queryVector = self.buildQueryVector(searchList)

        # queryIndex = map(lambda val: ([i for i, x in enumerate(queryVector) if queryVector[i] != 0]), queryVector)[0]
        # print queryIndex

        ratings = [self.JaccardScore(documentVector,queryVector) for documentVector in self.documentVectors]

        dictionary = dict(zip(docID, ratings))

        sorted_doc = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)

        for word, score in sorted_doc[:5]:

            print("\n{}\t{}".format(word, round(score, 6)))

        return

    def JaccardScore(self,documentVector,queryVector):

        # queryIndex = map(lambda val: ([i for i, x in enumerate(queryVector) if queryVector[i] != 0]), queryVector)[0]
        # print queryIndex

        # docIndex = map(lambda val: ([i for i in xrange(len(documentVector)) if documentVector[i] != 0]), documentVector)[0]
        # print docIndex

        prod_of_lists = multiply(queryVector,documentVector)
        print prod_of_lists
        c = map(lambda val: ([i for i in xrange(len(prod_of_lists)) if prod_of_lists[i] != 0]), prod_of_lists)[0]
        print c

        return  float(len(c)) / (len(queryVector) + len(documentVector) - len(c))

    def tfidfcos(self,searchList,docID):

        n_containings = []



        queryVector = self.buildQueryVector(searchList)

        for word in self.vectorKeywordIndex:
            n_containings.append(self.n_containing(self.vectorKeywordIndex[word],self.documentVectors))
        # print n_containings

        for n_containing in n_containings:
            self.idfs.append(self.idf(queryVector,n_containing))
        # print idfs

        for  i,(value,idf) in enumerate(zip(queryVector, self.idfs)):
            self.querytfidf.append(self.tfidf(queryVector[i],idf))
        # print querytfidf

        for  i ,(documentVector,idf) in enumerate(zip(self.documentVectors, self.idfs)):
            doctfidf = []
            for i in range(0, len(documentVector)):
                doctfidf.append(self.tfidf(documentVector[i],idf))
            self.doctfidfs.append(doctfidf)
        # print doctfidfs

        ratings = [util.cosine(self.querytfidf, doc) for doc in self.doctfidfs]

        dictionary = dict(zip(docID, ratings))

        sorted_doc = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)

        for word, score in sorted_doc[:5]:

            print("\n{}\t{}".format(word, round(score, 6)))

        self.feedBack(searchList,sorted_doc[0][0],docID)

        return

    def n_containing(self,word, documentVectors):

        return sum(1 for documentVector in documentVectors  if documentVector[word] != 0)

    def idf(self,documentVectors, n_containing):

        return math.log(len(documentVectors) / n_containing)

    def tfidf(self,value,idf):

        return float(value * idf)

    def tfidfJaccard(self,searchList,docID):

        ratings = [self.JaccardScore(self.querytfidf, doc) for doc in self.doctfidfs]

        dictionary = dict(zip(docID, ratings))

        sorted_doc = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)

        for word, score in sorted_doc[:5]:

            print("\n{}\t{}".format(word, round(score, 6)))

        # print("Feedback Queries TF-IDF Weighting + Jaccard Similarity : ")
        # print ("\nDocID\tScore")

        # get the first document and call self.feedBack(sorted_doc[1],docID)
        # print sorted_doc[0][0]

        return

    def feedBack(self,searchList,document1,docID):

        querytfidf = []

        queryVector = self.buildQueryVector(searchList)

        filename = './documents/' + document1 + '.product'

        FILE = open(filename,"r")

        content = FILE.read()

        # print content

        fdbkQuery = self.getNV(content)

        print fdbkQuery

        document1Vector = [0] * len(self.vectorKeywordIndex)

        for word in fdbkQuery:
            document1Vector[self.vectorKeywordIndex[word]] += 1; #Use simple Term Count Model

        # print document1Vector

        document1Vector = [i * 0.5 for i in document1Vector]
        queryVector = [i * 1 for i in queryVector]

        newQueryVector = [sum(x) for x in zip(document1Vector, queryVector)]

        print newQueryVector

        for  i,(value,idf) in enumerate(zip(newQueryVector, self.idfs)):
            querytfidf.append(self.tfidf(newQueryVector[i],idf))
        print querytfidf

        ratings = [self.JaccardScore(querytfidf, doc) for doc in self.doctfidfs]

        dictionary = dict(zip(docID, ratings))

        sorted_doc = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)

        for word, score in sorted_doc[:5]:

            print("\n{}\t{}".format(word, round(score, 6)))

        return

    def getNV(self,document1):

        vocabularyList = self.parser.tokenise(document1)
        vocabularyList = self.parser.removeStopWords(vocabularyList)
        uniqueVocabularyList = util.removeDuplicates(vocabularyList)

        tagged_sent = pos_tag(uniqueVocabularyList)
        fdbkQuery = [word for word,pos in tagged_sent if pos.startswith('V') or pos.startswith('N')]
        # print fdbkQuery

        return fdbkQuery



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

        base = os.path.basename(filename)
        docID.append(os.path.splitext(base)[0])

    vectorSpace = VectorSpace(documents)
    # vectorSpace.tfconsine(query.split(' '),docID)

    # print("Term Frequency (TF) Weighting + Jaccard Similarity : ")
    # print ("\nDocID\tScore")
    # vectorSpace.tfJaccard(query.split(' '),docID)

    # vectorSpace.tfJaccard(["drill wood sharp"],docID)

    # print("TF-IDF Weighting + Cosine Similarity : ")
    # print ("\nDocID\tScore")
    # vectorSpace.tfidfcos(query.split(' '),docID)

    vectorSpace.tfidfcos(["store"],docID)

    # print("TF-IDF Weighting + Jaccard Similarity : ")
    # print ("\nDocID\tScore")
    # vectorSpace.tfidfJaccard(query.split(' '),docID)

    # vectorSpace.tfidfJaccard(["drill wood sharp"],docID)

    # print("Feedback Queries TF-IDF Weighting + Jaccard Similarity : ")
    # print ("\nDocID\tScore")
    # vectorSpace.feedBack(query.split(' '),docID)

    # vectorSpace.feedBack(["drill wood sharp"],docID)



###################################################
