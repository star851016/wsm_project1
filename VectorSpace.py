from pprint import pprint
from Parser import Parser
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

    def tfJaccard(self,searchList):

        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList)

        ratings = [self.JaccardScore(queryVector, documentVector) for documentVector in self.documentVectors]

        ratings.sort(reverse=True)

        for i in range(0,5,+1):

            print ("\n{}\t{}".format(" ",round(ratings[i],6)))

        return

    def JaccardScore(self,labels1, labels2):

        n11 = n10 = n01 = 0
        n = len(labels1)

        for i, j in itertools.combinations(xrange(n), 2):
            comembership1 = labels1[i] == labels1[j]
            comembership2 = labels2[i] == labels2[j]
            if comembership1 and comembership2:
                n11 += 1
            elif comembership1 and not comembership2:
                n10 += 1
            elif not comembership1 and comembership2:
                n01 += 1
        return float(n11) / (n11 + n10 + n01)


    # def JaccardScore(self,queryVector, documentVector):
    #
    #     dist = (np.double(np.bitwise_and((queryVector != documentVector),np.bitwise_or(queryVector != 0, documentVector != 0)).sum()) /np.double(np.bitwise_or(queryVector != 0, documentVector != 0).sum()))
    #
    #     return dist


if __name__ == '__main__':

    # query = raw_input("\nTerm Frequency (TF) Weighting + Cosine Similarity : ")

    # print ("\nDocID\tScore")

    documents = []
    docID = []

    path = './documents/*.product'
    files = glob.glob(path)

    for filename in files:

        with open(filename, 'r') as f:

            documents.append(f.read())

        base = os.path.basename(filename)

        docID.append(os.path.splitext(base)[0])

    # print docID

    vectorSpace = VectorSpace(documents)

    # print vectorSpace.vectorKeywordIndex

    # print vectorSpace.documentVectors

    # pprint(vectorSpace.related(1))

    # vectorSpace.tfconsine(query.split(' '),docID)

    vectorSpace.tfconsine(["drill wood sharp"],docID)

    # print("Term Frequency (TF) Weighting + Jaccard Similarity : ")
    # print ("\nDocID\tScore")
    # vectorSpace.tfJaccard(["drill wood sharp"])
    # vectorSpace.tfJaccard(query.split(' '))

###################################################
