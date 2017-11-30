

import logging
import gensim
import pandas as pd
import re
import numpy as np
import scipy.spatial.distance


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
vectorSize = 300
word2VecModelFile = 'text.model.bin'
wordVecArrayFile = 'wordVecArray'
caseNumberArrayFile = 'caseNumberArray'

class MySentences():
    def __init__(self,csvFile):
        # User can also add prune words.
        self.pruneWords = ['Mit freundlichen','best Regards','take care','yours sincerely','yours faithfully','sincerely','thanks','Regards','yours']
        self.df = pd.read_csv(csvFile)


    def getSentences(self):
        # Think of an iterator implementation
        dummyDF = self.df
        sentences = dummyDF.Description.apply(self.cleanSentence)
        sentences = sentences.values.T.tolist()
        return sentences

    def caseID(self):
        return  self.df['Case Number'].values.values.T.tolist()

    def cleanSentence(self,sentence):
        if str(type(sentence)) != "<class 'str'>": # If the text is not a string
            sentence = str(sentence)
            return sentence
        for word in self.pruneWords :
            searchWord = re.search(r'\b({})\b'.format(word), sentence,re.IGNORECASE)
            if searchWord:
                start = searchWord.start()
                sentence = sentence[:start]
                break # There is only one way of thanking
        morePruneWord = ['MATLAB Version','>> ver']

        for word in morePruneWord :
            searchWord = re.search(r'\b({})\b'.format(word), sentence,re.IGNORECASE)
            if searchWord:
                start = searchWord.start()
                sentence = sentence[:start]

        sentence = re.sub(r'^https?:\/\/.*[\r\n]*', '', sentence, flags=re.MULTILINE) # Remove Hyperlink
        #sentence = re.sub(r'([^\s\w])+', '', sentence) # Remove special characters
        return sentence.split()


class TSBuildCaseVector(MySentences):
    def __init__(self,csvFile,model):
        super(TSBuildCaseVector,self).__init__(csvFile)
        self.__model = model


    def sentencePruner(self):
        pass

    def buildVector(self,sentence):
        #A Run of sentencePruner is required
        averageWordVector = np.zeros(300)
        count = 1
        for word in sentence:
            try:
                wordVector = self.__model[word]
                count += 1
            except:
                continue
            averageWordVector += self.__model[word]
        averageWordVector /= count
        return averageWordVector


    def buildWordVecTable(self):
        sentences = self.getSentences()
        wordVectorArray = []
        for sentence in sentences:
            averageWordVector = self.buildVector(sentence)
            wordVectorArray.append(averageWordVector)
        wordVectorArray = np.array(wordVectorArray)
        np.save(wordVecArrayFile, wordVectorArray)
        self.caseMap()

    def caseMap(self):
        casesNumber = self.df['Case Number'].as_matrix()
        np.save(caseNumberArrayFile, casesNumber)

class SimilarCaseGenerator(TSBuildCaseVector):
    def __init__(self,csvFile,model):
        super(SimilarCaseGenerator,self).__init__(csvFile,model)
        self.wordVecTable = np.load(wordVecArrayFile + '.npy')
        self.caseNumber = np.load(caseNumberArrayFile + '.npy')

    def __buildVectorFortheDescrition(self,description):
        sentence = self.cleanSentence(description)
        vector = self.buildVector(sentence)
        return vector

    def similarCases(self,description,count):
        print(description)
        vector = self.__buildVectorFortheDescrition(description)
        dotProducts = np.dot(self.wordVecTable,vector)
        caseIndex = np.argsort(dotProducts)
        sortData = np.sort(dotProducts)
        [print(i) for i in sortData]
        bestBetIndex = caseIndex[len(caseIndex) - count:len(caseIndex)]
        #bestBetIndex = caseIndex[:count]
        bestBetIndex = bestBetIndex[::-1]
        bestBet = self.caseNumber[bestBetIndex]
        self.__getCaseDescription(bestBet)



    def __getCaseDescription(self,caseList):
        csvFile = 'report1511501851930.csv'
        df = pd.read_csv(csvFile)
        print(caseList)
        #df.set_index('Case Number', inplace=True)
        for case in caseList:
            print('Case Number : ' + str(case))
            sDf = df[df['Case Number'] == '0'+str(case)]
            sDf = sDf.reset_index()
            try:
                print(sDf.get_value(0,'Description'))
                print('-------------------------------------------------')
            except:
                pass


def buildModel(csvFile): # Make this part of the main class
    SentenceBuilder = MySentences(csvFile)
    sentences = SentenceBuilder.getSentences()
    model = gensim.models.Word2Vec(sentences, min_count=5, size=vectorSize,  window=8)
    model.wv.save_word2vec_format(word2VecModelFile, binary=True)
    return model

def main():
    csvFile = 'report1511501851930.csv'
    model = buildModel(csvFile)
    #model = gensim.models.KeyedVectors.load_word2vec_format('text.model.bin', binary=True)
    #caseVector = TSBuildCaseVector(csvFile,model)
    #caseVector.buildWordVecTable

    similarCases = SimilarCaseGenerator(csvFile,model)
    with open('inputLine.txt', 'r') as f:
        lines = f.readlines()
    print(lines)
    similarCases.similarCases(lines,20)









if __name__ == '__main__':
    main()
