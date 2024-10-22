# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 13:58:39 2021
TextProcessing.py v1
Last Udate: Dec 22, 2021

@author: Kardi Teknomo
"""
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
import IFNC.classifier as clf

class TextProcessing():
    def __init__(self, markovOrder=1,name="TextClassifier"):
        self.markovOrder=markovOrder
        self.ifnc=clf.Classifier(markovOrder,name=name)
        
    
    '''
    return list of paragraphs from the given text string
    '''
    def text2Paragraphs(self,text):
        paragraphs=[]
        para=text.split('\n\n')
        for p in para:
            p = p.strip()
            if len(p) > 0:
                paragraphs.append(p)
        return paragraphs
                
        
    
    '''
    return list of sentences from the given text string
    '''
    def text2Sentences(self,text):
        sentences=nltk.tokenize.sent_tokenize(text)
        return sentences
    
    '''
    return the list of tokens (= words, punctuation)
    from a sentence
    '''
    def sentence2Tokens(self,sentence):
        lstTokens = nltk.word_tokenize(sentence)
        return lstTokens
        
    '''
    return a sentence from list of tokens
    '''
    def tokens2Sentence(self,lstTokens):
        sentence=TreebankWordDetokenizer().detokenize(lstTokens)
        sentence=sentence.replace('\"', '"')
        sentence=sentence.replace("''", "")
        sentence=sentence.replace('\"', "")
        sentence=sentence.replace('``', "")
        sentence=sentence.replace(" \ ", "")
        return sentence
        
        
    '''
    conversion of text into matrix X and vector y
    and fill up the list of variable as empty
    
    each sentence is considered as one trajectory.
    each word as a node
    the vector y is just repetition of string category into a vector
    '''
    def prepareTextInput(self,text,category):
        
        sentences=self.text2Sentences(text)
        
        mR=len(sentences)
        self.variables=None
        y=[category]*mR
        X=[]
        for sentence in sentences:
            lstTokens = self.sentence2Tokens(sentence)
            # lstCode=[]
            # for text in lstTokens:
            #     hashCode=self.ifnc.updateLUT(text)
            #     lstCode.append(hashCode)
                
            # X.append(lstCode) 
            X.append(lstTokens)
    
        return X,y

if __name__=='__main__':
    import time
    start_time = time.time()
    
    dataFolder=r"C:\Users\Kardi\Documents\Kardi\Personal\Tutorial\NetworkScience\IdealFlow\Software\Python\Data Science\ReusableData\Text\\"
    # fName="LookingBackward.txt"
    fName="diary.txt"
    f = open(dataFolder+fName,'r')
    textInput=f.read()
    f.close()
    
    # textInput2="Aku mau makan nasi, bukan bubur madura. bukan bubur yang itu. bubur yang ini. Maka dari itu sebabnya."
    # textInput="because the world never been never impossible."
    
    category=fName[:-4]
    print("download time: %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    tp=TextProcessing(markovOrder=2,name=category)
    X,y=tp.prepareTextInput(textInput,category)
    print("prepare input time: %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    
    print('accuracy = ',tp.ifnc.fit(X, y),'\n')
    
    # X,y=tp.prepareTextInput(textInput2,"category2")
    # print('accuracy = ',tp.ifnc.fit(X, y),'\n')
    # lut=tp.ifnc.lut
    # print('\nlut=',lut)
    tp.ifnc.save()
    # # ifnc.show()
    print("training time: %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    # print('Sentence for',category,":\n")
    # sentences=""
    # for i in range(15):
    #     tr=tp.ifnc.generate(category)
    #     sentence=tp.tokens2Sentence(tr)
    #     sentences=sentences+" "+ sentence
    # print(sentences,"\n")
    # print("generating time: %s seconds ---" % (time.time() - start_time))
    
    tp=TextProcessing(markovOrder=2,name=category)
    tp.ifnc.load()
    sentences=""
    for i in range(15):
        tr=tp.ifnc.generate(category)
        sentence=tp.tokens2Sentence(tr)
        sentences=sentences+" "+ sentence
    print(sentences,"\n")
    
    print(tp.ifnc.IFNs['diary'].nodesFlow())
    
    