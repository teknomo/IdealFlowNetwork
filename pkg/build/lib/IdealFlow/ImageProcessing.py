# -*- coding: utf-8 -*-
"""
OCR.py
Created on Sat Dec 18 13:48:57 2021
last update: Dec 18, 2021

@author: Kardi
"""
import numpy as np
import IdealFlow.Classifier as clf
import csv
import pymsgbox
import os

class BinaryImageProcessing():
     def __init__(self,mR,mC,markovOrder=1,name="BinImageClassifier"):
         self.markovOrder=markovOrder
         self.mR=mR
         self.mC=mC
         self.Xdata=[]   # in hashcode
         self.ydata=[]   # in string
         self.dataFileName=name+str(mR)+"_"+str(mC)+".csv"
         self.ifnc=clf.Classifier(markovOrder,name=name)
         
     '''
     convert binary image array to hash code while updating the LUT
     '''
     def trajImage2Code(self,lst):
        lst2=[]
        for i in range(len(lst)):
            VarVal=str(i)+ ":" + str(lst[i])
            hashCode=self.ifnc.updateLUT(VarVal)
            lst2.append(hashCode)
            # lst2.append(VarVal)
        return lst2
    
     '''
     convert var:val array to binary image array based on the LUT
     '''
     def trajVarVal2Image(self,lst):
        lst2=[]
        for i in range(len(lst)):
            varVal=lst[i]
            if varVal!="#z#":
                var,val=varVal.split(":")
                lst2.append(int(val))
        return lst2
            
     '''
     convert binary image array to VarVal 
     '''
     def trajImage2VarVal(self,lst):
        lst2=[]
        for i in range(len(lst)):
            VarVal=str(i)+ ":" + str(lst[i])
            lst2.append(VarVal)
        return lst2
    
    
     '''
     -------------------------------------------------------
        machine learning operations
     -------------------------------------------------------
     '''
     
     '''
     train whole dataset into IFNs and return the accuracy
     without redefine the classifier
     '''
     def fit(self,markovOrder=1):
         try:
             self.ifnc=clf.Classifier(markovOrder) # redefine
             # self.ifnc.markovOrder=markovOrder
             # Xdata in hash code, yData in string label
             accuracy = self.ifnc.fit(self.Xdata,self.ydata)
             return accuracy
         except:
             return "load data first"
     
         
     '''
     train one list into IFN-label
     input:
         lst is list of a binary image
     '''
     def train(self,lst,label):
         # lst2=self.trajImage2Code(lst)  # convert to list of hashcode
         lst2=self.trajImage2VarVal(lst)
         self.Xdata.append(lst2)
         self.ydata.append(label)
         self.ifnc.train([lst2],[label])
     
        
     '''
     predict the label of lst
     input:
         lst is list fo a binary image
         
     '''
     def predict(self,lst):
         if lst:
             # lst1=self.trajImage2Code(lst) # convert to list of hashcode
             lst1=self.trajImage2VarVal(lst)
             yPred,lstConfidence=self.ifnc.predict([lst1])
             return yPred[0],lstConfidence[0]
     
     '''
     generate image based on IFN-label
     output:
         list of a binary image
     '''
     def generate(self,label):
         tr=self.ifnc.generate(label) # in var-val (without cloud node)
         if tr!=None:
             lst=self.trajVarVal2Image(tr)         # convert varVal to binary
             return lst
         else:
             return None
     
     '''
     return optimum markov order
     use linear order
     '''
     def optimumMarkovOrder(self):
         optMarkovOrder=-1
         maxAccuracy=0
         dic={}
         for order in range(1,self.mR*self.mC):
             accuracy=self.fit(order)
             if accuracy<1.0:
                 if maxAccuracy<accuracy:
                    maxAccuracy=accuracy
                 optMarkovOrder=order
                 dic[order]=maxAccuracy
             else:
                 maxAccuracy=accuracy
                 optMarkovOrder=order
                 dic[order]=maxAccuracy
                 break
             
         return optMarkovOrder, maxAccuracy#, dic
     
     '''
     return optimum markov order and accuracy
     use binary search
     '''
     def optimumMarkovOrder2(self):
         low  = 1
         high = self.mR*self.mC         
         while low <= high:
             
             mid = (high + low) // 2
             lA = self.fit(low)
             mA = self.fit(mid)
             if lA<mA<1:
                 low=mid+1
             elif mA>=1.0:
                 high=mid-1
             else:
                 break
         if mA<1.0:
             return mid+1,self.fit(mid+1)
         else:
             return mid,mA
         
     
     
     '''
     -------------------------------------------------------
        data operations
     -------------------------------------------------------
     '''
     
     '''
     fill up self.data
     
     data is save in binary except the label
     Xdata is in hashcode
     we convert the binary into hash code except the label
     '''
     def loadData(self):
         self.Xdata=[]
         self.ydata=[]
         if os.path.isfile(self.dataFileName):
             table = list(csv.reader(open(self.dataFileName)))
             for aRow in table:
                 binImg=aRow[:-1]
                 label=aRow[-1]
                 # tr=self.trajImage2Code(binImg)
                 tr=self.trajImage2VarVal(binImg)
                 self.Xdata.append(tr)
                 self.ydata.append(label)
             return len(table)
         else:
             return -1
         
         
     
        
     '''
     save self.data to self.dataFileName
     self.Xdata is in hash code
     we convert hash code to binary 
     then we add with label
     before saving
     the saved data would be in binary and label in string
     '''
     def saveData(self):
         retVal=pymsgbox.confirm("This action would replace the existing data file. Are you sure?")
         if retVal=="OK":
             data=[]
             for idx,aRow in enumerate(self.Xdata):
                 # lst=self.ifnc.trajCode2VarVal(aRow)  # from code to varVal
                 tr=self.trajVarVal2Image(aRow)                # from varVal to binary
                 label=self.ydata[idx]
                 tr.append(label)
                 data.append(tr)
                 
             data=np.asarray(data)
             np.savetxt(self.dataFileName, data, delimiter=",", fmt="%s")
             return len(data)
        
        
     '''
     return list of data of current id
     
     self.data is in hash code
     
     '''
     def getData(self,id):
         try:
             trVV=self.Xdata[id]
             # trCode=self.Xdata[id]
             # trVV=self.ifnc.trajCode2VarVal(trCode) # from code to varVal
             trBin=self.trajVarVal2Image(trVV)               # var-val to binary
             X=trBin
             y=self.ydata[id]
             return X,y
         except:
             return self.zeroOperation(),-1
     
     '''
     return the number rows in the data
     '''
     def totalData(self):
         return len(self.ydata)
     
     '''
     delete data from IFN and from the self.data
     '''
     def deleteData(self,lst,label):
         pass
        
        
     '''
     -------------------------------------------------------
        drawing operations
     -------------------------------------------------------
     '''
     
     '''
     return list after inverse operation 0->1; 1->0
     '''
     def zeroOperation(self):
         return [0]*self.mR*self.mC
     
     
     '''
     return list after inverse operation 0->1; 1->0
     '''
     def inverseOperation(self,lst):
         lst2=[]
         for i in range(len(lst)):
             lst2.append(lst[i]^1)
         return lst2
     
     '''
     return list after shift up operation 
     '''
     def shiftUp(self,lst):
         return self.shift(lst,1)
     
        
     '''
     return list after shift down operation 
     '''
     def shiftDown(self,lst):
         return self.shift(lst,-1)
     
        

     '''
     return list after shift left operation 
     '''
     def shiftLeft(self,lst):
         return self.shift(lst,self.mR)
     
     
     '''
     return list after shift right operation
     '''
     def shiftRight(self,lst):
         return self.shift(lst,-self.mR)
     
     
     '''
     return list after glide up right
     '''
     def glideUp(self,lst):
         return self.shift(lst,-self.mC)
         
     
    
     '''
     return list after glide down left
     '''
     def glideDown(self,lst):
         return self.shift(lst,self.mC)
         # for i in range(1):
         #      lst=self.shift(lst,self.mR)
         #      lst=self.reverseIndex(lst)
         #      # lst=self.transpose(lst)
         #      # lst=self.reverseIndex(lst)
         #     # 
         # return lst

     '''
     return list after flip vertical
     '''
     def flipVert(self,lst):
         lst=self.shift(lst,self.mC) #1 this 2 steps
         lst=self.reverseIndex(lst)  #2 of flip vert but shifted
         lst=self.shift(lst,1)       # shiftUp         
         for i in range(4):
              lst=self.shift(lst,self.mR) # shift left 
         return lst

     '''
     return list after flip horizontal
     '''
     def flipHorz(self,lst):
        for i in range(13):
            lst=self.reverseIndex(lst)
            lst=self.swap(lst,self.mC,self.mR)
            # lst=self.modSeq(lst,int(self.mC))
            
        # lst=self.shift(lst,int(self.mR/2))
            # lst=self.transpose2(lst)
        # this to reverse horizontal but shifted
        # for i in range(13):
        #     # lst=self.reverseIndex(lst)
        #     lst=self.transpose2(lst)
        #     lst=self.reverseIndex(lst)
        # for i in range(3):
        #     lst=self.shift(lst,1)         # shiftUp 
        # for i in range(3):
        #       lst=self.shift(lst,self.mR) # shift left 
        return lst
     

     def rndPic(self):
         arr = np.random.randint(2, size=(self.mR*self.mC,))
         # lst=random.randint(0, 2*self.mR*self.mC - 1)
         return arr

     '''
     ------------------------------------------------
             reusable libraries
     ------------------------------------------------
     '''
    
     
     
     '''
     return row,col location of a cell index in a matrix size mR,mC
     assume natural matrix filled by col
     index start from 0, (row,col) start from 0
     '''
     def idx2rowcol(self,idx):
         row=idx % self.mR          # mod (idx,mR)
         col=int((idx/self.mR)//1)  # floor (idx/mR)
         return (row,col)
     
     '''
     return index location of a cell in a matrix size mR,mC
     assume natural matrix filled by col
     index start from 0, (row,col) start from 0
     '''
     def rowcol2idx(self,row,col):
         return self.mR*col+row
     
        
     def reverseIndex(self,seq):
         n=len(seq)
         lst=[0]*n
         for idx,content in enumerate(seq):
             idx2=n-idx-1       # reverse the index
             lst[idx2]=content
         return lst
     
     def transpose(self,seq):
          lst=[0]*len(seq)
          for idx,content in enumerate(seq):
              (row,col)=self.idx2rowcol(idx)
              idx2=self.mC*row+col #transpose
              lst[idx2]=content
          return lst
     
     def transpose2(self,seq):
         lst=[0]*len(seq)
         for idx,content in enumerate(seq):
             (row,col)=self.idx2rowcol(idx)
             idx2=self.mC*row+col #transpose
             (row2,col2)=self.idx2rowcol(idx2)
             idx3=self.mR*col2+row2 #transpose
             lst[idx3]=content
         return lst
     
     def rotate(self,seq,n):
         return seq[n:]+seq[:n]
     
     '''
     source: https://stackoverflow.com/questions/2150108/efficient-way-to-rotate-a-list-in-python
     '''
     def shift(self,seq, n):
         n = n % len(seq)
         return seq[n:] + seq[:n]
     
     def swap(self,seq,n1,n2):
         start=seq[:n1]
         mid=seq[n1:n2]
         last=seq[n2:]
         return last+mid+start
    
     def modSeq(self,seq,n):
          #preparation
          lst=[]
          for i in range(abs(n)):
              lst2=[]
              lst.append(lst2)
              
          for idx,content in enumerate(seq):
              group=abs(idx % n)
              lstGrp=lst[group]
              lstGrp.append(content)
              lst[group]=lstGrp  # replace
          # flatten lst
          lst3=[]
          for i in range(abs(n)-1,-1,-1): # reverse group
          # for i in range(abs(n)): # reverse group
              lstGrp=lst[i]
              lstGrp=self.reverseIndex(lstGrp)
              # for cell in lstGrp[::-1]: # reverse per group
              #     lst3.append(cell)
          return lst3
     
     def floorSeq(self,seq,n):
          lst=[]
          for idx,content in enumerate(seq):
              idx2= int((idx/n)//1)
              lst[idx2]=content
          return lst
     
     def ceilSeq(self,seq,n):
          lst=[]
          for idx,content in enumerate(seq):
              idx2= idx // n
              lst[idx2]=content
          return lst
    
if __name__ == "__main__":
    import time
    start_time = time.time()
    mR=10
    mC=8
    bip=BinaryImageProcessing(mR,mC,name="ocrData_")
    totalData=bip.loadData()
    print("totalData = ",totalData)
    print("Wait.. still computing the optimization....")
    optMarkovOrder, maxAccuracy=bip.optimumMarkovOrder()
    print("Optimum Markov Order = "+str(optMarkovOrder)+" with accuracy {:.2f}".format(maxAccuracy*100)+"%")
    print("optimization history")
    # for k,v in dic.items():
    #     print(k,v*100)    
    print("--- computation time: %s seconds ---" % (time.time() - start_time))
    # print('opt=',bip.optimumMarkovOrder())
    