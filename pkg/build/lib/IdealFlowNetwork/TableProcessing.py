# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 14:09:16 2021
TableProcessing.py v1
Last Update: Dec 22, 2021
@author: Kardi
"""
import numpy as np
import IFNC.classifier as clf

class TableProcessing():
    def __init__(self,markovOrder=1,name="TableClassifier"):
        self.variables=[]  # list of variable name in an order
        self.markovOrder=markovOrder
        self.ifnc=clf.Classifier(markovOrder,name=name)
        
        
        
    '''
    conversion of data table into matrix X and vector y
    and fill up the list of variable
    '''
    def prepareDataTable(self, table):
        table=np.array(table)         # convert to numpy table
        mR,mC=table.shape
        Xdata=self.deleteAColumn(table, mC-1)  # still include variable name
        y=self.__takeAColumn__(table,mC-1)     # still include variable name
        self.variables=self.__takeARow__(table,0) # get list of variable name
        Xdata=self.deleteARow(Xdata, 0)
        y=self.deleteARow(y, 0)
        X=self.convertTable2Code(Xdata)
        return X, y
    
    # return a list from a column in the given table
    def __takeAColumn__(self,table,col):
        return table[:,col]
    
        # return a list from a row in the given table
    def __takeARow__(self,table,row):
        return table[row]

    # return table after deleting a row
    def deleteARow(self,table, row):
        return np.delete(table,row,0)
    
    def convertTable2Code(self,X):
        newtable=[]
        
        for aRow in X:
            mC=len(aRow)
            lst=[]
            for i in range(mC):
                VarVal=self.variables[i]+ ":" + aRow[i]
                # hashCode=self.ifnc.updateLUT(VarVal)
                # lst.append(hashCode)
                lst.append(VarVal)
            newtable.append(lst)
        return newtable
    
    # return table after deleting a column
    def deleteAColumn(self,table, col):
        return np.delete(table,col,1)

if __name__=='__main__':
    dataPath=r"C:\Users\Kardi\Documents\Kardi\Personal\Tutorial\NetworkScience\IdealFlow\Software\Python\Data Science\ReusableData\Supervised\\"
    # ifnc=Classifier.Classifier(markovOrder=1)
    
    lstDataFile=['BooleanXnor','BooleanTF','BooleanTautology',
                  'BooleanProjectionX2','BooleanProjectionX1',
                  'BooleanNor','BooleanNegationX2','BooleanNegationX1',
                  'BooleanNand','BooleanMaterialNonImplication',
                  'BooleanImplication','BooleanConverseNonImplication',
                  'BooleanConverseImplication','BooleanContradiction',
                  'BooleanAnd','BooleanOR','BooleanXOR',
                  'buyComputer','NumberOCR','TransportationMode',
                  'Mammal','Mammal2'
                  ]
    result={}
    for i in range(len(lstDataFile)):
        fName=dataPath+lstDataFile[i]+'.csv'
        tp=TableProcessing(markovOrder=2)
        table=np.array(tp.ifnc.readCSV(fName))
        X, y = tp.prepareDataTable(table)
        # ifnc=Classifier.Classifier(markovOrder=1)
        accuracy = tp.ifnc.fit(X, y)
        result[lstDataFile[i]]=accuracy
    print('result=',result)

    # fName='BooleanTF.csv'      
    # fName='BooleanXnor.csv'
#    fName='BooleanTautology.csv'     
#    fName='BooleanProjectionX2.csv'
#    fName='BooleanProjectionX1.csv'
#    fName='BooleanNor.csv'
#    fName='BooleanNegationX2.csv'
#    fName='BooleanNegationX1.csv'
#    fName='BooleanNand.csv'
#    fName='BooleanMaterialNonImplication.csv'
#    fName='BooleanImplication.csv'
#    fName='BooleanConverseNonImplication.csv'
#    fName='BooleanConverseImplication.csv'
#    fName='BooleanContradiction.csv'
#    fName='BooleanAnd.csv'
#    fName='BooleanOR.csv'
#    fName='BooleanXOR.csv'
    # fName='buyComputer.csv'
#    fName='NumberOCR.csv'
    fName='TransportationMode.csv'
    # fName='Mammal.csv'
    # fName='Mammal2.csv'

    
    print('\n',table,'\n')
    # ifnc=Classifier.Classifier(markovOrder=1)
    tp=TableProcessing(markovOrder=1)
    table=np.array(tp.ifnc.readCSV(dataPath+fName))
    X, y= tp.prepareDataTable(table)
    print('\nX=',X,'\ny=',y)
    
    
    print('accuracy = ',tp.ifnc.fit(X, y),'\n')
    
    # ifnc.show()
    # X=table[1:,:-1]
    # print('X',X)
    # y=list(table[1:,-1])
    print("\nprediction =",tp.ifnc.predict(X),'\n','\ntrue y val=',y,'\n')
    category=y[1]
    tr=tp.ifnc.generate(category)
    print('trajectory for',category,":\n",tr)