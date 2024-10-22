# -*- coding: utf-8 -*-
"""
IFN Classifier v15.2
* reusable library

last update: Dec 29, 2021

@author: Kardi Teknomo
"""
import IdealFlow.Network as net
import hashlib
import json
import ast
import os.path
import sys

class Classifier(net.IFN):
    def __init__(self,markovOrder=1,name="classifier"):
        self.IFNs={}       # dictionary IFN ojects
        self.markovOrder=markovOrder  # external parameter
        self.lut={}        # dictionary of {hashcode:varVal} varVal=word in text processing
        self.fName=name    # filename for saving and loading
    
    
    def __str__(self):
        parameter={}
        parameter["lut"]=self.lut
        parameter["Markov"]=self.markovOrder
        parameter["IFNs"]=str(self.IFNs)
        return str(parameter)
    
    
    '''
        public functions of IFN-Classification
    
    * fit(X,y): accuracy of training and predicting at the same data
    * train(X,y): training the self.IFNs
    * predict(X): return category of X
    * generate(category):: return list of var-val from a category
    * show(): show network
    
    '''
    
    
    '''
    save parameters as JSON
    '''
    def save(self):
        fileName=self.fName+".json"
        parameter={}
        parameter["lut"]=self.lut
        parameter["Markov"]=self.markovOrder
        parameter["IFNs"]=str(self.IFNs)
        with open(fileName,'w') as fp:
            json.dump(parameter,fp,sort_keys=True,indent=4)
    
    '''
    load parameters of classifier
    '''
    def load(self):
        fileName=self.fName+".json"
        file_exists = os.path.exists(fileName)
        if file_exists:
            with open(fileName,'r') as fp:
                parameter=json.load(fp)
            self.lut=parameter["lut"]
            self.markovOrder=parameter["Markov"]
            IFNs=ast.literal_eval(parameter['IFNs'])
            self.IFNs={}
            for name,adjList in IFNs.items():
                n=net.IFN(name)
                n.setData(adjList)
                self.IFNs[name]=n



    '''
    return accuracy based on training and testing 
    on the same data
    
    X and y must have the same number of rows
    '''
    def fit(self,X, y):
        self.train(X, y)
        accuracy=self.__test__(X, y)
        return accuracy
    
    
    '''
    predict class or category of X
    
    assume the X data is without header but 
    in the same order as the trained data
    '''
    def predict(self,X):
        mR=len(X)
        yPred=[]
        lstConfidence=[]
        for row in range(mR):
            traj=self.__trajectoryRow__(X,row)                
            y,pctEntropy=self.__predictTrajectory__(traj)
            yPred.append(y)
            lstConfidence.append(pctEntropy)
        return yPred,lstConfidence
    
    
    '''
    training the classification by setting self.IFNs
    
    Xdata and y must have the same number of rows
    '''
    def train(self,X,y):
        mR=len(X)          
        for row in range(0,mR):
            traj=self.__trajectoryRow__(X,row)
            category=y[row]
            ifn=self.__searchIFNs__(category) # search for the correct IFN
            trajSuper=ifn.to_markov_order(traj,toOrder=self.markovOrder)
            ifn.assign(trajSuper) 
    
    
    '''
    return list of variable-value (without cloud node)
    from IFN of a category
    '''
    def generate(self,category):
        ifn=self.__searchIFNs__(category)
        if str(ifn)!='{}':
            trajSuper=ifn.generate()
            traj=ifn.order_markov_lower(trajSuper) # make it first order markov
            traj=self.trajCode2VarVal(traj) # make it varVal
            return traj
        else:
            return None
    
    
    '''
    show the networks of all IFNs or only particular ifn name
    '''
    def show(self,name=""):
        for ifnName,ifn in self.IFNs.items():
            if name=="":
                ifn.show(layout = "Circular")
            elif ifnName==name:
                ifn.show(layout = "Circular")
    
    
    def topNodes(self,ifnName,N):
        try:
            ifn=self.IFNs[ifnName]
            dic=ifn.nodesFlow()
            nodeFlow=self.__sortDicVal__(dic)
            
            res = list(nodeFlow.keys())[0:N]
            highestNodes={}
            for k in res:
                v=nodeFlow[k]
                if k!='#z#':
                    arr=k.split("|")
                    for kk in arr: 
                        w=self.lut[kk]
                        highestNodes[w]=v
            return highestNodes
        except:
            return None
    
    
    '''
    online update of Look Up Table
    input: 
        VarVal is a string of VarVal
        lut = dictionary {hashcode:VarVal}
    output:
        hash code of the VarVal
    algorithm:
        if VarVal exists in lut, return hashcode and previous lut
        if VarVal not exist in lut, create new hash code, update lut and then return hashcode and updated lut
    '''
    def updateLUT(self,VarVal):
        if VarVal in self.lut.values():
            # if VarVal exists in lut
            hashCode=self.find_key_in_dict(VarVal,self.lut)
            return hashCode
        else:
            # if VarVal not exist in lut
            digit=2
            while True:
                code = int(hashlib.sha1(str(VarVal).encode("utf-8")).hexdigest(), 16) % (10 ** digit)
                code1=self.num_to_excel_col(int(str(code)[0]))
                hashCode=code1+str(code)[1:]
                if hashCode not in self.lut:
                    break
                else:
                    digit=digit+1
            
            self.lut[hashCode]=VarVal  # update LUT
            return hashCode
    
    
    '''
    return list of varVal from trajectory
    
    cloud node is removed
    traj must be in first order markov of hash code
    lut = {hashcode: varVal}
    '''
    def trajCode2VarVal(self,traj):
        lst=[]
        for node in traj:
            if node !="#z#":
                if self.lut=={}:
                    lst.append(node)
                else:
                    varVal=self.lut[node]
                    lst.append(varVal)
        return lst


    '''
    return list of hash code from trajectory
    
    cloud node is removed
    traj must be in varVal which is in the value of lut
    lut = {hashcode: varVal}
    '''
    def trajVarVal2Code(self,traj):
        inv_lut=net.IFN.inverse_dict(self.lut)
        lst=[]
        for node in traj:
            if node !="#z#":
                code=inv_lut[node][0]
                lst.append(code)            
        return lst
    
    
    '''
    
        private functions
    
    '''
    
    
    '''
    return accuracy of the prediction
    '''
    def __test__(self,X,y):
        yPred,lstAccuracy=self.predict(X)
        error=0
        mR=len(X)
        for row in range(1,mR):
            category=y[row]
            y_pred=yPred[row]            
            if category!=y_pred:
                error=error+1
        accuracy=1-error/mR
        return accuracy
    
    
    '''
    
    return the correct ifn and the list of IFNs
    for all categories
    
    '''
    def __searchIFNs__(self,category):
        isFound=False
        for name,n in self.IFNs.items():
            c=n.name
            if c==category:
                isFound=True
                ifn=n
        # if not found, create new one
        if isFound==False:
            # hopefully the following code never be executed
            ifn=net.IFN()
            ifn.name=category
            self.IFNs[category]=ifn
        return ifn
    
    
    '''
    return the class prediction of trajectory
    '''
    def __predictTrajectory__(self,trajectory):
        ifn=net.IFN()
        trajSuper=ifn.to_markov_order(trajectory,toOrder=self.markovOrder)
        return ifn.match(trajSuper, self.IFNs)
    
    
    '''    
    return trajectory cycle (of hash code) from a row
    
    '''
    def __trajectoryRow__(self,X,row):
        aRow= self.__takeARow__(X,row)        
        # form a trajectory cycle from variable-value
        lst=["#z#"]  # start from cloud node
        for val in aRow:
            hashCode=self.updateLUT(val)
            lst.append(hashCode)    
        lst.append("#z#") # end with cloud node
        return lst
    
    
    '''
    
        utilities
    
    '''
    # return a list from a row in the given table
    def __takeARow__(self,table,row):
        return table[row]
    
    # return sorted dictionary by value in reverse order
    def __sortDicVal__(self,dic):
        return {k: v for k, v in sorted(dic.items(), key=lambda item: item[1], reverse=True)}



class IFNC():
    def __init__(self):
        self.ifnc={}
        # self.controller(argv)
        # print(argv[1:])
    
    
    '''
    direct the commands
    '''
    def controller(self,argv):
        try:
            if argv[1]=="-t": # training
                name=argv[2]    
                y=argv[3]
                X=argv[4].split()
                if len(argv)>5:
                    mo=int(argv[5])
                else:
                    mo=1
                
                self.ifnc=Classifier(markovOrder=mo,name=name)
                self.ifnc.load()
                self.ifnc.train([X],[y])                
                self.ifnc.save()
                return 1
            elif argv[1]=="-p": # predicting
                name=argv[2]    
                X=argv[3].split()
                
                self.ifnc=Classifier(markovOrder=1,name=name)
                self.ifnc.load()
                estY,lstConfidence=self.ifnc.predict([X])
                print(estY[0],"(",round(lstConfidence[0]*100,2),"%)")
                return estY[0]
            elif argv[1]=="-g": # generating
                name=argv[2]    
                y=argv[3]
                
                self.ifnc=Classifier(markovOrder=1,name=name)
                self.ifnc.load()
                xx=self.ifnc.generate(y)
                print(xx)
                return xx
            else:
                print(self.strHelp())
                return 0
        except IndexError as e:
            if  str(e)=='list index out of range':
                print("lack of arguments")
                print(self.strHelp())
            return 0
        except Exception as e:
            print("General Error = ", str(e))
            return 0
    
   

            
    def strHelp(self):
        strHelp=r"""
        Usage:
            IFNC command name data [parameter]
        
        Description:
            commands:
            -t : training 
                 produce/update parameter file
            -p : predicting 
                 produce label from parameters
            -g : generating 
                 yield random input from parameters
            
            name is the parameter file name 
            (in JSON without .JSON extension)
            
            data: 
              for training: x y 
              for predicting: x  
              for generating: y
            
            x = "one row list separated by white space between double quotes"
            y = a string category label
            
            Markov order parameter is optional.
            The default is 1
            It can be supplied first time during training.
            it must be an integer start from 1.
            
            
        Example:
            -t test y1 "x1 x2" 3
            -t test y2 "x3 x4"
            -p test x1  # produce y1
            -p test x4  # produce y2
            -g test y1  # yield "x1 x2"
            -g test y2  # yield "x3 x4"
            
        
        IFNC - Ideal Flow Network for Classification 
        version 0.15.2
        Copyright (c) 2022 Kardi Teknomo
        https://people.revoledu.com/kardi/
        """
        return strHelp
if __name__=='__main__':
    arg = sys.argv
    ifnc=IFNC()
    ifnc.controller(arg)