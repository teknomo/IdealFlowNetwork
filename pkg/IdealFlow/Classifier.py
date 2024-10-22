# -*- coding: utf-8 -*-
"""
IFN Classifier v15.2

Classifier class for IFN (Ideal Flow Network) based classification with Markov model support.

    Attributes:
    -----------
    markovOrder : int
        The order of the Markov model used for classification (default is 1).
    version : str
        Version of the classifier.
    IFNs : dict
        A dictionary storing Ideal Flow Network (IFN) objects for each category.
    lut : dict
        Look-up table mapping hash codes to variable values (useful for text processing).
    fName : str
        Filename for saving and loading the classifier.

    Methods:
    --------
    __str__() -> str:
        Returns a string representation of the classifier parameters.
        
    save() -> None:
        Saves the classifier's parameters as a JSON file.

    load() -> None:
        Loads the classifier's parameters from a JSON file.

    train(X: list, y: list) -> None:
        Trains the classifier on the provided dataset X and labels y.

    predict(X: list) -> str:
        Predicts the category for the provided dataset X.

    fit(X: list, y: list) -> float:
        Trains the classifier and calculates prediction accuracy.

    generate(category: str) -> list:
        Generates a list of variable-values for a given category.

    show() -> None:
        Displays the IFN network structure.

    top_nodes(n: int = 10) -> dict:
        Returns the top n nodes by frequency.

    update_lut(val: str) -> str:
        Updates the lookup table (lut) with a hash code for a variable-value.

    trajCode2VarVal(trajectory: list) -> list:
        Converts a trajectory of hash codes back to their variable-values.

    trajVarVal2Code(trajectory: list) -> list:
        Converts a trajectory of variable-values to their hash codes.

    Example:
    --------
    >>> clf = Classifier(markovOrder=2, name="text_classifier")
    >>> print(clf)
    {'lut': {}, 'Markov': 2, 'IFNs': '[]'}

    Copyright:
        © 2018-2024 Kardi Teknomo

    first build: 
        May 31, 2019
    last update: 
        Oct 21,2024
"""
import IdealFlow.Network as net
import hashlib
import json
import ast
import os.path

class Classifier(net.IFN):
    def __init__(self, markovOrder: int = 1, name: str = "classifier") -> None:
        """
        Initializes the Classifier with a specified Markov order and name.

        Parameters:
        -----------
        markovOrder : int, optional
            The order of the Markov model (default is 1).
        name : str, optional
            The name of the classifier (default is 'classifier').

        Example:
        --------
        >>> clf = Classifier(markovOrder=2, name="text_classifier")
        >>> clf.markovOrder
        2
        """
        super().__init__(name) # Initialize the superclass
        self.version="0.15.2"
        self.IFNs={}       # dictionary IFN ojects
        self.markovOrder=markovOrder  # external parameter
        self.lut={}        # dictionary of {hashcode:varVal} varVal=word in text processing
        self.inv_lut = {}  # Dictionary of {VarVal:hashCode}
        self.fName=name    # filename for saving and loading
        
    def __str__(self) -> str:
        """
        Returns a string representation of the classifier's parameters.

        Returns:
        --------
        str
            String representation of the classifier's lookup table, Markov order, and IFNs.

        Example:
        --------
        >>> clf = Classifier(markovOrder=2)
        >>> str(clf)
        {'lut': {}, 'Markov': 2, 'IFNs': '{}'}
        """
        parameter = {
            "lut": self.lut,
            "Markov": self.markovOrder,
            "IFNs": str(self.IFNs)
        }
        return str(parameter)
    
    
    '''
        public functions of IFN-Classification
    
    * fit(X,y): accuracy of training and predicting at the same data
    * train(X,y): training the self.IFNs
    * predict(X): return category of X
    * generate(category):: return list of var-val from a category
    * show(): show network
    
    '''
    
    
    def save(self) -> None:
        """
        Saves the classifier's parameters to a JSON file.

        The parameters include the lookup table (lut), the Markov order, and the IFN dictionary.

        Raises:
        -------
        OSError
            If the file cannot be saved.

        Example:
        --------
        >>> clf = Classifier()
        >>> clf.save()
        File 'classifier.json' will be created.
        """
        fileName = self.fName + ".json"
        parameter = {
            "lut": self.lut,
            "Markov": self.markovOrder,
            "IFNs": str(self.IFNs)
        }

        try:
            with open(fileName, 'w') as fp:
                json.dump(parameter, fp, sort_keys=True, indent=4)
        except OSError as e:
            print(f"Error saving file {fileName}: {e}")
    

    def load(self) -> None:
        """
        Loads the classifier's parameters from a JSON file.

        If the file exists, it loads the lookup table (lut), Markov order, and IFN dictionary.
        If the file does not exist, an error message is displayed.

        Raises:
        -------
        FileNotFoundError
            If the JSON file is not found.

        Example:
        --------
        >>> clf = Classifier()
        >>> clf.load()
        """
        fileName = self.fName + ".json"
        if os.path.exists(fileName):
            try:
                with open(fileName, 'r') as fp:
                    parameter = json.load(fp)
                self.lut = parameter["lut"]
                self.markovOrder = parameter["Markov"]
                IFNs=ast.literal_eval(parameter['IFNs'])
                self.IFNs={}
                for name,adjList in IFNs.items():
                    n=net.IFN(name)
                    n.setData(adjList)
                    self.IFNs[name]=n
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Error loading file {fileName}: {e}")
        else:
            print(f"File {fileName} not found.")


    def fit(self, X: list, y: list) -> float:
        """
        Trains the classifier on the dataset X and labels y, then calculates the accuracy of predictions. X and y must have the same number of rows.

        Parameters:
        -----------
        X : list
            List of input data.
        y : list
            List of correct labels (categories) for the input data.

        Returns:
        --------
        float
            Accuracy of the classifier (ranging from 0 to 1).

        Example:
        --------
        >>> clf = Classifier()
        >>> X = [[0, 1], [1, 0], [1, 1]]
        >>> y = ["A", "B", "A"]
        >>> clf.fit(X, y)
        0.67
        """
        self.train(X, y)
        accuracy=self.__test__(X, y)
        return accuracy
    
    
    def predict_table(self, X: list) -> str:
        """
        Predicts the category label for the provided input data X. The X data is without header but in the same order as the trained data

        Parameters:
        -----------
        X : list
            Input data for which to predict the category.

        Returns:
        --------
        str
            Predicted category for the input data.

        Example:
        --------
        >>> clf = Classifier()
        >>> X = [[0, 1], [1, 0]]
        >>> clf.predict(X)
        'A'
        """
        # trajectory = self.__trajectoryRow__(X, 0)
        # return self.predict(trajectory)
        mR=len(X)
        yPred=[]
        lstConfidence=[]
        for row in range(mR):
            traj=self.__trajectoryRow__(X,row)                
            y,pctEntropy=self.predict(traj)
            yPred.append(y)
            lstConfidence.append(pctEntropy)
        return yPred,lstConfidence
    
    
    def train(self, X: list, y: list) -> None:
        """
        Trains the classifier on the provided dataset  by setting self.IFNs. X and y must have the same number of rows.

        Parameters:
        -----------
        X : list
            List of input data.
        y : list
            List of corresponding labels (categories) for the input data.        

        Example:
        --------
        >>> clf = Classifier()
        >>> X = [[0, 1], [1, 0], [1, 1]]
        >>> y = ["A", "B", "A"]
        >>> clf.train(X, y)
        """
        for i in range(len(X)):
            mR=len(X)          
            for row in range(0,mR):
                traj=self.__trajectoryRow__(X,row)
                category=y[row]
                ifn=self.__searchIFNs__(category) # search for the correct IFN
                trajSuper=ifn.to_markov_order(traj,toOrder=self.markovOrder)
                ifn.assign(trajSuper)             
    

    def predict(self, trajectory: list)  -> tuple:
        """
        Predicts the class for the given trajectory of variable-values. It return the class prediction of trajectory.

        Parameters:
        -----------
        trajectory : list
            List of variable-values representing a trajectory.

        Returns:
        --------
            tuple: The predicted category and the confidence percentage (float). The Predicted category for the trajectory is the name (str) of the IFN with the maximum entropy.

        Example:
        --------
        >>> clf = Classifier()
        >>> clf.predict(['var1', 'var2'])
        ('A',0.62)
        """
        ifn=net.IFN()
        trajSuper=ifn.to_markov_order(trajectory,toOrder=self.markovOrder)
        return ifn.match(trajSuper, self.IFNs)
    

    '''
    return list of variable-value (without cloud node)
    from IFN of a category
    '''
    def generate(self, category: str) -> list:
        """
        Generates a list of variable-value pairs for a given category. It returns list of variable-value (without cloud node) from IFN of a category 

        Parameters:
        -----------
        category : str
            The category for which to generate variable-values.

        Returns:
        --------
        list
            List of variable-values belonging to the given category.

        Example:
        --------
        >>> clf = Classifier()
        >>> clf.generate('A')
        ['var1', 'var2', ...]
        """
        ifn=self.__searchIFNs__(category)
        if str(ifn.adjList)!='{}':
            trajSuper=ifn.generate()
            traj=ifn.order_markov_lower(trajSuper) # make it first order markov
            traj=self.trajCode2VarVal(traj) # make it varVal
            return traj
        else:
            return None
    

    def show(self, name="") -> None:
        """
        Displays the IFN network structure. It show the networks of all IFNs or only particular ifn name

        Example:
        --------
        >>> clf = Classifier()
        >>> clf.show()
        Displays the network structure in a graphical or textual format.
        """
        for ifnName,ifn in self.IFNs.items():
            if name=="":
                ifn.show(layout = "Circular")
            elif ifnName==name:
                ifn.show(layout = "Circular")
    
    
    def top_nodes(self, ifnName: str, N: int = 10) -> dict:
        """
        Returns the top n nodes by frequency from the classifier's IFNs.

        Parameters:
        -----------
        n : int, optional
            Number of top nodes to return (default is 10).

        Returns:
        --------
        dict
            Dictionary of the top n nodes and their frequencies.

        Example:
        --------
        >>> clf = Classifier()
        >>> clf.top_nodes(5)
        {'node1': 15, 'node2': 13, ...}
        """
        try:
            ifn=self.IFNs[ifnName]
            dic=ifn.nodesFlow()
            nodeFlow=self.__sortDicVal__(dic)
            
            res = list(nodeFlow.keys())[0:N]
            highestNodes={}
            for k in res:
                v=nodeFlow[k]
                if k!=self.cloud_name:
                    arr=k.split("|")
                    for kk in arr: 
                        w=self.lut[kk]
                        highestNodes[w]=v
            return highestNodes
        except:
            return None
    
    
    def update_lut(self, VarVal: str) -> str:
        """
        Updates the lookup table (lut) by adding a hash code for the given variable-value.

        Algorithm:
            If VarVal exists in lut, return hashcode and previous lut.
            If VarVal not exist in lut, create new hash code, update lut and then return hashcode and updated lut.

        Parameters:
        -----------
        VarVal : str
            Variable-value to hash and add to the lookup table.
        
        Internal Variable:
            self.lut = dictionary {hashcode:VarVal}

        Returns:
        --------
        str
            The hash code for the given variable-value.

        Example:
        --------
        >>> clf = Classifier()
        >>> clf.update_lut('var1')
        '3f2b5e...'
        """
        
        if VarVal in self.inv_lut:
            # VarVal exists, return existing hashCode
            hashCode = self.inv_lut[VarVal]
            return hashCode
        # elif VarVal in self.lut.values():
        #     # if VarVal exists in lut
        #     hashCode=self.find_key_in_dict(VarVal,self.lut)
        #     return hashCode
        else:
            # if VarVal not exist in lut
            digit=2
            while True:
                code = int(hashlib.sha1(str(VarVal).encode("utf-8")).hexdigest(), 16) % (10 ** digit) + 1
                code1=self.num_to_excel_col(int(str(code)))
                hashCode=code1+str(code)[1:]
                if hashCode not in self.lut:
                    break
                else:
                    digit=digit+1
            
            self.lut[hashCode]=VarVal  # update LUT
            self.inv_lut[VarVal] = hashCode
            return hashCode
    
    
    def trajCode2VarVal(self, trajectory: list) -> list:
        """
        Converts a trajectory of hash codes into a list of corresponding variable-values. It returns list of varVal from trajectory where the cloud node is removed.
        The trajectory must be in first order markov of hash code. The lut = {hashcode: varVal}

        Parameters:
        -----------
        trajectory : list
            List of hash codes representing a trajectory.

        Returns:
        --------
        list
            List of corresponding variable-values for the hash codes.

        Example:
        --------
        >>> clf = Classifier()
        >>> clf.trajCode2VarVal(['3f2b5e...', '1a2b3c...'])
        ['var1', 'var2']
        """
        lst=[]
        for node in trajectory:
            if node !=self.cloud_name:
                if self.lut=={}:
                    lst.append(node)
                else:
                    varVal=self.lut[node]
                    lst.append(varVal)
        return lst


    def trajVarVal2Code(self, trajectory: list) -> list:
        """
        Converts a trajectory of variable-values into a list of corresponding hash codes. It returns list of hash code from trajectory where the cloud node is removed.
        The trajectory must be in first order markov of hash code. The lut = {hashcode: varVal}

        Parameters:
        -----------
        trajectory : list
            List of variable-values.

        Returns:
        --------
        list
            List of corresponding hash codes for the variable-values.

        Example:
        --------
        >>> clf = Classifier()
        >>> clf.trajVarVal2Code(['var1', 'var2'])
        ['3f2b5e...', '1a2b3c...']
        """
        # inv_lut=net.IFN.inverse_dict(self.lut)
        lst=[]
        for node in trajectory:
            if node !=self.cloud_name:
                try:
                    code=self.inv_lut[node]
                    lst.append(code) 
                except (KeyError, IndexError):
                     # Code does not exist, skip this node
                    continue       
        return lst
    
    
    '''
    
        private functions
    
    '''
    
    
    
    def __test__(self,X,y):
        '''
        return accuracy of the prediction
        '''
        yPred,lstAccuracy=self.predict_table(X)
        error=0
        mR=len(X)
        for row in range(1,mR):
            category=y[row]
            y_pred=yPred[row]            
            if category!=y_pred:
                error=error+1
        accuracy=1-error/mR
        return accuracy
    
    
    
    def __searchIFNs__(self, category: str):
        """
        Searches for the IFN associated with a given category. It returns the correct ifn and the list of IFNs for all categories

        Parameters:
        -----------
        category : str
            Category for which to search or create an IFN.

        Returns:
        --------
        ifn
            The corresponding IFN for the category.

        Example:
        --------
        >>> clf = Classifier()
        >>> clf.__searchIFNs__('A')
        Returns the IFN object for category 'A'.
        """
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
    
    
    
    
    
    
    def __trajectoryRow__(self, X: list, row: int) -> list:
        """
        Returns a trajectory cycle (of hash code) from a given row in the input dataset.

        Parameters:
        -----------
        X : list
            Input dataset.
        row : int
            Row number from which to extract the trajectory.

        Returns:
        --------
        list
            List of variable-values representing the trajectory.

        Example:
        --------
        >>> clf = Classifier()
        >>> clf.__trajectoryRow__(X, 1)
        ['var1', 'var2', ...]
        """
        aRow= self.__takeARow__(X,row)        
        # form a trajectory cycle from variable-value
        lst=[self.cloud_name]  # start from cloud node
        for val in aRow:
            hashCode=self.update_lut(val)
            lst.append(hashCode)    
        lst.append(self.cloud_name) # end with cloud node
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


if __name__=='__main__':
    clf = Classifier()