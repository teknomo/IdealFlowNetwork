# -*- coding: utf-8 -*-
import numpy as np
import IdealFlow.Classifier as clf


class Table_Classifier(clf.Classifier):
    """
    Table_Classifier class for processing data tables and converting them into matrix X and vector y for classification.
    Inherits from the Classifier class.

    This class provides methods to prepare data tables for classification, extract rows and columns,
    and convert data tables into encoded forms.

    Attributes
    ----------
    markovOrder : int
        The order of the Markov model used for classification (default is 1).
    version : str
        Version of the table processing module.
    variables : list
        List of variable names (extracted from the data table header).

    Methods
    -------
    prepare_data_table(table: list) -> tuple
        Converts the data table into a matrix X and vector y for classification.
    __take_a_column__(table: np.ndarray, col: int) -> np.ndarray
        Extracts a column from the data table.
    __takeARow__(table: np.ndarray, row: int) -> np.ndarray
        Extracts a row from the data table.
    __delete_a_row__(table: np.ndarray, row: int) -> np.ndarray
        Deletes a specified row from the data table.
    __convert__table_to_code__(X: np.ndarray) -> list
        Converts the table into a list of variable-value pairs.
    __delete_a_column__(table: np.ndarray, col: int) -> np.ndarray
        Deletes a specified column from the data table.

    © 2018-2024 Kardi Teknomo

    first build: Jun 1, 2019
    last update: Oct 21,2024
    """
    def __init__(self, markovOrder: int = 1, name: str = "TableClassifier") -> None:
        """
        Initializes the TableProcessing class with a specified Markov order and name.

        Parameters
        ----------
        markovOrder : int, optional
            The order of the Markov model (default is 1).
        name : str, optional
            The name of the table classifier (default is 'TableClassifier').

        Example
        -------
        >>> tp = TableProcessing(markovOrder=2)
        >>> print(tp.version)
        '0.2.1'
        """
        super().__init__(markovOrder,name)   
        self.version="0.2.1"
        self.variables=[]  # list of variable name in an order
        self.markovOrder=markovOrder
    

    def prepare_data_table(self, table: list) -> tuple:
        """
        Converts a data table into matrix X and vector y for classification.

        This method also fills the list of variables from the header row.

        Parameters
        ----------
        table : list
            The input data table, where the last column is the target (y) and the rest are features (X).

        Returns
        -------
        tuple
            A tuple (X, y) where X is the feature matrix and y is the target vector.

        Example
        -------
        >>> tp = TableProcessing()
        >>> table = [['var1', 'var2', 'target'], ['A', 'B', 'Y'], ['C', 'D', 'N']]
        >>> X, y = tp.prepare_data_table(table)
        >>> X
        [['var1:A', 'var2:B'], ['var1:C', 'var2:D']]
        >>> y
        ['Y', 'N']
        """
        table=np.array(table)         # convert to numpy table
        mR,mC=table.shape
        Xdata=self.__delete_a_column__(table, mC-1)  # still include variable name
        y=self.__take_a_column__(table,mC-1)     # still include variable name
        self.variables=self.__takeARow__(table,0) # get list of variable name
        Xdata=self.__delete_a_row__(Xdata, 0)
        y=self.__delete_a_row__(y, 0)
        X=self.__convert__table_to_code__(Xdata)
        return X, y
    
    def __take_a_column__(self, table: np.ndarray, col: int) -> np.ndarray:
        """
        Extracts a column from the data table.

        Parameters
        ----------
        table : np.ndarray
            The input data table.
        col : int
            The index of the column to extract.

        Returns
        -------
        np.ndarray
            The extracted column.

        Example
        -------
        >>> table = np.array([['var1', 'var2', 'target'], ['A', 'B', 'Y'], ['C', 'D', 'N']])
        >>> tp = TableProcessing()
        >>> tp.__take_a_column__(table, 2)
        array(['target', 'Y', 'N'], dtype='<U6')
        """
        return table[:,col]
    

    def __takeARow__(self, table: np.ndarray, row: int) -> np.ndarray:
        """
        Extracts a row from the data table.

        Parameters
        ----------
        table : np.ndarray
            The input data table.
        row : int
            The index of the row to extract.

        Returns
        -------
        np.ndarray
            The extracted row.

        Example
        -------
        >>> table = np.array([['var1', 'var2', 'target'], ['A', 'B', 'Y'], ['C', 'D', 'N']])
        >>> tp = TableProcessing()
        >>> tp.__takeARow__(table, 0)
        array(['var1', 'var2', 'target'], dtype='<U6')
        """
        return table[row]


    def __delete_a_row__(self, table: np.ndarray, row: int) -> np.ndarray:
        """
        Deletes a specified row from the data table.

        Parameters
        ----------
        table : np.ndarray
            The input data table.
        row : int
            The index of the row to delete.

        Returns
        -------
        np.ndarray
            The table with the specified row removed.

        Example
        -------
        >>> table = np.array([['var1', 'var2', 'target'], ['A', 'B', 'Y'], ['C', 'D', 'N']])
        >>> tp = TableProcessing()
        >>> tp.__delete_a_row__(table, 0)
        array([['A', 'B', 'Y'],
               ['C', 'D', 'N']], dtype='<U6')
        """
        return np.delete(table,row,0)
    
    def __delete_a_column__(self, table: np.ndarray, col: int) -> np.ndarray:
        """
        Deletes a specified column from the data table.

        Parameters
        ----------
        table : np.ndarray
            The input data table.
        col : int
            The index of the column to delete.

        Returns
        -------
        np.ndarray
            The table with the specified column removed.

        Example
        -------
        >>> table = np.array([['var1', 'var2', 'target'], ['A', 'B', 'Y'], ['C', 'D', 'N']])
        >>> tp = TableProcessing()
        >>> tp.__delete_a_column__(table, 2)
        array([['var1', 'var2'],
               ['A', 'B'],
               ['C', 'D']], dtype='<U6')
        """
        return np.delete(table,col,1)
    

    def __convert__table_to_code__(self, X: np.ndarray) -> list:
        """
        Converts the data table into a list of variable-value pairs.

        Each element in the table is transformed into a "variable:value" format.

        Parameters
        ----------
        X : np.ndarray
            The input feature matrix.

        Returns
        -------
        list
            A list of variable-value pairs for each row.

        Example
        -------
        >>> tp = TableProcessing()
        >>> tp.variables = ['var1', 'var2']
        >>> X = np.array([['A', 'B'], ['C', 'D']])
        >>> tp.__convert__table_to_code__(X)
        [['var1:A', 'var2:B'], ['var1:C', 'var2:D']]
        """
        newtable=[]
        
        for aRow in X:
            mC=len(aRow)
            lst=[]
            for i in range(mC):
                VarVal=self.variables[i]+ ":" + aRow[i]                
                lst.append(VarVal)
            newtable.append(lst)
        return newtable
    
# END OF class Table_Classifier