import IdealFlow.Classifier as clf
import sys

class CommandLine(clf.Classifier):
    """
    CommandLine class for executing training, prediction, and generation commands via command-line interface.
    Inherits from clf.Classifier.

    This class processes command-line arguments for training, predicting, and generating data using the IFN classifier.

    Attributes
    ----------
    ifnc : clf.Classifier
        An instance of the clf.Classifier class to handle classification.

    Methods
    -------
    controller(argv: list) -> int
        Processes command-line arguments and directs commands for training, prediction, and generation.
    __str_help__() -> str
        Returns a help message explaining how to use the command-line interface.

    Example
    -------
    >>> ifnc = CommandLine()
    >>> ifnc.controller(["-t", "test", "y1", "x1 x2", 3])
    >>> ifnc.controller(["-p", "test", "x1"])
    """

    def __init__(self) -> None:
        """
        Initializes the CommandLine class and prepares an instance of clf.Classifier.

        Example
        -------
        >>> ifnc = CommandLine()
        """
        super().__init__()
        self.ifnc = None  # hold an instance of clf.Classifier
    
    
    def controller(self, argv: list) -> int:
        """
        Processes command-line arguments and directs the appropriate commands for training, prediction, or generation.

        Commands:
        -t : Training command
        -p : Prediction command
        -g : Generation command

        Parameters
        ----------
        argv : list
            List of command-line arguments. Should include the command, classifier name, data, and optional parameters.

        Returns
        -------
        int
            1 on success, 0 on failure.

        Raises
        ------
        IndexError
            If there are insufficient command-line arguments.
        Exception
            For any general error during command execution.

        Example
        -------
        >>> ifnc = CommandLine()
        >>> ifnc.controller(["-t", "test", "y1", "x1 x2", "3"])
        >>> ifnc.controller(["-p", "test", "x1"])
        >>> ifnc.controller(["-g", "test", "y1"])
        """
        try:
            if argv[1]=="-t": # training
                name=argv[2]    
                y=argv[3]
                X=argv[4].split()
                mo = int(argv[5]) if len(argv) > 5 else 1
                if not self.ifnc:
                    # Initialize Classifier only if not already
                    self.ifnc = clf.Classifier(markovOrder=mo, name=name)  
                self.ifnc.load()
                self.ifnc.train([X],[y])                
                self.ifnc.save()
                return 1
            
            elif argv[1]=="-p": # predicting
                name=argv[2]    
                X=argv[3].split()
                
                if not self.ifnc:
                    # Initialize Classifier only if not already
                    self.ifnc = clf.Classifier(markovOrder=1, name=name)  
                self.ifnc.load()
                estY,lstConfidence=self.ifnc.predict([X])
                print(f"{estY[0]} ({round(lstConfidence[0] * 100, 2)}%)")
                return estY[0]
            
            elif argv[1]=="-g": # generating
                name=argv[2]    
                y=argv[3]
                if not self.ifnc:
                    # Initialize Classifier only if not already
                    self.ifnc = clf.Classifier(markovOrder=1, name=name)  
                self.ifnc.load()
                xx=self.ifnc.generate(y)
                print(xx)
                return xx
            
            else:
                print(self.__str_help__())
                return 0
            
        except IndexError as e:
            if str(e) == 'list index out of range':
                print("Error: insufficient arguments")
                print(self.__str_help__())
            return 0
        
        except Exception as e:
            print(f"General Error: {str(e)}")
            return 0
    
            
    def __str_help__(self) -> str:
        """
        Returns a help message explaining how to use the command-line interface for training, predicting, and generating data.

        Returns
        -------
        str
            A help string explaining the available commands.

        Example
        -------
        >>> ifnc = CommandLine()
        >>> print(ifnc.__str_help__())
        """
        str_help = r"""
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

        Ideal Flow Network 

        Copyright (c) 2024 Kardi Teknomo
        https://people.revoledu.com/kardi/
        """
        return str_help


if __name__ == '__main__':
    arg = sys.argv
    ifnc = CommandLine()
    ifnc.controller(arg)