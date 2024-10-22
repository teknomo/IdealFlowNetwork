Unit Tests
==============

All the unit tests contains examples of how to use the IFN class.

Unit tests are located in `test` folder:

    * `IFN_test_1.py`` contains 131 unit tests, all 100% correctly tested.
    
    * `IFN_test_2.py` contains contains 32 unit tests, all 100% correctly tested.

    * `IFN_test_3.py` contains 75 unit tests, all 100% correctly tested.

To run unit tests automatically:

    >>> 
    # Manually set the path to the test directory
    test_path = os.path.abspath(os.path.join(os.getcwd(), '../test'))
    sys.path.append(test_path)

    >>> 
    # Now you can import the IFN_unit_tests module
    import unittest
    import IFN_test_1
    import IFN_test_2
    import IFN_test_3

    >>>
    # Load the test suite from IFN_unit_tests
    loader = unittest.TestLoader()
    suite1 = loader.loadTestsFromModule(IFN_test_1)
    suite2 = loader.loadTestsFromModule(IFN_test_2)
    suite3 = loader.loadTestsFromModule(IFN_test_3)

    >>>
    # Run the test suite
    runner = unittest.TextTestRunner(verbosity=2)
    print("--- IFN_Tests_1 ----")
    runner.run(suite1)
    print("--- IFN_Tests_2 ----")
    runner.run(suite2)
    print("--- IFN_Tests_3 ----")
    runner.run(suite3)