# example_sample_plugin.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../IdealFlow")))
from Automation import Automation
import matplotlib.pyplot as plt


if __name__=='__main__':
    # Close the previously shown figure
    plt.close('all')
    
    af = Automation()

    # Using SamplePlugin for Notepad
    af.add_node("a", action='message', params={'text':'Node a', 'delay': 0.5}, is_start=True)
    af.add_node("b", action='message', params={'text':'Node b', 'delay': 0.5})
    af.add_node("c", action='message', params={'text':'Node c', 'delay': 0.5})
    af.add_node("d", action='message', params={'text':'Node d', 'delay': 0.5})
    af.add_node("e", action='message', params={'text':'Node e', 'delay': 0.5}, is_end=False)
    # OR need can be executed by any of the incoming is True. This is for asynchronous
    af.add_node("f", action='message', params={'text':'Node f', 'delay': 0.5}, dependency_type='OR') 
    af.add_node("g", action='message', params={'text':'Node g', 'delay': 0.1})
    af.add_node("h", action='message', params={'text':'Node h', 'delay': 0.1}, is_end=True)

    
    # Define dependencies
    af.assign(["a",'b', 'c','g', 'h'])
    af.assign(["a",'b', 'd','e', 'f'])
    af.add_link('b','f')
    af.add_link('f','g')
    af.add_link('h','a')
    
    # af.save("sample.json")
    
    # Execute the automation
    af.execute()

    # show the IFN
    af.show("Circular")

   