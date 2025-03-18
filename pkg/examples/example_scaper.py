# example_crawler.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../IdealFlow")))
from Automation import Automation

if __name__=='__main__':
    af = Automation("Scraper Automation")
    # Add nodes for recording control
    af.add_node("a", action='message', params={'text':'Start Node', 'delay': 0.1}, is_start=True)
    af.add_node("b", action='browse', params={'url': "https://github.com/teknomo/IdealFlowNetwork"})
   
    # Define execution order
    af.add_link("a","b")
    
    # Execute automation
    af.execute()
    af.show()
    