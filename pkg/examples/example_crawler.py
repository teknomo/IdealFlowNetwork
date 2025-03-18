# example_crawler.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../IdealFlow")))
from Automation import Automation

if __name__=='__main__':
    
    af = Automation("Crawler Automation")
    
    # Add nodes for recording control
    af.add_node("a", action='message', params={'text':'Start Node', 'delay': 0.1}, is_start=True)    
    af.add_node("b", action='crawl', params={'start_url': "https://github.com/teknomo/IdealFlowNetwork", 'max_size': 5})
    # af.add_node("c", action='generate_site_map', params={'start_url': "https://github.com/teknomo/IdealFlowNetwork", 'filename':'sitemap.xml'})    
    af.add_node("d", action='message', params={'text':'End Node', 'delay': 0.1}, is_end=True)
    
    # Define execution order
    af.assign(["a","b","d"])
    
    # Execute automation
    af.execute()
    af.show()
    