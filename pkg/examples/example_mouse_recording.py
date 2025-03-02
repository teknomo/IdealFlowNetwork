# example_mouse_recording.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from Automation import Automation
if __name__=='__main__':    
    filename = 'mouse_record.txt'
    af = Automation("Mouse Recording")
    
    # Add nodes for recording control
    af.add_node("a", action='message', params={'text':'Start Node', 'delay': 0.1}, is_start=True)
    af.add_node("b", action='start_recording', params={'start_key': 'tab'})
    af.add_node("c", action='save_recording', params={'filename': filename})    
    af.add_node("d", action='message', params={'text':'End Node', 'delay': 0.1}, is_end=True)
    
    af.assign(["a", "b", "c", "d"])  # Define execution order
    af.execute()                     # Execute automation
    af.show()                        # show network
    
    
    