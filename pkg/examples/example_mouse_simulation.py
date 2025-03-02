# example_mouse_simulation.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from Automation import Automation
if __name__=='__main__':
    folder = r"c:\Users\kardi\Dropbox\CurDocs\IdealFlow\Software\Python\Automation\Experiment\Exp4\result"
    fName = 'mouse_record.txt'
    filename = os.path.join(folder, fName)    
    
    af = Automation("Mouse Simulation")
    
    # Add nodes for recording control
    af.add_node("start", action='message', params={'text':'Start Node', 'delay': 0.1}, is_start=True)
    af.add_node("play_rec", action='play_recording', params={'filename': filename, 'speed': 'fast'})
    af.add_node("end node", action='message', params={'text':'End Node', 'delay': 0.1}, is_end=True)
    
    af.assign(["start","play_rec","end node"])   # Define execution order
    af.execute()                                 # Execute automation
    af.show()                                    # show network
    