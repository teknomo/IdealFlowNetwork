# example_notepad.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../pkg/IdealFlow")))
from Automation import Automation

if __name__=='__main__':
    # Define the file path
    file_path = "notepad.json"

    af = Automation("Notepad Automation Network")
    # Check if the file exists
    if os.path.exists(file_path):
        # If the file exists, load the automation
        af.load(file_path)
    else:
        # If the file doesn't exist, create the network
        # Using SamplePlugin for Notepad
        af.add_node("a", action='run_notepad')
        af.add_node("b", action='type_text_in_notepad', params={'text': 'This is a test in Notepad from automation.'})
        af.add_node("c", action='save_notepad_as', params={'filename': 'test1.txt'})  # First save, sets the saved_file_path
        af.add_node("d", action='save_notepad')  # Save with a pre-set path

        # Define dependencies
        af.assign(["a", "b", "c", "d"])

        # Save the automation for future loading
        af.save(file_path)

    # Execute the automation
    af.execute()

    # Show the IFN structure
    af.show()
    # # Using SamplePlugin for Notepad
    # af.add_node("a", action='run_notepad')
    # af.add_node("b", action='type_text', params={'text': 'This is a test in Notepad from automation.'})
    # # af.add_node("c", action='save_notepad_file', params={'filename': 'test.txt', 'folder':'result'})

    # af.add_node("c", action='save_notepad_as', params={'filename': 'test1.txt'})  # First save, sets the saved_file_path
    # af.add_node("d", action='save_notepad')  # 

    # # Define dependencies
    # af.assign(["a",'b', 'c','d'])
    
    # # Execute the automation
    # af.execute()

    # # show the IFN
    # af.show()