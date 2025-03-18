#example_user_data.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../IdealFlow")))
from Automation import Automation

# Initialize the Automation framework
af = Automation("User Data Automation")

# Define your own custom modifier function to override process_data
def custom_modifier(data):
    return data[::-1]  # Example: Reverse the input string

# Define nodes and actions
af.add_node("a", action="get_user_input", params= {"automation":af}, blocking=True)   # to provide data for process_data
af.add_node("b", action="process_data", params= {"automation":af, "modifier": custom_modifier})  # define your own customer function
af.add_node("c", action= "output_data",params= {"automation":af})      # print user input and print output

# Define dependencies
af.assign(["a", "b", "c"])

# Execute the automation sequence
af.execute()
