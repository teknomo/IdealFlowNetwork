# plugins/sample_plugin.py
import time
from Automation import PluginInterface

class SamplePlugin(PluginInterface):
    def get_actions(self):
        return {
            'print_message': self.print_message,
            'wait': self.wait_action,            
            'message': self.message  
        }

    def print_message(self, message, **kwargs):
        print(f"Message: {message}")

    def wait_action(self, duration, **kwargs):
        time.sleep(duration)
    
    def message(self, text, delay, **kwargs):
        print(f"Message: {text}")
        time.sleep(delay)