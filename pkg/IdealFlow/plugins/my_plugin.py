# plugins/my_plugin.py
from Automation import PluginInterface

class MyPlugin(PluginInterface):
    def get_actions(self):
        return {
            'my_action': self.my_action
        }

    def my_action(self, param1, param2, **kwargs):
        # Implement the action logic here
        pass