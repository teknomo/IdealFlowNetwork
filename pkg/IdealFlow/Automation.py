from IdealFlow.Network import IFN
from plugin import PluginManager, ExecutionManager
from Node import Node

class Automation(IFN):
    def __init__(self, name="IFN_Automation"):
        super().__init__(name=name)
        self.dict_nodes = {}
        self.plugin_manager = PluginManager(plugin_folder='plugins')
        
        # Initialize Execution Manager
        self.execution_manager = ExecutionManager()

        # Load Plugins
        action_handlers = self.plugin_manager.get_action_handlers()
        for action_name, action_function in action_handlers.items():
            self.execution_manager.add_action_handler(action_name, action_function)
    

    def add_node(self,name, action, params=None, repeatable=False, 
                 dependency_type='AND',is_start=False, is_end=False):
        '''
        override parent addnode
        '''
        super().add_node(name) # Add the node to the IFN graph
        node = Node(node_id=name, 
                    action_name=action, 
                    parameters=params, 
                    repeatable=repeatable,
                    dependency_type=dependency_type,
                    is_start=is_start, 
                    is_end=is_end)
        self.dict_nodes[name]=node


    def execute(self):
        # Set strongly connected property from IFN (inherited) into the execution manager.
        self.execution_manager.is_strongly_connected = self.is_strongly_connected  # CHANGE: pass network property
          
        # Add Nodes to Execution Manager
        for node_id, node in self.dict_nodes.items():
            self.execution_manager.add_node(node)

        # Set up dependencies (edges) based on IFN 
        edges = self.get_links
        for u, v in edges:
            self.dict_nodes[u].add_successor(self.dict_nodes[v])
            self.dict_nodes[v].add_predecessor(self.dict_nodes[u])

        # Run the Execution Manager
        self.execution_manager.run()

        # Release Plugins
        self.plugin_manager.release_plugins()


if __name__=='__main__':
    af = Automation()
    # Using SamplePlugin for Notepad
    af.add_node("a", action='run_notepad')
    af.add_node("b", action='type_text', params={'text': 'This is a test in Notepad.'})
    af.add_node("c", action='save_notepad_as', params={'filename': 'test.txt'})

    # # Using WordPlugin for Word processing
    # af.add_node("d", action='run_word')
    # af.add_node("e", action='type_text', params={'text': 'This is a test in Word.'})
    # af.add_node("f", action='save_document', params={'filename': 'word_test.docx'})

    # # Using ExcelPlugin for Excel
    # af.add_node("g", action='run_excel')
    # af.add_node("h", action='type_text', params={'text': 'Excel data test.'})
    # af.add_node("i", action='save_workbook', params={'filename': 'excel_test.xlsx'})

    # Define dependencies
    af.add_link("a", "b")  # Open Notepad, then type text
    af.add_link("b", "c")  # Type text, then save in Notepad
    # af.add_link("d", "e")  # Open Word, then type text
    # af.add_link("e", "f")  # Type text, then save in Word
    # af.add_link("g", "h")  # Open Excel, then type data
    # af.add_link("h", "i")  # Type data, then save in Excel
    

    # Execute the automation
    af.execute()

    # show the IFN
    af.show()