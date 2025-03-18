#plugin.py
"""
IdealFlow
plugin.py v1

@author: Kardi Teknomo
"""
from abc import ABC, abstractmethod
import importlib
import os
from Node import NodeState
import queue
import threading

class PluginInterface(ABC):
    @abstractmethod
    def get_actions(self):
        """Return a dictionary of action names to functions."""
        pass
    
    def get_result_path(self,filename, folder='result'):
        # Get the absolute path to the result folder
        base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), folder)
        # Ensure the result folder exists
        # os.makedirs(base_path, exist_ok=True)
        # Construct the full path
        return os.path.join(base_path, filename)

    def get_unique_filename(self, filename: str) -> str:
        """
        Generate a unique filename by appending a number if the file already exists.

        Parameters:
            filename (str): The initial desired filename.

        Returns:
            str: A unique filename with a number appended if necessary.
        """
        base_path = self.get_result_path(filename)
        name, ext = os.path.splitext(base_path)
        counter = 0

        # Increment the counter until a unique filename is found
        unique_filename = base_path
        while os.path.exists(unique_filename):
            counter += 1
            unique_filename = f"{name}{counter}{ext}"
        
        return unique_filename


# Loading Plugins
class PluginManager:
    def __init__(self, plugin_folder='plugins'):
        self.plugin_folder = os.path.join(os.path.dirname(__file__), plugin_folder)
        self.plugins = {}
        self.load_plugins()

    def register_plugin(self, name, plugin_instance):
        """Manually register a plugin instance with parameters."""
        self.plugins[name] = plugin_instance

    def load_plugins(self):
        # Verify if the folder exists
        if not os.path.isdir(self.plugin_folder):
            raise FileNotFoundError(f"Plugin folder '{self.plugin_folder}' not found.")
        
        for filename in os.listdir(self.plugin_folder):
            if filename.endswith('.py') and not filename.startswith('__'):
                module_name = filename[:-3]
                # module_path = f"{self.plugin_folder.replace(os.sep, '.')}.{module_name}"
                # module_path = f"{self.plugin_folder}.{module_name}"

                # Construct the module path as "plugins.module_name"
                module_path = f"plugins.{module_name}"

                # Import the module
                module = importlib.import_module(module_path)

                # Load plugin class if it implements PluginInterface
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and issubclass(attr, PluginInterface) and attr is not PluginInterface:
                        try:
                            plugin_instance = attr()
                            self.plugins[module_name] = plugin_instance
                        except TypeError:
                            # Skip plugins that require initialization arguments
                            # print(f"Skipping plugin '{module_name}' as it requires arguments.")
                            pass
                        
    def get_action_handlers(self):
        action_handlers = {}
        for plugin in self.plugins.values():
            action_handlers.update(plugin.get_actions())
        return action_handlers

    def release_plugins(self):
        self.plugins.clear()


class ExecutionManager:
    def __init__(self):
        self.nodes = {}
        self.event_queue = queue.Queue()
        self.action_handlers = {}
        self.terminate_flag = threading.Event()
        self.end_nodes = []
        self.is_strongly_connected = False  # flag set by Automation based on IFN
        
    def add_node(self, node):
        self.nodes[node.node_id] = node
    
    def add_action_handler(self, action_name, action_function):
        self.action_handlers[action_name] = action_function

    def run(self):
        # Initialize nodes that can start executing: 
        # trigger nodes differently based on network type.
        if self.is_strongly_connected:
            # For strongly connected networks, only trigger nodes explicitly marked as start.
            for node in self.nodes.values():
                if node.is_start:
                    node.state = NodeState.READY
                    node.execute(self.action_handlers, self.event_queue)
        else:
            # For acyclic networks, trigger nodes that pass the dependency check.
            for node in self.nodes.values():
                if node.can_execute():
                    node.state = NodeState.READY
                    node.execute(self.action_handlers, self.event_queue)

        while not self.terminate_flag.is_set():
            try:
                # Process the event queue.
                event, node = self.event_queue.get(timeout=0.1)
                if event == 'node_completed':
                    print(f"Node {node.node_id} completed.")
                    
                    # Check if completed node is an end node
                    if node.is_end:
                        print("Reached end node, terminating execution")
                        self.terminate_flag.set()
                        
                    # Trigger successors normally.
                    for successor in node.successors:
                        if successor.state == NodeState.WAITING and successor.can_execute():
                            successor.state = NodeState.READY
                            successor.execute(self.action_handlers, self.event_queue)
                elif event == 'node_failed':
                    print(f"Node {node.node_id} failed.")
                    node.execute(self.action_handlers, self.event_queue)
                elif event == 'node_retrying':
                    print(f"Node {node.node_id} retrying ({node.retry_count}/{node.max_retries}).")
                    node.execute(self.action_handlers, self.event_queue)
                elif event == 'node_terminated':
                    print(f"Node {node.node_id} terminated after retries.")
                elif event == 'node_completed_and_waiting':
                    print(f"Node {node.node_id} completed and waiting for next run.")
                    # For both network types, try to trigger successors.
                    for successor in node.successors:
                        if successor.state == NodeState.WAITING and successor.can_execute():
                            successor.state = NodeState.READY
                            successor.execute(self.action_handlers, self.event_queue)
                else:
                    pass  # Handle other events as needed
                self.event_queue.task_done() 
            except queue.Empty:
                # TIMEOUT HANDLING: different policies for strongly connected vs. acyclic.
                if self.is_strongly_connected:
                    # In strongly connected networks, reset only non-end repeatable nodes
                    non_end_repeatable = [n for n in self.nodes.values() if n.repeatable and not n.is_end]
                    if non_end_repeatable and all(n.executed_in_cycle for n in non_end_repeatable):
                        print("Cycle complete (strongly connected). Resetting eligible non-end repeatable nodes for next cycle.")
                        for n in non_end_repeatable:
                            if n.execution_count < n.max_cycles:
                                n.state = NodeState.WAITING
                                n.executed_in_cycle = False
                                n.state = NodeState.READY
                                n.execute(self.action_handlers, self.event_queue)
                            else:
                                print(f"Node {n.node_id} reached max cycles; marking as COMPLETED.")
                                n.state = NodeState.COMPLETED
                    # Termination: if all nodes (including end nodes) are terminal, we finish.
                    if all(n.state in [NodeState.COMPLETED, NodeState.TERMINATED, NodeState.FAILED] 
                           for n in self.nodes.values()):
                        print("Execution completed (strongly connected).")
                        break
                else:
                    # For acyclic networks
                    active_threads = any(n.execution_thread and n.execution_thread.is_alive() for n in self.nodes.values())
                    if not active_threads:
                        # Mark repeatable nodes with no successors as COMPLETED.
                        for n in self.nodes.values():
                            if n.repeatable and not n.successors:
                                n.state = NodeState.COMPLETED
                        if all(n.state in [NodeState.COMPLETED, NodeState.TERMINATED, NodeState.FAILED] 
                               for n in self.nodes.values()):
                            print("Execution completed.")
                            break
                        repeatable_with_successors = [n for n in self.nodes.values() if n.repeatable and n.successors]
                        if repeatable_with_successors and all(n.executed_in_cycle for n in repeatable_with_successors):
                            print("Cycle complete (acyclic). Resetting eligible repeatable nodes for the next cycle.")
                            for n in repeatable_with_successors:
                                if n.execution_count < n.max_cycles:
                                    n.state = NodeState.WAITING
                                    n.executed_in_cycle = False
                                else:
                                    print(f"Node {n.node_id} reached max cycles; marking as COMPLETED.")
                                    n.state = NodeState.COMPLETED

        # Cleanup
        self.force_terminate()
        print("Done.")

    def force_terminate(self):
        # Force termination of all threads when end condition met
        for node in self.nodes.values():
            if node.execution_thread and node.execution_thread.is_alive():
                node.terminate()
                node.execution_thread.join()