# Automation.py

from IdealFlow.Network import IFN
import json
from enum import Enum, auto
import threading
import queue
from abc import ABC, abstractmethod
import importlib
import os
from typing import Callable, Optional, Dict, Any, Type, List, Set
import logging


'''=========================================
'*      AUTOMATION MODULE
'*
'*      * class Automation(IFN)
'*
'*      NODE MODULE
'*      PLUGIN MODULE
'**=========================================
'''

class Automation(IFN):
    """
    Automation class that manages nodes and plugins for automating workflows.

    Attributes:
        version (str): The version of the automation.
        dict_nodes (Dict[str, Node]): Dictionary to store nodes.
        description (str): Description of the automation.
        purpose (str): Purpose of the automation.
        data (Dict[str, Any]): Data used for inter-agent communication.
        plugin_manager (PluginManager): Manages the plugins.
        execution_manager (ExecutionManager): Manages node execution.
    
    Example:
        >>> af = Automation(name="Example Automation", description="Test automation", purpose="Testing")
        >>> af.add_node("Node1", "action_name", {"param1": "value1"})
        >>> af.execute()
    """

    def __init__(self, name: str = "IFN_Automation", description: str = "", purpose: str = "") -> None:
        """
        Initialize the Automation instance with basic attributes.

        Parameters:
            name (str): The name of the automation instance.
            description (str): Description of the automation instance.
            purpose (str): Purpose of the automation instance.

        Example:
            >>> af = Automation(name="MyAutomation", description="Automation for testing", purpose="Demo")
        """
        super().__init__(name=name)
        self.version = "0.2.2"
        self.dict_nodes: Dict[str, Node] = {}
        self.description = description
        self.purpose = purpose
        self.data: Dict[str, Any] = {}  # Communication data for agents
        self.plugin_manager = PluginManager(plugin_folder='plugins')
        self.execution_manager = ExecutionManager()
        self.trigger_manager = TriggerManager()

        action_handlers = self.plugin_manager.get_action_handlers() # Load Plugins
        for action_name, action_function in action_handlers.items():
            self.execution_manager.add_action_handler(action_name, action_function)

    '''
    '
    '   Node Handling
    '
    '''

    def add_node(self, name: str, action: Optional[str] = None, params: Optional[Dict[str, Any]] = None, 
                 is_supernode: bool = False, blocking: bool = False) -> None:
        """
        Adds a node to the automation graph with the specified action, parameters, and blocking behavior.

        Parameters:
            name (str): The unique identifier for the node.
            action (str): The action to be executed for this node.
            params (Optional[Dict[str, Any]]): Parameters for the action.
            is_supernode (bool): If True, this node is treated as a supernode containing other nodes.
            blocking (bool): If True, this node’s action will execute in a blocking manner.

        Example:
            >>> automation.add_node("node1", "get_user_input", {"automation": automation}, blocking=True)
        """
        super().add_node(name)  # Add the node to the IFN graph
        node = Node(node_id=name, action_name=action or "default_action", parameters=params, 
                is_supernode=is_supernode, blocking=blocking)
        # node = Node(node_id=name, action_name=action, parameters=params, is_supernode=is_supernode, blocking=blocking)
        self.dict_nodes[name] = node

        # If it's a supernode, add subnodes and dependencies to IFN
        if is_supernode:
            for sub_node in node.sub_nodes:
                self.add_node(sub_node.node_id, sub_node.action_name, sub_node.parameters, 
                            sub_node.is_supernode, sub_node.blocking)
                # Register subnode dependencies in IFN adjacency list
                for predecessor in sub_node.predecessors:
                    self.set_path([predecessor.node_id, sub_node.node_id], 1)  # Link predecessors to subnode


    def clear_nodes(self) -> None:
        """
        Clear all nodes and links in the automation network.

        Example:
            >>> automation = Automation()
            >>> automation.add_node("Node1", "action1")
            >>> automation.clear_nodes()
            >>> print(automation.dict_nodes)  # Output: {}
        """
        self.dict_nodes.clear()  # Clear internal dictionary of nodes
        self.set_data({})  # Clear the network in IFN by setting an empty adjacency list


    '''
    '
    '   Execution Control
    '
    '''

    def start(self) -> None:
        """Start the automation process, applying start triggers and initializing nodes."""
        self.trigger_manager.execute_start_trigger("start_node")
        # Additional start logic...


    def pause(self) -> None:
        """Pause the automation process."""
        self.trigger_manager.execute_end_trigger("pause")
        # Additional pause logic...


    def stop(self) -> None:
        """Stop the automation process gracefully, triggering any end conditions."""
        self.trigger_manager.execute_end_trigger("end_node")
        # Additional stop logic...


    def execute(self) -> None:        
        """
        Execute the automation by adding nodes to the execution manager,
        setting up dependencies, and running the execution manager.

        Example:
            >>> automation = Automation()
            >>> automation.add_node("Node1", "action1")
            >>> automation.add_node("Node2", "action2")
            >>> automation.execute()
        """
        # Add Nodes to Execution Manager
        for node_id, node in self.dict_nodes.items():
            self.execution_manager.add_node(node)

        # Set up dependencies (edges)
        edges = self.get_links
        for u, v in edges:
            self.dict_nodes[u].add_successor(self.dict_nodes[v])
            self.dict_nodes[v].add_predecessor(self.dict_nodes[u])

        # Run the Execution Manager
        self.execution_manager.run()

        # Release Plugins
        self.plugin_manager.release_plugins()



    '''
    '
    '   State Management
    '
    '''

    
    def is_running(self) -> bool:
        """Check if the automation process is currently active."""
        return self.current_state == "running"


    def get_current_state(self) -> str:
        """Retrieve the current state of the automation."""
        return self.current_state


    '''
    '
    '   File Handling
    '
    '''

    def save(self, filename: str) -> None:
        """
        Save the automation network structure to a JSON file.

        Parameters:
            filename (str): The path to the file where the network will be saved.

        Example:
            >>> automation = Automation(name="Example Automation")
            >>> automation.add_node("Node1", "action1", {"param1": "value1"})
            >>> automation.save("network.json")
        """
        data = {
            "meta": {
                "name": self.name,
                "description": self.description,
                "purpose": self.purpose
            },
            "nodes": [
                {
                    "id": node_id,
                    "action": node.action_name,
                    "params": node.parameters
                }
                for node_id, node in self.dict_nodes.items()
            ],
            "links": [
                {"from": u, "to": v, "weight": weight}
                for u, connections in self.get_data().items()
                for v, weight in connections.items()
            ]
        }
        
        with open(filename, "w") as file:
            json.dump(data, file, indent=4)
        print(f"Automation saved to {filename}")


    def load(self, filename: str) -> None:
        """
        Load the automation network from a JSON file.

        Parameters:
            filename (str): The path to the JSON file containing the automation network data.

        Example:
            >>> automation = Automation()
            >>> automation.load("network.json")
        """
        with open(filename, "r") as file:
            data = json.load(file)
        
        # Load meta information
        self.name = data["meta"].get("name", self.name)
        self.description = data["meta"].get("description", "")
        self.purpose = data["meta"].get("purpose", "")
        
        # Clear existing nodes and links
        self.dict_nodes.clear()
        self.clear_nodes()  # Assuming IFN has a method to clear the graph structure

        # Reconstruct nodes
        for node_data in data["nodes"]:
            node_id = node_data["id"]
            action = node_data["action"]
            params = node_data.get("params", {})
            self.add_node(node_id, action, params)
        
        # Reconstruct links
        for link in data["links"]:
            self.add_link(link["from"], link["to"], weight=link.get("weight", 1))
        
        print(f"Automation loaded from {filename}")



    '''
    '
    '   Event Handling
    '
    '''

    def add_event_listener(self, event_name: str, callback: Callable) -> None:
        """Add an event listener for a specific event."""
        if event_name not in self.event_listeners:
            self.event_listeners[event_name] = []
        self.event_listeners[event_name].append(callback)


    def trigger_event(self, event_name: str, *args, **kwargs) -> None:
        """Trigger an event, calling all listeners registered for this event."""
        if listeners := self.event_listeners.get(event_name):
            for callback in listeners:
                callback(*args, **kwargs)

    '''
    '
    '   Logging and Monitoring
    '
    '''

    def log(self, message: str, level: str = "INFO") -> None:
        """Log a message with the specified level."""
        logging.log(getattr(logging, level), message)


    def get_logs(self) -> List[str]:
        """Retrieve log entries for review."""
        # Assuming logs are stored in a list or accessed from a logging service
        return self.logs



    '''
    '
    '   Configuration Management
    '
    '''

    def load_config(self, config_file: str) -> None:
        """Load configuration settings from a file."""
        with open(config_file, "r") as file:
            self.config = json.load(file)


    def save_config(self, config_file: str) -> None:
        """Save the current configuration settings to a file."""
        with open(config_file, "w") as file:
            json.dump(self.config, file, indent=4)


    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Retrieve a configuration value, returning a default if not found."""
        return self.config.get(key, default)


    '''
    '
    '   Dependency Management
    '
    '''

    def add_dependency(self, node1: 'Node', node2: 'Node') -> None:
        """Add a dependency between node1 and node2."""
        node1.add_successor(node2)
        node2.add_predecessor(node1)

    def remove_dependency(self, node1: 'Node', node2: 'Node') -> None:
        """Remove a dependency between node1 and node2."""
        node1.successors.discard(node2)
        node2.predecessors.discard(node1)

    def get_dependencies(self, node: 'Node') -> List['Node']:
        """Retrieve all dependencies for a given node."""
        return list(node.predecessors)



'''=========================================
'*      NODE MODULE
'*      * NodeState(Enum)
'*      * Node
'**=========================================
'''

class NodeState(Enum):
    """
    Enum representing the possible states of a Node during execution.

    Values:
        WAITING: Node is waiting to be processed.
        READY: Node is ready to execute.
        EXECUTING: Node is currently executing.
        COMPLETED: Node has finished execution.
        FAILED: Node execution failed.
        RETRYING: Node is retrying after a failure.
        PAUSED: Node execution is paused.
        TERMINATED: Node execution is terminated.

    Example:
        >>> state = NodeState.READY
        >>> print(state)  # Output: NodeState.READY
    """
    WAITING = auto()
    READY = auto()
    EXECUTING = auto()
    COMPLETED = auto()
    FAILED = auto()
    RETRYING = auto()
    PAUSED = auto()
    TERMINATED = auto()


class Node:
    """
    Represents a node in the automation network, managing state, dependencies,
    and execution of actions. It capable of hierarchical structures when acting as a supernode.

    Attributes:
        node_id (str): Unique identifier for the node.
        action_name (str): Name of the action associated with this node.
        parameters (Dict[str, Any]): Parameters for the action.
        state (NodeState): Current state of the node.
        max_retries (int): Maximum number of retries upon failure.
        retry_count (int): Current retry count.
        predecessors (set): Set of predecessor nodes.
        successors (set): Set of successor nodes.
        successors (Set['Node']): Set of successor nodes.
        sub_nodes (Optional[List['Node']]): List of subnodes if this node is a supernode.
        is_supernode (bool): Indicates if this node is a supernode.
        parent_node (Optional['Node']): Reference to the parent node if applicable.
        metadata (Dict[str, Any]): Additional metadata for the node.
        blocking (Optional bool): False = asynchronous, True = synchronous
        lock (threading.Lock): Lock to manage thread safety.
        execution_thread (Optional[threading.Thread]): Thread for asynchronous execution.
        
    
    Example:
        >>> node = Node(node_id="Node1", action_name="example_action")
        >>> node.add_predecessor(Node("Node0", "start_action"))
        >>> print(node.can_execute())  # Output: False
    """

    def __init__(self, node_id: str, action_name: str, parameters: Optional[Dict[str, Any]] = None, 
                 is_supernode: bool = False, parent_node: Optional['Node'] = None,
                 blocking: bool = False,
                 metadata: Optional[Dict[str, Any]] = None, max_retries: int = 3) -> None:
        """
        Initialize a Node instance.

        Parameters:
            node_id (str): Unique identifier for the node.
            action_name (str): Action associated with the node.
            parameters (Optional[Dict[str, Any]]): Parameters for the action.
            is_supernode: Indicates if this node is a supernode.
            parent_node: Reference to the parent node in a hierarchy.
            blocking (bool): Indicates if this node should execute in a blocking manner (synchronous).
            metadata (Optional[Dict[str, Any]]): Additional metadata for the node.
            max_retries (int): Maximum retries for execution.
        """
        self.node_id = node_id
        self.action_name = action_name
        self.parameters = parameters or {}
        self.is_supernode = is_supernode
        self.sub_nodes = [] if is_supernode else None
        self.blocking = blocking
        self.state = NodeState.WAITING
        self.max_retries = max_retries
        self.retry_count = 0
        self.predecessors = set()
        self.successors = set()
        self.lock = threading.Lock()
        self.execution_thread = None
        self.metadata = metadata or {}
        self.parent_node = parent_node  # attribute for hierarchy traversal


    def add_sub_node(self, node: 'Node') -> None:
        """
        Add a sub-node to this supernode.

        :param node: Node to be added as a sub-node.
        :raises TypeError: If the node is not a supernode.
        """
        if self.is_supernode:
            node.parent_node = self
            self.sub_nodes.append(node)
        else:
            raise TypeError("Only supernodes can contain sub-nodes.")


    def add_predecessor(self, node: 'Node') -> None:
        """
        Add a predecessor node.

        Parameters:
            node (Node): The predecessor node to be added.

        Example:
            >>> node = Node("Node1", "action")
            >>> predecessor = Node("Node0", "start_action")
            >>> node.add_predecessor(predecessor)
        """
        self.predecessors.add(node)


    def add_successor(self, node: 'Node') -> None:
        """
        Add a successor node.

        Parameters:
            node (Node): The successor node to be added.

        Example:
            >>> node = Node("Node1", "action")
            >>> successor = Node("Node2", "end_action")
            >>> node.add_successor(successor)
        """
        self.successors.add(node)


    def can_execute(self) -> bool:
        """
        Check if the node can execute by verifying that all predecessors are completed.

        Returns:
            bool: True if the node can execute, otherwise False.

        Example:
            >>> node = Node("Node1", "action")
            >>> print(node.can_execute())  # Output depends on predecessors' states
        """
        return all(pred.state == NodeState.COMPLETED for pred in self.predecessors)


    def execute(self, action_handlers: Dict[str, Any], event_queue: queue.Queue) -> None:
        """
        Execute the node or recursively execute subnodes if it's a supernode,
        ensuring each node follows the dependency order.
        The `execute` method checks if the node is a supernode and, if so, recursively executes its `sub_nodes`.
        Regular nodes execute their assigned action via `run_action`.

        Parameters:
            action_handlers (Dict[str, Any]): Handlers for node actions.
            event_queue (queue.Queue): Queue for event handling.

        Example:
            >>> node = Node("Node1", "action")
            >>> node.execute(action_handlers, event_queue)
        """
        with self.lock:
            # if self.state not in [NodeState.WAITING, NodeState.RETRYING]:
            #     return
            if self.state in [NodeState.PAUSED, NodeState.TERMINATED, NodeState.COMPLETED, NodeState.EXECUTING]:
                return
            if self.state == NodeState.FAILED and self.retry_count >= self.max_retries:
                self.state = NodeState.TERMINATED
                return
            
            if self.is_supernode:
                # Iterate over sub-nodes in sequence and ensure dependencies are respected.
                for sub_node in self.sub_nodes:
                    # Check if sub_node dependencies are met before executing
                    if sub_node.can_execute():
                        sub_node.execute(action_handlers, event_queue)
                    
            else:
                # Non-supernode, execute the action if dependencies allow.
                if not self.can_execute():
                    self.state = NodeState.WAITING
                    return
                self.state = NodeState.EXECUTING
                self.execution_thread = threading.Thread(target=self.run_action, args=(action_handlers, event_queue))
                self.execution_thread.start()
            # if self.state in [NodeState.PAUSED, NodeState.TERMINATED, NodeState.COMPLETED, NodeState.EXECUTING]:
            #     return
            # if self.state == NodeState.FAILED and self.retry_count >= self.max_retries:
            #     self.state = NodeState.TERMINATED
            #     return            
            # if not self.can_execute():
            #     self.state = NodeState.WAITING
            #     return
            # self.state = NodeState.EXECUTING
            # self.execution_thread = threading.Thread(target=self.run_action, args=(action_handlers, event_queue))
            # self.execution_thread.start()


    def run_action(self, action_handlers: Dict[str, Any], event_queue: queue.Queue) -> None:
        """
        Execute the assigned action, handling main thread requirements and exceptions.

        Parameters:
            action_handlers (Dict[str, Any]): Dictionary of available action functions.
            event_queue (queue.Queue): Queue to handle events related to node status.

        Example:
            >>> node = Node("Node1", "example_action", {"param1": "value"})
            >>> node.run_action(action_handlers, event_queue)
        """
        action_function = action_handlers.get(self.action_name)
        if not action_function:
            self.state = NodeState.FAILED
            event_queue.put(('node_failed', self))
            return
        
        try:
            # Check if the action requires execution on the main thread
            if getattr(action_function, "main_thread_required", False):
                action_function(**self.parameters)
            else:
                action_function(**self.parameters)

            self.state = NodeState.COMPLETED
            event_queue.put(('node_completed', self))

        except Exception:
            self.handle_action_exception(event_queue)
        
    
    def handle_action_exception(self, event_queue: queue.Queue) -> None:
        """
        Handle exceptions during action execution, updating state and retrying if applicable.

        Parameters:
            event_queue (queue.Queue): Queue to handle events related to node retries or termination.

        Example:
            >>> node = Node("Node1", "example_action")
            >>> node.handle_action_exception(event_queue)
        """
        self.state = NodeState.FAILED
        self.retry_count += 1
        if self.retry_count < self.max_retries:
            self.state = NodeState.RETRYING
            event_queue.put(('node_retrying', self))
        else:
            self.state = NodeState.TERMINATED
            event_queue.put(('node_terminated', self))


    def get_depth(self) -> int:
        """
        Calculate the depth level in the hierarchy
        by counting the chain of `parent_node` .

        :return: Depth level as an integer.
        """
        depth = 0
        node = self
        while node.parent_node is not None:
            depth += 1
            node = node.parent_node
        return depth


    def get_root(self) -> 'Node':
        """
        Retrieve the root node of the hierarchy
        by traversing upwards through `parent_node` references.

        :return: The root node.
        """
        node = self
        while node.parent_node is not None:
            node = node.parent_node
        return node
    

    def pause(self) -> None:
        """
        Pause the execution of the node if it is currently in an executable state.

        Example:
            >>> node = Node("Node1", "example_action")
            >>> node.pause()
        """
        with self.lock:
            if self.state == NodeState.EXECUTING and self.execution_thread:
                # Note: Proper thread pausing is complex; here we'll just set the state
                self.state = NodeState.PAUSED


    def resume(self):
        with self.lock:
            if self.state == NodeState.PAUSED:
                self.state = NodeState.READY


    def terminate(self):
        with self.lock:
            self.state = NodeState.TERMINATED



'''=========================================
'*      PLUGIN MODULE
'*
'*      * main_thread_required (DECORATOR)
'*      * class PluginInterface(ABC)
'*      * class PluginManager
'*      * class ExecutionManager
'**=========================================
'''


def main_thread_required(action_func: Callable) -> Callable:
    """
    Decorator to mark actions that need to run on the main thread.

    Parameters:
        action_func (Callable): The action function to mark as main thread required.

    Returns:
        Callable: The original function with an added attribute indicating it requires the main thread.

    Example:
        >>> @main_thread_required
        ... def action_function():
        ...     pass
    """
    action_func.main_thread_required = True
    return action_func


class PluginInterface(ABC):
    """
    Abstract base class for plugins, defining required methods and utilities for action handling.

    Methods:
        get_actions: Abstract method that must return a dictionary mapping action names to functions.
        get_result_path: Utility to generate a full path for saving results.
    
    Example:
        >>> class SamplePlugin(PluginInterface):
        ...     def get_actions(self):
        ...         return {"action_name": self.sample_action}
        ...
        ...     def sample_action(self):
        ...         pass
    """

    @abstractmethod
    def get_actions(self) -> Dict[str, Callable]:
        """
        Return a dictionary of action names mapped to their respective functions.

        Returns:
            Dict[str, Callable]: A dictionary with action names as keys and functions as values.

        Example:
            >>> class SamplePlugin(PluginInterface):
            ...     def get_actions(self):
            ...         return {"sample_action": self.sample_action}
        """
        pass
    

    @staticmethod 
    def get_result_path(filename: str, folder: str = 'result') -> str:
        """
        Generate a full path to save results, ensuring the result folder exists.

        Parameters:
            filename (str): Name of the file to save.
            folder (str): Folder where the file will be saved, default is "result".

        Returns:
            str: Full path to the file in the result folder.

        Example:
            >>> PluginInterface.get_result_path("output.txt")
            '/path/to/result/output.txt'
        """
        # Define the base path two directories up from the current file's directory
        base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), folder)
        
        # Ensure the result folder exists
        if not os.path.exists(base_path): os.makedirs(base_path) 
        
        # Construct the full path to the filename in the result folder
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
    """
    Manages the loading and registration of plugins from a specified directory.

    Attributes:
        plugin_folder (str): Directory path where plugins are located.
        plugins (Dict[str, PluginInterface]): Dictionary of loaded plugins.
    
    Example:
        >>> manager = PluginManager(plugin_folder="plugins")
        >>> manager.load_plugins()
    """

    def __init__(self, plugin_folder: str = 'plugins') -> None:
        """
        Initialize the PluginManager and load plugins from the specified folder.

        Parameters:
            plugin_folder (str): Directory path where plugin files are located.
        
        Example:
            >>> manager = PluginManager("plugins")
        """
        self.plugin_folder = os.path.join(os.path.dirname(__file__), plugin_folder)
        self.plugins: Dict[str, PluginInterface] = {}
        self.action_handlers = {}
        self.simple_name_map = {}  # Maps simple action names to namespaced actions
        self.load_plugins()
        self.print_action_handlers()


    def register_plugin(self, name: str, plugin_instance: Type[PluginInterface]) -> None:
        """
        Manually register a plugin instance.

        Parameters:
            name (str): Unique name for the plugin.
            plugin_instance (PluginInterface): Instance of the plugin to register.

        Example:
            >>> manager = PluginManager()
            >>> manager.register_plugin("custom_plugin", CustomPlugin())
        """
        self.plugins[name] = plugin_instance


    def load_plugins(self) -> None:
        """
        Load plugins from the plugin folder, 
        registering namespaced actions, 
        registering classes that implement PluginInterface.

        Raises:
            FileNotFoundError: If the specified plugin folder does not exist.

        Example:
            >>> manager = PluginManager("plugins")
            >>> manager.load_plugins()
        """
        # Verify if the folder exists
        if not os.path.isdir(self.plugin_folder):
            raise FileNotFoundError(f"Plugin folder '{self.plugin_folder}' not found.")
        
        for filename in os.listdir(self.plugin_folder):
            if filename.endswith('.py') and not filename.startswith('__'):
                module_name = filename[:-3]

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
                            plugin_name = module_name  # Use module name as the plugin's namespace
                            self.register_namespaced_actions(plugin_name, plugin_instance)
                            print(f"loading plugin'{module_name}'.")
                        except TypeError:
                            # Skip plugins that require initialization arguments
                            print(f"Skipping plugin '{module_name}' as it requires initialization arguments.")


    def get_action_handlers(self):
        action_handlers = {}
        for plugin in self.plugins.values():
            action_handlers.update(plugin.get_actions())
        return action_handlers


    def register_namespaced_actions(self, plugin_name: str, plugin_instance: PluginInterface):
        """
        Registers actions with a namespace to avoid conflicts.
        
        Parameters:
            plugin_name (str): The namespace derived from the plugin's module name.
            plugin_instance (PluginInterface): The instance of the plugin.
        """
        actions = plugin_instance.get_actions()
        for action_name, action_function in actions.items():
            # Create a namespaced action identifier
            namespaced_action_name = f"{plugin_name}.{action_name}"
            self.action_handlers[namespaced_action_name] = action_function

            # Register the simple name, checking for conflicts
            if action_name in self.simple_name_map:
                raise ValueError(f"Conflict detected in '{plugin_name}': Action '{action_name}' is already defined by another plugin.")
            self.simple_name_map[action_name] = namespaced_action_name

    def release_plugins(self):
        self.plugins.clear()


    def print_action_handlers(self) -> None:
        """
        Print all available action handlers provided by the loaded plugins.

        Example:
            >>> manager = PluginManager("plugins")
            >>> manager.print_action_handlers()
        """
        action_handlers = self.get_action_handlers()
        print("Loaded Action Handlers:")
        for action_name, action_function in action_handlers.items():
            # Extract the class path in the format plugins.module_name.ClassName
            class_path = f"{action_function.__self__.__module__}.{action_function.__self__.__class__.__name__}"
            print(f"Action: {action_name}, Location: {class_path}")
        print()


class ExecutionManager:
    """
    Manages the execution of nodes, handling actions and events during execution.

    Attributes:
        nodes (Dict[str, Node]): Dictionary of nodes managed by this execution manager.
        event_queue (queue.Queue): Queue for handling events.
        action_handlers (Dict[str, Callable]): Dictionary mapping action names to handler functions.
        max_iterations (int): Maximum number of iterations allowed to prevent infinite loops.
    
    Example:
        >>> manager = ExecutionManager()
        >>> manager.add_node(Node("Node1", "action_name"))
        >>> manager.add_action_handler("action_name", example_action_function)
    """

    def __init__(self) -> None:
        """
        Initialize the ExecutionManager with default settings.
        
        Example:
            >>> manager = ExecutionManager()
        """
        self.nodes: Dict[str, 'Node'] = {}
        self.event_queue = queue.Queue()
        self.action_handlers: Dict[str, Callable] = {}
        self.max_iterations = 10  # Limit to prevent infinite loops


    def add_node(self, node: 'Node') -> None:
        """
        Add a node to the execution manager.

        Parameters:
            node (Node): The node to be added.
        
        Example:
            >>> manager = ExecutionManager()
            >>> manager.add_node(Node("Node1", "action"))
        """
        self.nodes[node.node_id] = node


    def add_action_handler(self, action_name: str, action_function: Callable) -> None:
        """
        Register an action handler for a specific action.

        Parameters:
            action_name (str): Name of the action.
            action_function (Callable): Function to handle the action.
        
        Example:
            >>> manager = ExecutionManager()
            >>> manager.add_action_handler("action_name", example_action_function)
        """
        self.action_handlers[action_name] = action_function


    def run(self):
        # Initialize nodes that can start executing
        for node in self.nodes.values():
            if node.can_execute():
                node.state = NodeState.READY
                node.execute(self.action_handlers, self.event_queue)

        iterations = 0
        while iterations < self.max_iterations:
            try:
                event, node = self.event_queue.get(timeout=1)

                if event == 'node_completed':
                    print(f"Node {node.node_id} completed.")
                    # Trigger successors
                    for successor in node.successors:
                        if successor.state == NodeState.WAITING and successor.can_execute():
                            successor.state = NodeState.READY
                            successor.execute(self.action_handlers, self.event_queue)
                
                elif event == 'node_failed':
                    print(f"Node {node.node_id} failed.")
                    # node.execute(self.action_handlers, self.event_queue)
                    if node.retry_count < node.max_retries:
                        node.retry_count += 1
                        node.state = NodeState.RETRYING
                        self.event_queue.put(('node_retrying', node))
                    else:
                        self.event_queue.put(('node_terminated', node))

                elif event == 'node_retrying':
                    print(f"Node {node.node_id} retrying ({node.retry_count}/{node.max_retries}).")
                    node.execute(self.action_handlers, self.event_queue)
                
                elif event == 'node_terminated':
                    print(f"Node {node.node_id} terminated after retries.")
                    node.state = NodeState.FAILED

                elif event == 'node_skipped':
                    print(f"Node {node.node_id} skipped.")
                    # Mark the node as skipped and proceed to successors
                    node.state = NodeState.SKIPPED
                    for successor in node.successors:
                        if successor.state == NodeState.WAITING and successor.can_execute():
                            successor.state = NodeState.READY
                            successor.execute(self.action_handlers, self.event_queue)

                elif event == 'execution_completed':
                    print("Execution Manager: All nodes have completed successfully.")
                    break  # Terminate the loop since the execution is complete

                elif event == 'execution_failed':
                    print("Execution Manager: Execution failed due to a critical error.")
                    break  # Terminate the loop due to failure

                else:
                    print(f"Unhandled event: {event} for node {node.node_id}")
            
            except queue.Empty:
                # No events to process
                pass
            iterations += 1

        print("Execution Manager: All nodes processed or max iterations reached")


    def run_node(self, node: 'Node') -> None:
        """
        Executes the action for a node. 
        Resolves action by simple name, ensuring unique actions across plugins. 
        If the node is marked as blocking, it will
        execute synchronously.

        Parameters:
            node (Node): The node to execute.
        """
        # Resolve the namespaced action using the simple name map
        namespaced_action = self.plugin_manager.simple_name_map.get(node.action_name)
        if namespaced_action:
            action_function = self.plugin_manager.action_handlers[namespaced_action]
            action_function(**node.parameters)
        else:
            raise ValueError(f"Action '{node.action_name}' not found in registered plugins.")
        
        if node.blocking:
            self._run_blocking(node)
        else:
            self._run_non_blocking(node)


    def _run_blocking(self, node: 'Node') -> None:
        """Executes the node's action in a synchronous (blocking) manner."""
        action = self.action_handlers.get(node.action_name)
        if action:
            action(**node.parameters)  # Execute the action synchronously


    def _run_non_blocking(self, node: 'Node') -> None:
        """Executes the node's action in a non-blocking manner (e.g., in a new thread)."""
        action = self.action_handlers.get(node.action_name)
        if action:
            threading.Thread(target=action, kwargs=node.parameters).start()

'''=========================================
'*      TRIGGER MODULE
'*      * class TriggerManager
'*      
'**=========================================
'''

class TriggerManager:
    def __init__(self):
        self.start_triggers: Dict[str, Callable] = {}
        self.end_triggers: Dict[str, Callable] = {}

    def add_start_trigger(self, trigger_name: str, trigger_func: Callable):
        self.start_triggers[trigger_name] = trigger_func

    def add_end_trigger(self, trigger_name: str, trigger_func: Callable):
        self.end_triggers[trigger_name] = trigger_func

    def execute_start_trigger(self, trigger_name: str):
        if trigger := self.start_triggers.get(trigger_name):
            trigger()

    def execute_end_trigger(self, trigger_name: str):
        if trigger := self.end_triggers.get(trigger_name):
            trigger()