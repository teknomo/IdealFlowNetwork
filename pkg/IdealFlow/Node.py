from enum import Enum, auto
import threading
import queue

class NodeState(Enum):
    WAITING = auto()
    READY = auto()
    EXECUTING = auto()
    COMPLETED = auto()
    FAILED = auto()
    RETRYING = auto()
    PAUSED = auto()
    TERMINATED = auto()

class Node:
    def __init__(self, node_id, action_name, parameters=None, metadata=None, 
                 max_retries=3, repeatable=False, dependency_type='AND',
                 max_cycles=10, is_start=False, is_end=False):
        self.node_id = node_id
        self.action_name = action_name
        self.parameters = parameters or {}
        self.state = NodeState.WAITING
        self.max_retries = max_retries
        self.retry_count = 0
        self.predecessors = set()
        self.successors = set()
        self.lock = threading.Lock()
        self.execution_thread = None
        self.metadata = metadata or {}  # to store any meta data
        self.repeatable = repeatable    # to run in cycle or strongly connected network
        self.dependency_type = dependency_type  # {'AND','OR'} execution based on incoming links
        self.executed_in_cycle = False  # flag to mark if a cycle is executed
        self.execution_count = 0        # count of executed cycles
        self.max_cycles = max_cycles    # limit on cycles per node
        self.is_start = is_start        # required if it has cycle or strongly connected network
        self.is_end = is_end            # optional, to end prematurely

    def add_predecessor(self, node):
        self.predecessors.add(node)

    def add_successor(self, node):
        self.successors.add(node)

    def can_execute(self):
        if self.is_start:
            return True
        if not self.predecessors:
            return True
        if self.dependency_type == 'AND':
            return all((pred.state == NodeState.COMPLETED or 
                        (pred.repeatable and pred.executed_in_cycle))
                    for pred in self.predecessors)
        else:  # OR
            return any((pred.state == NodeState.COMPLETED or 
                        (pred.repeatable and pred.executed_in_cycle))
                    for pred in self.predecessors)

    def execute(self, action_handlers, event_queue):
        with self.lock:
            if self.state == NodeState.PAUSED or self.state == NodeState.TERMINATED:
                return
            if self.state == NodeState.FAILED and self.retry_count >= self.max_retries:
                self.state = NodeState.TERMINATED
                return
            if self.state in [NodeState.COMPLETED, NodeState.EXECUTING]:
                return
            if not self.can_execute():
                self.state = NodeState.WAITING
                return
            self.state = NodeState.EXECUTING
            self.execution_thread = threading.Thread(target=self.run_action, args=(action_handlers, event_queue))
            self.execution_thread.start()

    def run_action(self, action_handlers, event_queue):
        action_function = action_handlers.get(self.action_name)
        if not action_function:
            self.state = NodeState.FAILED
            event_queue.put(('node_failed', self))
            return
        try:
            action_function(**self.parameters)
            self.execution_count += 1      # Increment execution count
            self.executed_in_cycle = True  # Mark that this node executed in the current cycle
            # Set state to COMPLETED for dependency satisfaction.
            self.state = NodeState.COMPLETED  
            if self.repeatable:
                # For repeatable nodes, signal that they’ve completed for this cycle.
                event_queue.put(('node_completed_and_waiting', self))
            else:
                event_queue.put(('node_completed', self))
        except Exception as e:
            self.state = NodeState.FAILED
            self.retry_count += 1
            if self.retry_count < self.max_retries:
                self.state = NodeState.RETRYING
                event_queue.put(('node_retrying', self))
            else:
                self.state = NodeState.TERMINATED
                event_queue.put(('node_terminated', self))

    def pause(self):
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

class ExecutionManager:
    def __init__(self):
        self.nodes = {}
        self.event_queue = queue.Queue()
        self.action_handlers = {}
        self.terminate_flag = threading.Event()
        
    def add_node(self, node):
        self.nodes[node.node_id] = node

    def add_action_handler(self, action_name, action_function):
        self.action_handlers[action_name] = action_function

    def run(self):
        # Initialize nodes that can start executing
        for node in self.nodes.values():
            if node.can_execute():
                node.state = NodeState.READY
                node.execute(self.action_handlers, self.event_queue)

        while not self.terminate_flag.is_set():
            try:
                event, node = self.event_queue.get(timeout=0.1)
                if event == 'node_completed':
                    print(f"Node {node.node_id} completed.")
                    # Trigger successors
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
                    # Trigger successors
                    for successor in node.successors:
                        if successor.state == NodeState.WAITING and successor.can_execute():
                            successor.state = NodeState.READY
                            successor.execute(self.action_handlers, self.event_queue)
                        else:
                            node.execute(self.action_handlers, self.event_queue)
                else:
                    pass  # Handle other events as needed
                self.event_queue.task_done() 
            except queue.Empty:
                # Check normal termination
                if all(n.state in [NodeState.COMPLETED, NodeState.TERMINATED] 
                                     for n in self.nodes.values()):
                    self.terminate_flag.set()
            
            all_terminated = all(n.state in [NodeState.COMPLETED, NodeState.TERMINATED, NodeState.FAILED] 
                                 for n in self.nodes.values())
            threads_alive = any(n.execution_thread and n.execution_thread.is_alive() 
                                for n in self.nodes.values())
            if all_terminated and not threads_alive:
                print("Execution completed.")
                break
        # Cleanup
        self.force_terminate()
        print("Done.")

    def force_terminate(self):
        # Force termination of all threads when end condition met
        for node in self.nodes.values():
            if node.execution_thread and node.execution_thread.is_alive():
                node.terminate()
                node.execution_thread.join()

