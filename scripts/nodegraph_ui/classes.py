# scripts/nodegraph_ui/classes.py

from enum import Enum

class SocketType(Enum):
    UNDEFINED = 0
    FLOAT = 1
    INT = 2
    STRING = 3
    BOOLEAN = 4
    PIL_IMG = 5
    PIL_IMG_MONOCH = 6
    ENUM = 7
    COLOR = 8
    LIST_COLORS = 9
    DICT = 10

class InputSocket:
    def __init__(self, node, tag: str, socket_type: SocketType, is_optional: bool = False):
        self.node = node
        self.name: str = tag
        self.socket_type: SocketType = socket_type
        self.connection: Connection | None = None
        self.is_optional = is_optional
    
    def get(self, allow_hard=True):
        return self.connection.output_socket.get(allow_hard=allow_hard) if self.connection else None


class OutputSocket:
    def __init__(self, node, tag: str, socket_type: SocketType = SocketType.UNDEFINED, is_modifiable: bool = False):
        self.node = node
        self.name: str = tag
        self.socket_type: SocketType = socket_type
        self.is_modifiable = is_modifiable

        self._cache = None
        self._dirty = True
    
    def get(self, allow_hard=True):
        if self._dirty:
            if getattr(self.node, 'is_soft_computation', False) or allow_hard:
                self.node.compute()
                self._dirty = False
        return self._cache
    
    def set(self, value):
        if self.is_modifiable:
            self._cache = value
            self.invalidate()
        else:
            raise ValueError("Attempted to set value of non-modifiable output socket")
    
    def invalidate(self):
        if not self._dirty:
            self._dirty = True
            # Optionally: notify dependents if needed


class Connection:
    def __init__(self, output_socket: OutputSocket, input_socket: InputSocket):
        self.input_socket: InputSocket = input_socket
        self.output_socket: OutputSocket = output_socket
        input_socket.connection = self
    
    def set_input(self, input_socket: InputSocket):
        self.input_socket = input_socket
    
    def set_output(self, output_socket: OutputSocket):
        self.output_socket = output_socket
    
class Node:
    @property
    def is_soft_computation(self):
        """Override in subclasses: True for math/get-data nodes, False for image effects, rescaling, etc."""
        return False
    def __init__(self):
        self.inputs: dict[str, InputSocket] = {}
        self.outputs: dict[str, OutputSocket] = {}
        self.dependents: list[Node] = []

        self.display_name: str = "Unnamed Node"
        self.description: str = ""
        self.tooltips_in: dict[str, str] = {}
        self.tooltips_out: dict[str, str] = {}

    @property
    def node_type(self):
        if isinstance(self, InputNode):
            return "input"
        elif isinstance(self, OutputNode):
            return "output"
        elif isinstance(self, ProcessorNode):
            return "processor"
        return "unknown"

    
    def mark_dirty(self):
        # Only propagate if at least one output was not already dirty
        any_newly_dirty = False
        for out in self.outputs.values():
            if not out._dirty:
                out.invalidate()
                any_newly_dirty = True
        if any_newly_dirty:
            for dep in self.dependents:
                if dep is not self:
                    dep._mark_dirty_from_upstream()

    def _mark_dirty_from_upstream(self):
        # Internal: called by upstream node's mark_dirty
        for out in self.outputs.values():
            out.invalidate()
        # Do not propagate further to avoid infinite loops

    
    def misses_required_inputs(self) -> bool:
        for inp in self.inputs.values():
            if not inp.is_optional and inp.connection is None:
                return True
        return False
    
    def compute(self):
        raise NotImplementedError("Compute method must be implemented by subclasses")

class InputNode(Node):
    """Input nodes have no input sockets, but can easily be extended to have graphical input controls like text input, sliders, toggles, dropdowns, etc."""

    def user_modified(self):
        """
        Call this method whenever the user changes the value of an input node (e.g., via UI).
        It will mark all direct dependents as dirty, forcing them to recompute when needed.
        """
        self.mark_dirty()
    
class ProcessorNode(Node):
    """Processor nodes have both input and output sockets."""
    pass

class OutputNode(Node):
    """Output nodes have no output sockets, but can easily be extended to have graphical output controls like image viewers, toFile renderers, etc."""
    pass

class rerouteNode(Node):
    """A simple node that has one input and one output of the same type, and just passes the value through. Useful for organizing complex graphs."""
    def __init__(self):
        super().__init__()
        self.inputs['in'] = InputSocket(self, 'in', SocketType.UNDEFINED)
        self.outputs['out'] = OutputSocket(self, 'out', SocketType.UNDEFINED)
    
    def compute(self):
        inp = self.inputs['in']
        out = self.outputs['out']
        if inp.connection is not None:
            out.set(inp.get())

class Graph:
    def __init__(self):
        self.nodes: list[Node] = []
        self.connections: list[Connection] = []
    
    def add_node(self, node: Node):
        self.nodes.append(node)
    
    def add_connection(self, connection: Connection):
        self.connections.append(connection)
    
    def connect(self, out: OutputSocket, inp: InputSocket) -> bool:
        if not self.can_connect(out, inp):
            return False
        if self.creates_cycle(out.node, inp.node):
            return False
        
        conn = Connection(out, inp)
        self.add_connection(conn)

        out.node.dependents.append(inp.node)

        inp.node.mark_dirty()
        return True

    def disconnect_input(self, input_socket: InputSocket) -> bool:
        """Disconnect and remove the connection that targets `input_socket`.

        Returns True if a connection was removed.
        """
        for c in list(self.connections):
            if c.input_socket is input_socket:
                try:
                    # remove dependent link
                    out_node = c.output_socket.node
                    if input_socket.node in out_node.dependents:
                        out_node.dependents.remove(input_socket.node)
                except Exception:
                    pass
                try:
                    # clear the input socket's connection
                    input_socket.connection = None
                except Exception:
                    pass
                try:
                    self.connections.remove(c)
                except Exception:
                    pass
                try:
                    input_socket.node.mark_dirty()
                except Exception:
                    pass
                return True
        return False

    def can_connect(self, output_socket: OutputSocket, input_socket: InputSocket) -> bool:
        if input_socket.connection is not None:
            return False
        if output_socket.socket_type == SocketType.UNDEFINED:
            return True
        if input_socket.socket_type == SocketType.UNDEFINED:
            return True
        # Exact match
        if output_socket.socket_type == input_socket.socket_type:
            return True

        # Compatibility rules:
        # - Monochrome image outputs may be accepted by full-image inputs
        # - Integer outputs may be accepted by float inputs
        try:
            ot = output_socket.socket_type
            it = input_socket.socket_type
            if ot == SocketType.PIL_IMG_MONOCH and it == SocketType.PIL_IMG:
                return True
            if ot == SocketType.INT and it == SocketType.FLOAT:
                return True
        except Exception:
            pass

        return False
    
    def creates_cycle(self, src: Node, dst: Node) -> bool:
        if src == dst:
            return True
        
        for inp in src.inputs.values():
            if inp.connection is not None:
                upstream = inp.connection.output_socket.node
                if self.creates_cycle(upstream, dst):
                    return True
        return False
        