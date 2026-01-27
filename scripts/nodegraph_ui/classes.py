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

class InputSocket:
    def __init__(self, node, tag: str, socket_type: SocketType, is_optional: bool = False):
        self.node = node
        self.name: str = tag
        self.socket_type: SocketType = socket_type
        self.connection: Connection | None = None
        self.is_optional = is_optional
        # UI may place an override value here when the input is editable
        self._override = None
    
    def get(self):
        # If an override has been set via the UI, prefer it when there's no connection
        try:
            if self._override is not None and self.connection is None:
                return self._override
        except Exception:
            pass
        return self.connection.output_socket.get() if self.connection else None


class OutputSocket:
    def __init__(self, node, tag: str, socket_type: SocketType = SocketType.UNDEFINED, is_modifiable: bool = False):
        self.node = node
        self.name: str = tag
        self.socket_type: SocketType = socket_type
        self.is_modifiable = is_modifiable

        self._cache = None
        self._dirty = True
    
    def get(self):
        if self._dirty:
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
        self._dirty = True


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
    def __init__(self):
        self.inputs: dict[str, InputSocket] = {}
        self.outputs: dict[str, OutputSocket] = {}
        self.dependents: list[Node] = []

        self.display_name: str = "Unnamed Node"
        self.description: str = ""
        self.tooltips_in: dict[str, str] = {}
        self.tooltips_out: dict[str, str] = {}

    
    def mark_dirty(self):
        for out in self.outputs.values():
            out.invalidate()

        for dep in self.dependents:
            dep.mark_dirty()

    
    def misses_required_inputs(self) -> bool:
        for inp in self.inputs.values():
            if not inp.is_optional and inp.connection is None:
                return True
        return False
    
    def compute(self):
        raise NotImplementedError("Compute method must be implemented by subclasses")


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
        