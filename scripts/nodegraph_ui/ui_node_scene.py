# scripts/nodegraph_ui/ui_node_scene.py

from PyQt5.QtWidgets import QGraphicsScene, QGraphicsProxyWidget
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QColor

from .ui_connection_item import ConnectionItem
from .ui_socket_item import SocketItem

class NodeScene(QGraphicsScene):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph
        # debug flag to render node coordinates
        self.show_coords = False
        # start a timer to update connection paths at a modest rate
        # Lowered from 16ms (60FPS) to reduce CPU when idle or with many items.
        self._conn_timer = QTimer(self)
        self._conn_timer.timeout.connect(self.update_connections)
        self._conn_timer.start(60)  # ~16 FPS — sufficient for smooth interaction
        # visual connection currently highlighted as 'would be overwritten'
        self._highlighted_connection = None
    def create_node_from_key(self, cls_name: str, scene_pos):
        """Create a node by class name and place it at scene_pos.

        This function is robust: it tries a mapping, getattr on the nodes
        module, and a normalized-name fallback (remove spaces, append 'Node').
        """
        try:
            # attempt package-relative import first, then common fallbacks
            nodes_mod = None
            try:
                from . import nodes as nodes_mod
            except Exception:
                try:
                    from . import nodes as nodes_mod
                except Exception:
                    try:
                        import nodes as nodes_mod
                    except Exception:
                        nodes_mod = None

            if nodes_mod is None:
                print(f"[ui_node_scene] create_node_from_key: failed to import nodes module for lookup")
                return None

            print(f"[ui_node_scene] create_node_from_key: nodes module = {getattr(nodes_mod, '__name__', repr(nodes_mod))}")

            # direct mapping of known classes
            mapping = {}
            for name in dir(nodes_mod):
                obj = getattr(nodes_mod, name)
                try:
                    if isinstance(obj, type):
                        mapping[name] = obj
                except Exception:
                    pass
            print(f"[ui_node_scene] create_node_from_key: mapping keys sample={list(mapping.keys())[:20]}")

            # try direct lookup
            cls = mapping.get(cls_name)
            print(f"[ui_node_scene] lookup attempt: cls_name='{cls_name}' -> {cls}")
            if cls is None:
                # try getattr on module (handles exact class names)
                cls = getattr(nodes_mod, cls_name, None)

            if cls is None:
                # try stripping module path (foo.Bar -> Bar)
                if '.' in cls_name:
                    short = cls_name.split('.')[-1]
                    cls = mapping.get(short) or getattr(nodes_mod, short, None)

            if cls is None:
                # normalize: remove non-alphanum and spaces, append Node if missing
                norm = ''.join(ch for ch in cls_name if ch.isalnum())
                if not norm.endswith('Node'):
                    norm2 = norm + 'Node'
                else:
                    norm2 = norm
                cls = mapping.get(norm2) or getattr(nodes_mod, norm2, None)

            if cls is None:
                # try case-insensitive match against available mapping
                lname = cls_name.lower()
                for k, v in mapping.items():
                    try:
                        if k.lower() == lname or k.lower() == (lname + 'node'):
                            cls = v
                            break
                    except Exception:
                        pass

            if cls is None:
                return None

            node = cls()
            self.graph.add_node(node)
            try:
                from .ui_node_item import NodeItemInput, NodeItemProcessor, NodeItemOutput
                from .classes import InputNode, ProcessorNode, OutputNode
                if isinstance(node, InputNode):
                    item = NodeItemInput(node)
                elif isinstance(node, ProcessorNode):
                    item = NodeItemProcessor(node)
                elif isinstance(node, OutputNode):
                    item = NodeItemOutput(node)
                else:
                    raise TypeError(f"Unknown node type: {type(node)}")
                item.setPos(scene_pos)
                self.addItem(item)
                return item
            except Exception as e:
                print(f"[ui_node_scene] create_node_from_key: failed to create NodeItem for {cls_name}: {e}")
                return None
        except Exception:
            return None

    def start_connection(self, socket_item):
        self.temp_connection = ConnectionItem(socket_item)
        self.addItem(self.temp_connection)

    def start_reroute_from_input(self, input_socket_item: SocketItem):
        # Called when user right-drags from an input that currently has a connection.
        backend_input = input_socket_item.socket
        conn = getattr(backend_input, 'connection', None)
        if conn is None:
            return
        out_backend = conn.output_socket

        # Remove backend connection
        try:
            self.graph.disconnect_input(backend_input)
        except Exception:
            pass

        # Remove existing visual ConnectionItem linking these sockets (if any)
        start_ui = None
        for it in self.items():
            if isinstance(it, SocketItem) and it.socket is out_backend:
                start_ui = it
                break

        if start_ui is None:
            return

        # remove existing visual connection that matched these endpoints
        for it in list(self.items()):
            if isinstance(it, ConnectionItem):
                try:
                    if getattr(it.start_socket, 'socket', None) is out_backend and getattr(it.end_socket, 'socket', None) is backend_input:
                        self.removeItem(it)
                except Exception:
                    pass

        # Start a temporary connection from the original output socket UI
        self.start_connection(start_ui)

    def mouseMoveEvent(self, event):
        # Update temporary connection end if any
        if hasattr(self, "temp_connection"):
            self.temp_connection.update_end(event.scenePos())
            # Check if hovering over an occupied input socket
            items = self.items(event.scenePos())
            hovered_input_ui = None
            for item in items:
                if isinstance(item, SocketItem) and not item.is_output:
                    if getattr(item.socket, 'connection', None) is not None:
                        hovered_input_ui = item
                        break

            if hovered_input_ui is not None:
                # Change cursor and highlight only if the new connection would be valid
                try:
                    can_override = self.graph.can_connect(self.temp_connection.start_socket.socket, hovered_input_ui.socket)
                except Exception:
                    can_override = False

                if can_override:
                    try:
                        self.views()[0].setCursor(Qt.DragCopyCursor)
                    except Exception:
                        try:
                            self.views()[0].setCursor(Qt.CursorShape.DragCopyCursor)
                        except Exception:
                            pass

                    # Find the visual ConnectionItem that ends at this input and highlight it
                    new_highlight = None
                    for it in self.items():
                        if isinstance(it, ConnectionItem):
                            try:
                                if getattr(it.end_socket, 'socket', None) is hovered_input_ui.socket:
                                    new_highlight = it
                                    break
                            except Exception:
                                pass

                    if new_highlight is not self._highlighted_connection:
                        # Un-highlight previous
                        try:
                            if self._highlighted_connection is not None:
                                try:
                                    self._highlighted_connection.highlight_off()
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        # Highlight new
                        try:
                            if new_highlight is not None:
                                try:
                                    new_highlight.highlight_on()
                                except Exception:
                                    pass
                            self._highlighted_connection = new_highlight
                        except Exception:
                            pass
                else:
                    # incompatible: show forbidden cursor and clear any highlight
                    try:
                        self.views()[0].setCursor(Qt.ForbiddenCursor)
                    except Exception:
                        try:
                            self.views()[0].setCursor(Qt.CursorShape.ForbiddenCursor)
                        except Exception:
                            pass
                    try:
                        if self._highlighted_connection is not None:
                            try:
                                self._highlighted_connection.highlight_off()
                            except Exception:
                                pass
                            self._highlighted_connection = None
                    except Exception:
                        pass
            else:
                try:
                    self.views()[0].setCursor(Qt.ArrowCursor)
                except Exception:
                    try:
                        self.views()[0].setCursor(Qt.CursorShape.ArrowCursor)
                    except Exception:
                        pass
                # Clear any highlighted visual connection
                try:
                    if self._highlighted_connection is not None:
                        try:
                            self._highlighted_connection.highlight_off()
                        except Exception:
                            pass
                        self._highlighted_connection = None
                except Exception:
                    pass

        # Centralized hover handling for sockets: detect socket under cursor
        items = self.items(event.scenePos())
        hovered = None
        for item in items:
            if isinstance(item, SocketItem):
                hovered = item
                break

        prev = getattr(self, '_hovered_socket', None)
        if hovered is not prev:
            if prev is not None:
                try:
                    prev.on_hover_leave()
                except Exception:
                    pass
            if hovered is not None:
                try:
                    hovered.on_hover_enter(event.screenPos())
                except Exception:
                    pass
            self._hovered_socket = hovered
        else:
            # same socket, update tooltip position
            if hovered is not None:
                try:
                    hovered.on_hover_move(event.screenPos())
                except Exception:
                    pass
        # Let the base class handle propagation to items (important for ItemIsMovable)
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        items = self.items(event.scenePos())
        # forward to base to allow item-specific handling
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        # Detect double-clicks on proxy widgets (viewer preview) and
        # forward to any callback they expose. This handles cases where
        # widget-level double-clicks are not delivered on some platforms.
        items = self.items(event.scenePos())
        for it in items:
            try:
                # If the item is a proxy widget created for viewer preview,
                # it exposes `_on_double_click` or `_on_pixmap_double_click`.
                if hasattr(it, '_on_double_click') and callable(getattr(it, '_on_double_click')):
                    try:
                        it._on_double_click(event)
                    except Exception:
                        pass
                    try:
                        event.accept()
                    except Exception:
                        pass
                    return
                # Fallback: some proxies may store the callback under a different name
                if hasattr(it, '_on_pixmap_double_click') and callable(getattr(it, '_on_pixmap_double_click')):
                    try:
                        it._on_pixmap_double_click(event)
                    except Exception:
                        pass
                    try:
                        event.accept()
                    except Exception:
                        pass
                    return
            except Exception:
                pass

        super().mouseDoubleClickEvent(event)

    def mouseReleaseEvent(self, event):
        # forward to base
        if not hasattr(self, "temp_connection"):
            super().mouseReleaseEvent(event)
            return

        items = self.items(event.scenePos())

        target_socket = None
        for item in items:
            if isinstance(item, SocketItem) and not item.is_output:
                target_socket = item
                break

        if target_socket:
            out_socket = self.temp_connection.start_socket.socket
            in_socket = target_socket.socket

            # If input socket is occupied, only allow overwrite if the new connection is compatible
            if getattr(in_socket, 'connection', None) is not None:
                try:
                    can_override = self.graph.can_connect(out_socket, in_socket)
                except Exception:
                    can_override = False

                if can_override:
                    # Remove backend connection
                    try:
                        self.graph.disconnect_input(in_socket)
                    except Exception:
                        pass

                    # Also remove any existing visual ConnectionItem that targeted this input
                    for it in list(self.items()):
                        if isinstance(it, ConnectionItem):
                            try:
                                if getattr(it.end_socket, 'socket', None) is in_socket:
                                    try:
                                        self.removeItem(it)
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                else:
                    # incompatible: abort connection attempt and clean up
                    try:
                        if self._highlighted_connection is not None:
                            try:
                                self._highlighted_connection.highlight_off()
                            except Exception:
                                pass
                            self._highlighted_connection = None
                    except Exception:
                        pass
                    try:
                        self.removeItem(self.temp_connection)
                    except Exception:
                        pass
                    try:
                        del self.temp_connection
                    except Exception:
                        pass
                    # Reset cursor
                    try:
                        self.views()[0].setCursor(Qt.ArrowCursor)
                    except Exception:
                        try:
                            self.views()[0].setCursor(Qt.CursorShape.ArrowCursor)
                        except Exception:
                            pass
                    return

            if self.graph.connect(out_socket, in_socket):
                print(f"[ui_node_scene] Connected {type(out_socket.node).__name__}({out_socket.name}) -> {type(in_socket.node).__name__}({in_socket.name})")
                self.temp_connection.set_end_socket(target_socket)
                # Do not trigger computation on connect. Update viewer UI state
                try:
                    from .ui_node_item import NodeItem
                    from .nodes import ViewerNode
                    for it in list(self.items()):
                        try:
                            if isinstance(it, NodeItem) and getattr(it, 'node', None) is in_socket.node and isinstance(it.node, ViewerNode):
                                try:
                                    # do not call out.get() here; just update status text to indicate a connection
                                    try:
                                        if hasattr(it, '_viewer_status_text'):
                                            it._viewer_status_text.setPlainText('input connected (idle)')
                                    except Exception:
                                        pass
                                except Exception:
                                    pass
                        except Exception:
                            pass
                except Exception:
                    pass

                # Un-highlight any visual connection that was marked for overwrite
                try:
                    if self._highlighted_connection is not None:
                        try:
                            self._highlighted_connection.highlight_off()
                        except Exception:
                            pass
                        self._highlighted_connection = None
                except Exception:
                    pass

                del self.temp_connection
                # Reset cursor
                try:
                    self.views()[0].setCursor(Qt.ArrowCursor)
                except Exception:
                    try:
                        self.views()[0].setCursor(Qt.CursorShape.ArrowCursor)
                    except Exception:
                        pass
                return

        # If release on empty space while Ctrl is held, create a reroute node
        try:
            if not target_socket and (event.modifiers() & Qt.ControlModifier):
                try:
                    from .classes import rerouteNode, SocketType
                    from .ui_node_item import NodeItemProcessor
                except Exception:
                    rerouteNode = None
                    NodeItemProcessor = None

                if rerouteNode is not None and NodeItemProcessor is not None:
                    try:
                        start_ui = self.temp_connection.start_socket
                        out_backend = start_ui.socket
                        # create backend reroute node adopting upstream socket type
                        rr = rerouteNode(getattr(out_backend, 'socket_type', SocketType.UNDEFINED))
                        # add to backend graph
                        try:
                            self.graph.add_node(rr)
                        except Exception:
                            pass

                        # create UI node for reroute and place at release point
                        try:
                            item = NodeItemProcessor(rr)
                            item.setPos(event.scenePos())
                            self.addItem(item)
                        except Exception:
                            item = None

                        # find the socket UI items for reroute node
                        in_ui = None
                        for it in self.items():
                            if isinstance(it, SocketItem) and getattr(it, 'socket', None) is rr.inputs.get('in'):
                                in_ui = it
                                break

                        # create backend connection and visual ConnectionItem
                        if in_ui is not None:
                            try:
                                # connect backend
                                self.graph.connect(out_backend, rr.inputs['in'])
                            except Exception:
                                pass
                            try:
                                conn_item = ConnectionItem(start_ui)
                                conn_item.set_end_socket(in_ui)
                                self.addItem(conn_item)
                            except Exception:
                                pass
                        # clean up temporary connection visual
                        try:
                            if hasattr(self, 'temp_connection'):
                                try:
                                    self.removeItem(self.temp_connection)
                                except Exception:
                                    pass
                                try:
                                    del self.temp_connection
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        # reset cursor
                        try:
                            self.views()[0].setCursor(Qt.ArrowCursor)
                        except Exception:
                            pass
                        return
                    except Exception:
                        pass
        except Exception:
            pass

        # failed
        # Clear highlight if present
        try:
            if self._highlighted_connection is not None:
                try:
                    self._highlighted_connection.highlight_off()
                except Exception:
                    pass
                self._highlighted_connection = None
        except Exception:
            pass

        self.removeItem(self.temp_connection)
        del self.temp_connection
        # Reset cursor
        try:
            self.views()[0].setCursor(Qt.CursorShape.ArrowCursor)
        except Exception:
            try:
                self.views()[0].setCursor(Qt.ArrowCursor)
            except Exception:
                pass

        super().mouseReleaseEvent(event)

    def update_connections(self):
        # iterate over connection items and refresh their path to follow sockets
        for item in self.items():
            if isinstance(item, ConnectionItem):
                try:
                    item.update_path()
                except Exception:
                    pass

    def drawBackground(self, painter, rect):
        """Draw a subtle dotted grid background so nodes don't look like a floating plate.

        The grid is drawn in scene coordinates and will naturally pan/zoom with the view.
        """
        try:
            spacing = 20
            # larger, higher-contrast dot for the snapping grid
            dot_color = QColor(0, 0, 0, 50)
            dot_size = 2
            painter.save()
            painter.setPen(Qt.NoPen)
            left = int(rect.left())
            top = int(rect.top())
            right = int(rect.right())
            bottom = int(rect.bottom())

            start_x = left - (left % spacing)
            start_y = top - (top % spacing)

            # draw slightly larger filled squares at grid intersections
            half = dot_size // 2
            x = start_x
            while x <= right:
                y = start_y
                while y <= bottom:
                    painter.fillRect(x - half, y - half, dot_size, dot_size, dot_color)
                    y += spacing
                x += spacing

            painter.restore()
        except Exception:
            try:
                super().drawBackground(painter, rect)
            except Exception:
                pass

    def drawForeground(self, painter, rect):
        """Optional debug overlay: draw node scene coordinates above each node."""
        try:
            if not getattr(self, 'show_coords', False):
                return
            # robust import for NodeItem (package vs script execution)
            try:
                from .ui_node_item import NodeItem
            except Exception:
                try:
                    from ui_node_item import NodeItem
                except Exception:
                    try:
                        from scripts.nodegraph_ui.ui_node_item import NodeItem
                    except Exception:
                        NodeItem = None
            try:
                print(f"[ui_node_scene] drawForeground: show_coords=True items={len(self.items())}")
            except Exception:
                pass
            from PyQt5.QtGui import QColor, QFont
            painter.save()
            pen = QColor(200, 50, 50)
            painter.setPen(pen)
            font = QFont()
            font.setPointSize(8)
            painter.setFont(font)

            drawn = 0
            for it in self.items():
                try:
                    node_it = None
                    # Prefer items that expose a backend `node` attribute (NodeItem and proxies)
                    try:
                        if getattr(it, 'node', None) is not None:
                            node_it = it
                        elif type(it).__name__ == 'NodeItem':
                            node_it = it
                    except Exception:
                        node_it = None

                    if node_it is None:
                        continue

                    br = node_it.sceneBoundingRect()
                    x = int(br.x())
                    y = int(br.y())
                    text = f"{x},{y}"
                    # draw semi-transparent background and text with shadow for legibility
                    metrics = painter.fontMetrics()
                    tw = metrics.horizontalAdvance(text)
                    th = metrics.height()
                    tx = int(br.x() + 4)
                    ty = int(br.y() - 6)
                    bg_x = tx - 2
                    bg_y = ty - th
                    painter.fillRect(bg_x, bg_y, tw + 6, th + 4, QColor(0, 0, 0, 200))
                    # shadow
                    shadow_pen = QColor(0, 0, 0, 200)
                    painter.setPen(shadow_pen)
                    painter.drawText(tx + 1, ty + 1, text)
                    # main text (contrast color)
                    painter.setPen(QColor(220, 180, 60))
                    painter.drawText(tx, ty, text)
                    drawn += 1
                except Exception:
                    pass
            try:
                print(f"[ui_node_scene] drawForeground: drew {drawn} coords")
            except Exception:
                pass

            painter.restore()
        except Exception:
            try:
                super().drawForeground(painter, rect)
            except Exception:
                pass
