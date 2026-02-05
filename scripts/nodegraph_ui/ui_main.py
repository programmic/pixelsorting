# scripts/nodegraph_ui/ui_main.py

import sys
from PyQt5.QtWidgets import QGraphicsView
from PyQt5.QtWidgets import QApplication, QGraphicsView, QMainWindow
from PyQt5.QtCore import Qt, QEvent

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QListWidget, QListWidgetItem, QToolBar
from PyQt5.QtCore import Qt, QEvent

from .ui_node_scene import NodeScene
from .ui_node_item import NodeItemInput, NodeItemProcessor, NodeItemOutput

from .classes import Graph, SocketType, Node, InputSocket, OutputSocket
from .nodes import SourceImageNode
from .nodes import *
import json
import os
from PyQt5.QtWidgets import QFileDialog

# helper to convert non-JSON-serializable objects (like PIL Images) into
# lightweight JSON-friendly representations when saving the graph.
def _safe_serialize(obj):
    try:
        # simple primitives
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        # lists/tuples
        if isinstance(obj, (list, tuple)):
            return [_safe_serialize(x) for x in obj]
        # dicts
        if isinstance(obj, dict):
            return {str(k): _safe_serialize(v) for k, v in obj.items()}
        # PIL Image -> do not serialize, return None
        try:
            from PIL.Image import Image as _PILImage
            if isinstance(obj, _PILImage):
                return None
        except Exception:
            pass
        # fallback to repr string
        try:
            return repr(obj)
        except Exception:
            return str(type(obj))
    except Exception:
        try:
            return str(obj)
        except Exception:
            return None

class IOTestNode(Node):
    # node with every I/O type for testing
    def __init__(self):
        super().__init__()

        for s in SocketType:
            self.inputs[s.name.lower()] = InputSocket(
                self, s.name.lower(), s
            )
            self.outputs[s.name.lower()] = OutputSocket(
                self, s.name.lower(), s
            )
        
        self.inputs["size"] = InputSocket(
            self, "size", SocketType.INT
        )
        self.inputs["size"].is_optional = True

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Node Graph Test")
        self.resize(900, 600)

        # backend graph
        self.graph = Graph()

        # scene + view
        self.scene = NodeScene(self.graph)

        # Custom view that enables middle-button panning by handling
        # middle-button press/move/release and adjusting scrollbars.
        class PanGraphicsView(QGraphicsView):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._panning = False
                self._pan_start = None
                # accept drops from palette
                self.setAcceptDrops(True)
                # ensure viewport also accepts drops
                try:
                    self.viewport().setAcceptDrops(True)
                except Exception:
                    pass

            def keyPressEvent(self, event):
                # Home: center/fit view to all node item bounding boxes
                try:
                    if event.key() == Qt.Key_Home:
                        self.center_on_nodes()
                        event.accept()
                        return
                    # Coordinate toggle via toolbar only; keypress disabled
                    # Delete: remove selected nodes from scene and backend graph
                    if event.key() == Qt.Key_Delete:
                        try:
                            s = self.scene()
                            if s is not None:
                                sel = list(s.selectedItems())
                                for it in sel:
                                    try:
                                        # remove backend node references
                                        node = getattr(it, 'node', None)
                                        if node is not None and node in s.graph.nodes:
                                            s.graph.nodes.remove(node)
                                        # remove connections that reference this node (use Graph.disconnect_input)
                                        for c in list(s.graph.connections):
                                            try:
                                                out_sock = getattr(c, 'output_socket', None)
                                                in_sock = getattr(c, 'input_socket', None)
                                                if out_sock is None or in_sock is None:
                                                    continue
                                                if out_sock.node is node or in_sock.node is node:
                                                    try:
                                                        # remove visual ConnectionItem(s) that match these endpoints
                                                        for it_conn in list(s.items()):
                                                            try:
                                                                from .ui_connection_item import ConnectionItem
                                                                if isinstance(it_conn, ConnectionItem):
                                                                    try:
                                                                        if getattr(it_conn.start_socket, 'socket', None) is out_sock and getattr(it_conn.end_socket, 'socket', None) is in_sock:
                                                                            try:
                                                                                s.removeItem(it_conn)
                                                                            except Exception:
                                                                                pass
                                                                    except Exception:
                                                                        pass
                                                            except Exception:
                                                                pass
                                                        # remove backend connection via Graph helper
                                                        try:
                                                            s.graph.disconnect_input(in_sock)
                                                        except Exception:
                                                            # fallback: try to remove from list directly
                                                            try:
                                                                if c in s.graph.connections:
                                                                    s.graph.connections.remove(c)
                                                            except Exception:
                                                                pass
                                                    except Exception:
                                                        pass
                                            except Exception:
                                                pass
                                    except Exception:
                                        pass
                                    try:
                                        s.removeItem(it)
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                        event.accept()
                        return
                except Exception:
                    pass
                super().keyPressEvent(event)

            def center_on_nodes(self):
                s = self.scene()
                if s is None:
                    return
                try:
                    rect = None
                    for item in s.items():
                        if isinstance(item, (NodeItemInput, NodeItemProcessor, NodeItemOutput)):
                            br = item.sceneBoundingRect()
                            rect = br if rect is None else rect.united(br)
                    if rect is None or rect.isEmpty():
                        self.centerOn(0, 0)
                        return
                    # Center on the center of the bounding rect
                    center = rect.center()
                    self.centerOn(center)
                    # Optionally fit in view with margin
                    margin = 40
                    rect = rect.adjusted(-margin, -margin, margin, margin)
                    self.fitInView(rect, Qt.KeepAspectRatio)
                except Exception:
                    try:
                        self.centerOn(0, 0)
                    except Exception:
                        pass

            def mousePressEvent(self, event):
                if event.button() == Qt.MiddleButton:
                    self._panning = True
                    self._pan_start = event.pos()
                    self.setCursor(Qt.ClosedHandCursor)
                    # Expand the scene rect to give an "infinite" canvas area
                    try:
                        s = self.scene()
                        if s is not None:
                            s.setSceneRect(-20000, -20000, 40000, 40000)
                    except Exception:
                        pass
                    # accept the event so we get subsequent moves
                    event.accept()
                    return
                super().mousePressEvent(event)

            def mouseMoveEvent(self, event):
                if self._panning and self._pan_start is not None:
                    delta = event.pos() - self._pan_start
                    # Prefer scrollbar movement (reliable); keep translate as
                    # a fallback for different viewport transforms.
                    try:
                        hbar = self.horizontalScrollBar()
                        vbar = self.verticalScrollBar()
                        hbar.setValue(hbar.value() - int(delta.x()))
                        vbar.setValue(vbar.value() - int(delta.y()))
                    except Exception:
                        try:
                            self.translate(-delta.x(), -delta.y())
                        except Exception:
                            pass
                    self._pan_start = event.pos()
                    event.accept()
                    return
                super().mouseMoveEvent(event)

            def mouseReleaseEvent(self, event):
                if event.button() == Qt.MiddleButton and self._panning:
                    self._panning = False
                    self._pan_start = None
                    self.setCursor(Qt.ArrowCursor)
                    event.accept()
                    return
                super().mouseReleaseEvent(event)

            def dragEnterEvent(self, event):
                try:
                    if event.mimeData().hasText():
                        event.acceptProposedAction()
                        return
                except Exception:
                    pass
                event.ignore()

            def dragMoveEvent(self, event):
                try:
                    if event.mimeData().hasText():
                        event.acceptProposedAction()
                        return
                except Exception:
                    pass
                event.ignore()

            def dropEvent(self, event):
                try:
                    txt = event.mimeData().text()
                    if not txt:
                        event.ignore()
                        return
                    if txt.startswith('node:'):
                        cls_name = txt.split(':', 1)[1]
                    else:
                        cls_name = txt

                    s = self.scene()
                    if s is None:
                        event.ignore()
                        return
                    scene_pos = self.mapToScene(event.pos())
                    created = None
                    try:
                        print(f"[ui_main] dropEvent: attempting create_node_from_key('{cls_name}')")
                        created = s.create_node_from_key(cls_name, scene_pos)
                    except Exception:
                        created = None

                    from PyQt5.QtCore import QTimer
                    if created is None:
                        # show temporary feedback text at drop position
                        try:
                            hint = s.addText(f'Unknown: {cls_name}')
                            hint.setDefaultTextColor(Qt.red)
                            hint.setPos(scene_pos)
                            QTimer.singleShot(1500, lambda: s.removeItem(hint))
                        except Exception:
                            pass
                        event.ignore()
                        return

                    # briefly flash a confirmation
                    try:
                        hint = s.addText(f'Added: {cls_name}')
                        hint.setDefaultTextColor(Qt.white)
                        hint.setPos(scene_pos)
                        QTimer.singleShot(800, lambda: s.removeItem(hint))
                    except Exception:
                        pass

                    event.acceptProposedAction()
                    return
                except Exception:
                    event.ignore()
                    return

        self.view = PanGraphicsView(self.scene)
        # Hide scrollbars to provide a clean, infinite-canvas appearance
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setStyleSheet("background-color: rgb(60,60,60);")
        # enable mouse tracking so hover events on QGraphicsItems are delivered
        self.view.setMouseTracking(True)
        self.view.viewport().setMouseTracking(True)
        # ensure the viewport delivers hover events to items
        self.view.viewport().setAttribute(Qt.WA_Hover, True)
        self.view.setAttribute(Qt.WA_Hover, True)
        # ensure the view processes item interaction
        self.view.setInteractive(True)
        # ensure view does not intercept drag gestures
        self.view.setDragMode(QGraphicsView.NoDrag)
        # create a horizontal layout: palette on left, view on right
        container = QWidget()
        h = QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        # palette
        # create the draggable palette list
        class PaletteList(QListWidget):
            def startDrag(self, supportedActions):
                item = self.currentItem()
                if item is None:
                    return
                key = item.data(Qt.UserRole)
                from PyQt5.QtCore import QMimeData
                from PyQt5.QtGui import QDrag
                md = QMimeData()
                md.setText(f'node:{key}')
                drag = QDrag(self)
                drag.setMimeData(md)
                # always use copy action for palette drags
                drag.exec_(Qt.CopyAction)

        pal = PaletteList()
        pal.setFixedWidth(160)
        pal.setDragEnabled(True)
        pal.setStyleSheet('''
            QListWidget { background-color: rgb(40,40,40); color: rgb(220,220,220); }
            QListWidget::item:selected { background-color: rgb(80,80,80); }
        ''')

        # populate palette dynamically from available Node classes in nodes.py
        try:
            from . import nodes as nodes_mod
            # collect nodes by category
            categories = {}
            for name in dir(nodes_mod):
                if not name.endswith('Node'):
                    continue
                obj = getattr(nodes_mod, name)
                try:
                    if isinstance(obj, type):
                        # instantiate safely to read metadata
                        try:
                            inst = obj()
                        except Exception:
                            inst = None

                        # display name: prefer `display_name`, fallback to registry name
                        display_text = None
                        if inst is not None:
                            display_text = getattr(inst, 'display_name', None)
                        display_text = display_text or name

                        # category: use `category` if present, otherwise place in 'Uncategorized'
                        category = None
                        if inst is not None:
                            category = getattr(inst, 'category', None)
                        if not category:
                            category = 'Uncategorized'

                        # tooltip: use `description` if present, otherwise no tooltip
                        tooltip = None
                        if inst is not None:
                            tooltip = getattr(inst, 'description', None)

                        categories.setdefault(category, []).append((display_text, name, tooltip))
                except Exception:
                    pass

            # sort categories alphabetically, but keep 'Uncategorized' last
            cat_keys = sorted([k for k in categories.keys() if k != 'Uncategorized'])
            if 'Uncategorized' in categories:
                cat_keys.append('Uncategorized')

            for cat in cat_keys:
                # add a non-selectable header item for the category
                hdr = QListWidgetItem(cat)
                hdr.setFlags(Qt.NoItemFlags)
                hdr.setBackground(Qt.darkGray)
                pal.addItem(hdr)

                # add nodes in this category
                for display_text, key, tooltip in sorted(categories[cat], key=lambda x: x[0]):
                    it = QListWidgetItem('  ' + display_text)
                    it.setData(Qt.UserRole, key)
                    if tooltip:
                        it.setToolTip(tooltip)
                    pal.addItem(it)
        except Exception:
            # fallback to hardcoded names if dynamic import fails
            def add_palette_item(text, key):
                it = QListWidgetItem(text)
                it.setData(Qt.UserRole, key)
                pal.addItem(it)

            add_palette_item('Source Image', 'SourceImageNode')
            add_palette_item('Blur', 'BlurNode')
            add_palette_item('Viewer', 'ViewerNode')
            add_palette_item('Value Int', 'ValueIntNode')
            add_palette_item('Value Float', 'ValueFloatNode')
            add_palette_item('Value String', 'ValueStringNode')
            add_palette_item('Value Bool', 'ValueBoolNode')
            add_palette_item('IO Test', 'IOTestNode')

        # store palette on self for toggle
        self.palette_widget = pal
        h.addWidget(self.palette_widget)
        h.addWidget(self.view)
        container.setLayout(h)
        self.setCentralWidget(container)

        # toolbar: zoom and palette toggle
        tb = QToolBar('View')
        tb.setStyleSheet('background-color: rgb(40,40,40); color: rgb(220,220,220);')
        self.addToolBar(tb)

        zoom_in = tb.addAction('Zoom In')
        zoom_in.triggered.connect(lambda: self.view.scale(1.2, 1.2))
        zoom_out = tb.addAction('Zoom Out')
        zoom_out.triggered.connect(lambda: self.view.scale(1/1.2, 1/1.2))
        reset = tb.addAction('Reset Zoom')
        reset.triggered.connect(lambda: self.view.resetTransform())
        toggle = tb.addAction('Toggle Palette')
        def _toggle_palette():
            try:
                vis = self.palette_widget.isVisible()
                self.palette_widget.setVisible(not vis)
            except Exception:
                pass
        toggle.triggered.connect(_toggle_palette)

        # Save / Load graph actions
        save_action = tb.addAction('Save Graph')
        load_action = tb.addAction('Load Graph')

        def _save_graph():
            try:
                path, _ = QFileDialog.getSaveFileName(self, 'Save Graph', 'saved/graph.json', 'JSON Files (*.json)')
                if not path:
                    return
                # collect node item instances in scene order
                nodes_data = []
                node_items = []
                for it in self.scene.items():
                    try:
                        if isinstance(it, (NodeItemInput, NodeItemProcessor, NodeItemOutput)):
                            node_items.append(it)
                    except Exception:
                        pass

                # reverse because QGraphicsScene.items returns in stacking order
                node_items = list(reversed(node_items))

                for ni in node_items:
                    try:
                        n = ni.node
                        cls_name = type(n).__name__
                        pos = [ni.pos().x(), ni.pos().y()]
                        # gather simple attributes for modifiable outputs and common fields
                        attrs = {}
                        try:
                            # for value nodes keep `value`
                            if hasattr(n, 'value'):
                                attrs['value'] = getattr(n, 'value')
                        except Exception:
                            pass
                        # also capture output caches, but skip images
                        outputs = {}
                        try:
                            for oname, out in n.outputs.items():
                                try:
                                    # Only save non-image outputs
                                    if hasattr(out, 'socket_type') and str(getattr(out, 'socket_type', '')) in [
                                        'SocketType.PIL_IMG', 'SocketType.PIL_IMG_MONOCH', 'SocketType.COLOR', 'SocketType.LIST_COLORS']:
                                        outputs[oname] = None
                                    else:
                                        val = getattr(out, '_cache', None)
                                        outputs[oname] = _safe_serialize(val)
                                except Exception:
                                    outputs[oname] = None
                        except Exception:
                            pass
                        nodes_data.append({'class': cls_name, 'pos': pos, 'attrs': attrs, 'outputs': outputs})
                    except Exception:
                        pass

                # collect connections referencing node indices and socket names
                connections = []
                try:
                    for c in self.graph.connections:
                        try:
                            out_node = c.output_socket.node
                            in_node = c.input_socket.node
                            # find indices in nodes_data by object identity
                            out_idx = None
                            in_idx = None
                            for idx, ni in enumerate(node_items):
                                try:
                                    if ni.node is out_node:
                                        out_idx = idx
                                    if ni.node is in_node:
                                        in_idx = idx
                                except Exception:
                                    pass
                            if out_idx is None or in_idx is None:
                                continue
                            connections.append({'out_idx': out_idx, 'out_name': c.output_socket.name, 'in_idx': in_idx, 'in_name': c.input_socket.name})
                        except Exception:
                            pass
                except Exception:
                    pass

                doc = {'nodes': nodes_data, 'connections': connections}
                try:
                    with open(path, 'w', encoding='utf-8') as f:
                        json.dump(doc, f, ensure_ascii=False, indent=2)
                except Exception:
                    # try fallback to current working dir
                    with open(os.path.basename(path), 'w', encoding='utf-8') as f:
                        json.dump(doc, f, ensure_ascii=False, indent=2)
            except Exception:
                import traceback
                traceback.print_exc()

        def _load_graph():
            try:
                path, _ = QFileDialog.getOpenFileName(self, 'Load Graph', '', 'JSON Files (*.json)')
                if not path:
                    return
                with open(path, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                nodes_doc = doc.get('nodes', [])
                conns_doc = doc.get('connections', [])

                # clear existing scene and graph
                try:
                    for it in list(self.scene.items()):
                        try:
                            if isinstance(it, (NodeItemInput, NodeItemProcessor, NodeItemOutput)):
                                # remove backend node
                                n = getattr(it, 'node', None)
                                try:
                                    if n in self.graph.nodes:
                                        self.graph.nodes.remove(n)
                                except Exception:
                                    pass
                                try:
                                    self.scene.removeItem(it)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    # clear connections list
                    try:
                        self.graph.connections.clear()
                    except Exception:
                        pass
                except Exception:
                    pass

                # create nodes
                created_items = []
                try:
                    from . import nodes as nodes_mod
                except Exception:
                    nodes_mod = None

                for nd in nodes_doc:
                    try:
                        cls_name = nd.get('class')
                        cls = None
                        if nodes_mod is not None:
                            cls = getattr(nodes_mod, cls_name, None)
                        if cls is None:
                            # try global lookup
                            cls = globals().get(cls_name)
                        if cls is None:
                            continue
                        node = cls()
                        # apply attrs
                        attrs = nd.get('attrs', {}) or {}
                        try:
                            for k, v in attrs.items():
                                try:
                                    setattr(node, k, v)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        self.graph.add_node(node)
                        if hasattr(node, 'node_type'):
                            if node.node_type == 'input':
                                item = NodeItemInput(node)
                            elif node.node_type == 'output':
                                item = NodeItemOutput(node)
                            else:
                                item = NodeItemProcessor(node)
                        else:
                            item = NodeItemProcessor(node)
                        pos = nd.get('pos', [0,0])
                        try:
                            item.setPos(pos[0], pos[1])
                        except Exception:
                            pass
                        self.scene.addItem(item)
                        # restore outputs cache if present, but skip images and mark them dirty
                        try:
                            outs = nd.get('outputs', {}) or {}
                            for oname, oval in outs.items():
                                try:
                                    if oname in node.outputs:
                                        out = node.outputs[oname]
                                        if hasattr(out, 'socket_type') and str(getattr(out, 'socket_type', '')) in [
                                            'SocketType.PIL_IMG', 'SocketType.PIL_IMG_MONOCH', 'SocketType.COLOR', 'SocketType.LIST_COLORS']:
                                            out._cache = None
                                            out._dirty = True
                                        else:
                                            out._cache = oval
                                            out._dirty = False
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        created_items.append(item)
                    except Exception:
                        pass

                # recreate connections
                try:
                    for cd in conns_doc:
                        try:
                            out_idx = int(cd.get('out_idx'))
                            in_idx = int(cd.get('in_idx'))
                            out_name = cd.get('out_name')
                            in_name = cd.get('in_name')
                            if out_idx < 0 or out_idx >= len(created_items) or in_idx < 0 or in_idx >= len(created_items):
                                continue
                            out_item = created_items[out_idx]
                            in_item = created_items[in_idx]
                            out_node = out_item.node
                            in_node = in_item.node
                            out_sock = out_node.outputs.get(out_name)
                            in_sock = in_node.inputs.get(in_name)
                            if out_sock is None or in_sock is None:
                                continue
                            try:
                                self.graph.connect(out_sock, in_sock)
                            except Exception:
                                pass
                        except Exception:
                            pass
                except Exception:
                    pass

            except Exception:
                import traceback
                traceback.print_exc()

        save_action.triggered.connect(_save_graph)
        load_action.triggered.connect(_load_graph)

        # debug: show node coordinates overlay
        show_coords_action = tb.addAction('Show Coords')
        show_coords_action.setCheckable(True)
        def _toggle_show_coords(checked):
            try:
                self.scene.show_coords = bool(checked)
                # trigger a redraw
                try:
                    self.scene.update()
                except Exception:
                    pass
                # when showing coords, force full-viewport updates to avoid
                # partial repaint artifacts (prevents "trails" when dragging)
                try:
                    if checked:
                        self.view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
                    else:
                        self.view.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
                except Exception:
                    pass
            except Exception:
                pass
        show_coords_action.toggled.connect(_toggle_show_coords)

        self._add_test_nodes()


    def eventFilter(self, source, event):
        # handle viewport resize to reposition trash
        try:
            if source is getattr(self.view, 'viewport')() and event.type() == QEvent.Resize:
                try:
                    self._reposition_trash()
                except Exception:
                    pass
        except Exception:
            pass
        return super().eventFilter(source, event)

    def _add_test_nodes_alt(self):
        nodes = []

        
        nodes.append(SourceImageNode())
        nodes.append(BlurNode())

        nodes.append(ViewerNode())

        nodes.append(IOTestNode())

        for i in nodes:
            self.graph.add_node(i)
            if hasattr(i, 'node_type'):
                if i.node_type == 'input':
                    node_ui = NodeItemInput(i)
                elif i.node_type == 'output':
                    node_ui = NodeItemOutput(i)
                else:
                    node_ui = NodeItemProcessor(i)
            else:
                node_ui = NodeItemProcessor(i)
            node_ui.setPos(250 * len(self.graph.nodes), 100)
            self.scene.addItem(node_ui)
    
    def _add_test_nodes_blur(self):
        nodes = []

        nodes.append(SourceImageNode())
        nodes.append(BlurNode())

        nodes.append(ValueIntNode())
        nodes.append(ValueStringNode())

        nodes.append(ViewerNode())

        for i in nodes:
            self.graph.add_node(i)
            if hasattr(i, 'node_type'):
                if i.node_type == 'input':
                    node_ui = NodeItemInput(i)
                elif i.node_type == 'output':
                    node_ui = NodeItemOutput(i)
                else:
                    node_ui = NodeItemProcessor(i)
            else:
                node_ui = NodeItemProcessor(i)
            node_ui.setPos(250 * len(self.graph.nodes), 100)
            self.scene.addItem(node_ui)
        
        if hasattr(nodes[2], 'node_type') and nodes[2].node_type == 'input':
            NodeItemInput(nodes[2]).setPos(300, 0)
        elif hasattr(nodes[2], 'node_type') and nodes[2].node_type == 'output':
            NodeItemOutput(nodes[2]).setPos(300, 0)
        else:
            NodeItemProcessor(nodes[2]).setPos(300, 0)
        if hasattr(nodes[3], 'node_type') and nodes[3].node_type == 'input':
            NodeItemInput(nodes[3]).setPos(500, 0)
        elif hasattr(nodes[3], 'node_type') and nodes[3].node_type == 'output':
            NodeItemOutput(nodes[3]).setPos(500, 0)
        else:
            NodeItemProcessor(nodes[3]).setPos(500, 0)
    
    def _add_test_nodes(self):
        nodes = []

        nodes.append(SourceImageNode())
        nodes.append(RescaleNode())
        nodes.append(DitherNode())
        nodes.append(ViewerNode())
        nodes.append(ValueStringNode())
        nodes.append(ValueIntNode())
        nodes.append(ValueStringNode())
        nodes.append(ValueBoolNode())
        nodes.append(ValueStringNode())

        # add nodes individually to the graph (avoid adding the list itself)
        for n in nodes:
            self.graph.add_node(n)
        positions = [(0,0), (200,0), (400,0), (600,0), (200,120), (200, 240), (0,120), (0,240), (200,360)]

        # set initial value on ValueStringNode before creating UI so the
        # editable widget picks up the value during node item construction
        try:
            nodes[4].value = 'atkinson'
            try:
                nodes[4].outputs['value']._cache = 'atkinson'
                nodes[4].outputs['value']._dirty = False
            except Exception: pass
        except Exception: pass

        try:
            nodes[5].value = 6
            try:
                nodes[5].outputs['value']._cache = 6
                nodes[5].outputs['value']._dirty = False
            except Exception: pass
        except Exception: pass

        try:
            nodes[6].value = '800x600'
            try:
                nodes[6].outputs['value']._cache = '800x600'
                nodes[6].outputs['value']._dirty = False
            except Exception: pass
        except Exception: pass

        try:
            nodes[7].value = True
            try:
                nodes[7].outputs['value']._cache = True
                nodes[7].outputs['value']._dirty = False
            except Exception: pass
        except Exception: pass

        try:
            nodes[8].value = 'diverse'
            try:
                nodes[8].outputs['value']._cache = 'diverse'
                nodes[8].outputs['value']._dirty = False
            except Exception: pass
        except Exception: pass

        for i, pos in zip(nodes, positions):
            if hasattr(i, 'node_type'):
                if i.node_type == 'input':
                    node_ui = NodeItemInput(i)
                elif i.node_type == 'output':
                    node_ui = NodeItemOutput(i)
                else:
                    node_ui = NodeItemProcessor(i)
            else:
                node_ui = NodeItemProcessor(i)
            node_ui.setPos(pos[0], pos[1])
            self.scene.addItem(node_ui)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
