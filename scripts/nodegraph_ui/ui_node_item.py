
__all__ = ["NodeItemInput", "NodeItemProcessor", "NodeItemOutput"]
import io
from PyQt5.QtWidgets import (
    QGraphicsRectItem, QGraphicsTextItem, QGraphicsProxyWidget,
    QPushButton, QSpinBox, QDoubleSpinBox, QLineEdit, QCheckBox,
    QMainWindow, QLabel, QApplication, QGraphicsItem, QGraphicsPixmapItem, QComboBox, QGraphicsProxyWidget
)
from PyQt5.QtCore import QRectF, Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QImage, QColor, QBrush
from PIL.ImageQt import ImageQt
from .ui_socket_item import SocketItem, RADIUS
from .nodes import ViewerNode, RenderToFileNode, SourceImageNode, DisplayDataNode
from .classes import SocketType, InputNode, OutputNode, ProcessorNode


# NodeItemInput: for InputNode
class NodeItemInput(QGraphicsRectItem):
    WIDTH = 160
    HEIGHT = 100
    SOCKET_SPACING = 20
    MARGIN_TOP = 25
    MARGIN_BOTTOM = 10

    def __init__(self, backend_node):
        super().__init__()
        self.node = backend_node
        self.input_items = {}
        self.output_items = {}
        self._height = self.HEIGHT
        self.setBrush(QColor(50, 50, 50))
        self.setPen(QColor(120, 120, 120))
        self.setFlag(self.ItemIsMovable)
        self.setFlag(self.ItemIsSelectable)
        display_name = getattr(self.node, 'display_name', 'Input Node')
        self._title_label = QGraphicsTextItem(display_name, self)
        self._title_label.setDefaultTextColor(QColor(180, 220, 180))
        self._title_label.setPos(10, 5)
        y_offset = self.MARGIN_TOP
        from .nodes import ValueIntNode, ValueFloatNode, ValueStringNode, ValueBoolNode
        if isinstance(self.node, (ValueIntNode, ValueFloatNode, ValueStringNode, ValueBoolNode)):
            # For value nodes, create the output socket (usually named 'value')
            y = self.MARGIN_TOP
            max_y = y
            for name, sock in self.node.outputs.items():
                item = SocketItem(sock, self, is_output=True)
                item.setPos(self.WIDTH, y)
                self.output_items[name] = item
                # Optionally, add a label for the socket (comment out if not wanted)
                # label = QGraphicsTextItem(name, self)
                # label.setDefaultTextColor(QColor(200, 200, 200))
                # label.setPos(self.WIDTH - 60, y - 8)
                y += self.SOCKET_SPACING
                max_y = max(max_y, y)
            self._height = max(self.HEIGHT, max_y + self.MARGIN_BOTTOM)
            self.setRect(0, 0, self.WIDTH, self._height)
        else:
            self._create_outputs()
        # Special UI for SourceImageNode (dropdown for images)
        if isinstance(self.node, SourceImageNode):
            combo = QComboBox()
            combo.addItems(getattr(self.node, '_image_files', []))
            combo.setCurrentIndex(getattr(self.node, 'index', 0))
            combo.setMaximumWidth(self.WIDTH - 20)
            def on_index_changed(idx):
                self.node.index = idx
                self.node.compute()
            combo.currentIndexChanged.connect(on_index_changed)
            proxy = QGraphicsProxyWidget(self)
            proxy.setWidget(combo)
            proxy.setPos(10, y_offset)
            y_offset += combo.sizeHint().height() + 5
        # Only show input widgets for Value nodes (not for SourceImageNode or others)
        if isinstance(self.node, (ValueIntNode, ValueFloatNode, ValueStringNode, ValueBoolNode)):
            widget_margin = 10
            widget_width = self.WIDTH - 2 * widget_margin
            val = getattr(self.node, 'value', None)
            widget = None
            if isinstance(val, int):
                widget = QSpinBox()
                widget.setRange(-1000000000, 1000000000)
                widget.setValue(val)
                widget.setMaximumWidth(widget_width)
                widget.valueChanged.connect(lambda v: setattr(self.node, 'value', int(v)))
            elif isinstance(val, float):
                widget = QDoubleSpinBox()
                widget.setRange(-1e12, 1e12)
                widget.setDecimals(4)
                widget.setValue(val)
                widget.setMaximumWidth(widget_width)
                widget.valueChanged.connect(lambda v: setattr(self.node, 'value', float(v)))
            elif isinstance(val, str):
                widget = QLineEdit()
                widget.setText(val)
                widget.setMaximumWidth(widget_width)
                widget.textChanged.connect(lambda v: setattr(self.node, 'value', v))
            elif isinstance(val, bool):
                widget = QCheckBox()
                widget.setChecked(val)
                widget.setMaximumWidth(widget_width)
                widget.toggled.connect(lambda v: setattr(self.node, 'value', bool(v)))
            else:
                print(f"[NodeItemInput] Unsupported value type for inline editor: {type(val)}")
            if widget is not None:
                proxy = QGraphicsProxyWidget(self)
                proxy.setWidget(widget)
                proxy.setPos(widget_margin, y_offset)
                y_offset += widget.sizeHint().height() + 5
        self.setRect(0, 0, self.WIDTH, max(self._height, y_offset))

    def update_dirty_visual(self):
        # Gray if any output is dirty, else normal color
        is_dirty = any(sock._dirty for sock in self.node.outputs.values())
        color = QColor(120, 120, 120) if is_dirty else QColor(180, 220, 180)
        self._title_label.setDefaultTextColor(color)
    def _create_outputs(self):
        y = self.MARGIN_TOP
        max_y = y
        for name, sock in self.node.outputs.items():
            item = SocketItem(sock, self, is_output=True)
            item.setPos(self.WIDTH, y)
            self.output_items[name] = item
            label = QGraphicsTextItem(name, self)
            label.setDefaultTextColor(QColor(200, 200, 200))
            label.setPos(self.WIDTH - 60, y - 8)
            y += self.SOCKET_SPACING
            max_y = max(max_y, y)
        self._height = max(self.HEIGHT, max_y + self.MARGIN_BOTTOM)
        self.setRect(0, 0, self.WIDTH, self._height)

    def boundingRect(self):
        return QRectF(0, 0, self.WIDTH, self._height)

    def paint(self, painter, option, widget=None):
        rect = QRectF(0, 0, self.WIDTH, self._height)
        painter.setBrush(self.brush())
        painter.setPen(self.pen())
        painter.drawRect(rect)

# NodeItemProcessor: for ProcessorNode
class NodeItemProcessor(QGraphicsRectItem):
    WIDTH = 160
    HEIGHT = 100
    SOCKET_SPACING = 20
    MARGIN_TOP = 25
    MARGIN_BOTTOM = 10

    def __init__(self, backend_node):
        super().__init__()
        self.node = backend_node
        self.input_items = {}
        self.output_items = {}
        self._height = self.HEIGHT
        self.setBrush(QColor(50, 50, 50))
        self.setPen(QColor(120, 120, 120))
        self.setFlag(self.ItemIsMovable)
        self.setFlag(self.ItemIsSelectable)
        display_name = getattr(self.node, 'display_name', 'Processor Node')
        self._title_label = QGraphicsTextItem(display_name, self)
        self._title_label.setDefaultTextColor(QColor(200, 200, 220))
        self._title_label.setPos(10, 5)
        self._create_sockets()
        self.setRect(0, 0, self.WIDTH, self._height)

    def update_dirty_visual(self):
        is_dirty = any(sock._dirty for sock in self.node.outputs.values())
        color = QColor(120, 120, 120) if is_dirty else QColor(200, 200, 220)
        self._title_label.setDefaultTextColor(color)
    def _create_sockets(self):
        y = self.MARGIN_TOP
        max_y = y
        for name, sock in self.node.inputs.items():
            item = SocketItem(sock, self, is_output=False)
            item.setPos(0, y)
            self.input_items[name] = item
            label = QGraphicsTextItem(name, self)
            label.setDefaultTextColor(QColor(200, 200, 200))
            label.setPos(RADIUS + 6, y - 8)
            y += self.SOCKET_SPACING
            max_y = max(max_y, y)
        y = self.MARGIN_TOP
        for name, sock in self.node.outputs.items():
            item = SocketItem(sock, self, is_output=True)
            item.setPos(self.WIDTH, y)
            self.output_items[name] = item
            label = QGraphicsTextItem(name, self)
            label.setDefaultTextColor(QColor(200, 200, 200))
            label.setPos(self.WIDTH - 60, y - 8)
            y += self.SOCKET_SPACING
            max_y = max(max_y, y)
        self._height = max(self.HEIGHT, max_y + self.MARGIN_BOTTOM)
        self.setRect(0, 0, self.WIDTH, self._height)

    def boundingRect(self):
        return QRectF(0, 0, self.WIDTH, self._height)

    def paint(self, painter, option, widget=None):
        rect = QRectF(0, 0, self.WIDTH, self._height)
        painter.setBrush(self.brush())
        painter.setPen(self.pen())
        painter.drawRect(rect)

# NodeItemOutput: for OutputNode
class NodeItemOutput(QGraphicsRectItem):
    WIDTH = 160
    HEIGHT = 100
    SOCKET_SPACING = 20
    MARGIN_TOP = 25
    MARGIN_BOTTOM = 10

    def __init__(self, backend_node):
        super().__init__()
        self.node = backend_node
        self.input_items = {}
        self.output_items = {}
        self._height = self.HEIGHT
        self.setBrush(QColor(50, 50, 50))
        self.setPen(QColor(120, 120, 120))
        self.setFlag(self.ItemIsMovable)
        self.setFlag(self.ItemIsSelectable)
        self._active_workers = []  # Fix crash for ViewerNode
        display_name = getattr(self.node, 'display_name', 'Output Node')
        self._title_label = QGraphicsTextItem(display_name, self)
        self._title_label.setDefaultTextColor(QColor(220, 180, 180))
        self._title_label.setPos(10, 5)
        
        # Special UI for ViewerNode
        if isinstance(self.node, ViewerNode):
            self._create_inputs()
            self._setup_viewer()
            self.setRect(0, 0, self.WIDTH, self._height)
        # Special UI for RenderToFileNode (save button)
        elif isinstance(self.node, RenderToFileNode):
            self.MARGIN_BOTTOM += 30
            self._create_inputs()
            self._setup_render_button()
            self.setRect(0, 0, self.WIDTH, self._height)
        elif isinstance(self.node, DisplayDataNode):
            # a node which takes one undefined value to display in UI
            self._create_inputs()
            self.setRect(0, 0, self.WIDTH, self._height)
            # create string to display data
            self._data_display = QGraphicsTextItem("", self)
            self._data_display.setDefaultTextColor(QColor(200, 200, 200))
            self._data_display.setPos(10, self.MARGIN_TOP + 40)
            # update display periodically
            self._timer = QTimer()
            self._timer.timeout.connect(self._refresh_data_display)
            self._timer.start(1000)  # Update every second
            # connect to input socket changes (user or propagated)
            inp = self.node.inputs.get("data")
            if inp:
                def on_socket_change():
                    visited = set()
                    if inp.connection:
                        self._compute_upstream(inp.connection.output_socket, visited, allow_hard=False)
                # Try to connect to both user and propagated changes
                try:
                    inp.connection_changed = on_socket_change
                except Exception:
                    pass
        else:
            self._create_inputs()
            self.setRect(0, 0, self.WIDTH, self._height)

    def update_dirty_visual(self):
        is_dirty = any(sock._dirty for sock in self.node.outputs.values())
        color = QColor(120, 120, 120) if is_dirty else QColor(220, 180, 180)
        self._title_label.setDefaultTextColor(color)
    def _refresh_data_display(self):
        # method to refresh data display for DisplayDataNode
        inp = self.node.inputs.get("data")
        if not inp or not inp.connection:
            self._data_display.setPlainText("no data")
            return
        # Always send a soft upstream compute before displaying
        visited = set()
        self._compute_upstream(inp.connection.output_socket, visited, allow_hard=False)
        out = inp.connection.output_socket
        data = out.get(allow_hard=False)
        if data is not None:
            display_str = str(data)
            if len(display_str) > 200:
                display_str = display_str[:200] + "..."
            self._data_display.setPlainText(display_str)
        else:
            self._data_display.setPlainText("no data")

    def _create_inputs(self):
        y = self.MARGIN_TOP
        max_y = y
        for name, sock in self.node.inputs.items():
            item = SocketItem(sock, self, is_output=False)
            item.setPos(0, y)
            self.input_items[name] = item
            label = QGraphicsTextItem(name, self)
            label.setDefaultTextColor(QColor(200, 200, 200))
            label.setPos(RADIUS + 6, y - 8)
            y += self.SOCKET_SPACING
            max_y = max(max_y, y)
        self._height = max(self.HEIGHT, max_y + self.MARGIN_BOTTOM)
        self.setRect(0, 0, self.WIDTH, self._height)

    def boundingRect(self):
        return QRectF(0, 0, self.WIDTH, self._height)

    def paint(self, painter, option, widget=None):
        rect = QRectF(0, 0, self.WIDTH, self._height)
        painter.setBrush(self.brush())
        painter.setPen(self.pen())
        painter.drawRect(rect)

    def _create_sockets(self):
        y = self.MARGIN_TOP
        max_y = y
        for name, sock in self.node.inputs.items():
            item = SocketItem(sock, self, is_output=False)
            item.setPos(0, y)
            self.input_items[name] = item
            label = QGraphicsTextItem(name, self)
            label.setDefaultTextColor(QColor(200, 200, 200))
            label.setPos(RADIUS + 6, y - 8)
            # Inline editor for modifiable input
            if getattr(sock, 'is_modifiable', False):
                widget = None
                if sock.socket_type == SocketType.INT:
                    widget = QSpinBox()
                    widget.setRange(-1000000000, 1000000000)
                    try:
                        val = getattr(self.node, name)
                        if isinstance(val, int):
                            widget.setValue(val)
                    except Exception:
                        pass
                    widget.valueChanged.connect(lambda v, s=sock, n=name: setattr(self.node, n, int(v)))
                elif sock.socket_type == SocketType.FLOAT:
                    widget = QDoubleSpinBox()
                    widget.setRange(-1e12, 1e12)
                    widget.setDecimals(4)
                    try:
                        val = getattr(self.node, name)
                        if isinstance(val, float):
                            widget.setValue(val)
                    except Exception:
                        pass
                    widget.valueChanged.connect(lambda v, s=sock, n=name: setattr(self.node, n, float(v)))
                elif sock.socket_type == SocketType.STRING:
                    widget = QLineEdit()
                    try:
                        val = getattr(self.node, name)
                        if isinstance(val, str):
                            widget.setText(val)
                    except Exception:
                        pass
                    widget.textChanged.connect(lambda v, s=sock, n=name: setattr(self.node, n, v))
                elif sock.socket_type == SocketType.BOOLEAN:
                    widget = QCheckBox()
                    try:
                        val = getattr(self.node, name)
                        widget.setChecked(bool(val))
                    except Exception:
                        pass
                    widget.toggled.connect(lambda v, s=sock, n=name: setattr(self.node, n, bool(v)))
                if widget is not None:
                    proxy = QGraphicsProxyWidget(self)
                    proxy.setWidget(widget)
                    proxy.setPos(RADIUS + 80, y - 10)
            y += self.SOCKET_SPACING
            max_y = max(max_y, y)
        y = self.MARGIN_TOP
        for name, sock in self.node.outputs.items():
            item = SocketItem(sock, self, is_output=True)
            item.setPos(self.WIDTH, y)
            self.output_items[name] = item
            label = QGraphicsTextItem(name, self)
            label.setDefaultTextColor(QColor(200, 200, 200))
            label.setPos(self.WIDTH - 60, y - 8)
            y += self.SOCKET_SPACING
            max_y = max(max_y, y)
        self._height = max(self.HEIGHT, max_y + self.MARGIN_BOTTOM)
        self.setRect(0, 0, self.WIDTH, self._height)

    def boundingRect(self):
        return QRectF(0, 0, self.WIDTH, self._height)

    def paint(self, painter, option, widget=None):
        rect = QRectF(0, 0, self.WIDTH, self._height)
        painter.setBrush(self.brush())
        painter.setPen(self.pen())
        painter.drawRect(rect)

    # Viewer UI
    def _setup_viewer(self):
        class ClickablePixmapItem(QGraphicsPixmapItem):
            def __init__(self, parent, on_double_click):
                super().__init__(parent)
                self.setAcceptHoverEvents(True)
                self.setAcceptedMouseButtons(Qt.LeftButton)
                self._on_double_click = on_double_click
            def mouseDoubleClickEvent(self, event):
                print("[Viewer] Double-click detected on image viewer (ClickablePixmapItem).")
                if self._on_double_click:
                    self._on_double_click(event)

        self._pix_item = ClickablePixmapItem(self, self._on_pixmap_double_click)
        self._pix_item.setOffset(5, self.MARGIN_TOP + 30)
        self._pix_item.setShapeMode(QGraphicsPixmapItem.BoundingRectShape)
        self._viewer_status = QGraphicsTextItem("", self)
        self._viewer_status.setDefaultTextColor(QColor(200, 200, 200))
        self._viewer_status.setPos(5, self.MARGIN_TOP + 135)
        get_btn = QPushButton("GET")
        fetch_btn = QPushButton("FETCH")
        get_proxy = QGraphicsProxyWidget(self)
        fetch_proxy = QGraphicsProxyWidget(self)
        get_proxy.setWidget(get_btn)
        fetch_proxy.setWidget(fetch_btn)
        y = self.MARGIN_TOP + 160
        get_proxy.setPos(5, y)
        fetch_proxy.setPos(65, y)
        get_btn.clicked.connect(self._on_get)
        fetch_btn.clicked.connect(self._on_fetch)
        self._height = max(self._height, y + 40)
        self.setRect(0, 0, self.WIDTH, self._height)
        self._timer = QTimer()
        self._timer.timeout.connect(self._refresh_viewer)
        self._timer.start(1000)

    def _on_pixmap_double_click(self, event):
        print("[Viewer] Double-click detected on image viewer.")
        # Show the image in a fullscreen window
        from PyQt5.QtWidgets import QLabel, QDialog, QVBoxLayout
        if hasattr(self, '_last_qpixmap') and self._last_qpixmap is not None:
            print("[Viewer] Opening fullscreen dialog.")
            dlg = QDialog()
            dlg.setWindowTitle("Image Fullscreen")
            dlg.setWindowState(Qt.WindowFullScreen)
            layout = QVBoxLayout(dlg)
            label = QLabel()
            label.setPixmap(self._last_qpixmap.scaled(dlg.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            layout.addWidget(label)
            dlg.setLayout(layout)
            dlg.exec_()
        else:
            print("[Viewer] No QPixmap available for fullscreen display.")

    def _compute_upstream(self, out_socket, visited, allow_hard=False):
        node = out_socket.node
        if node in visited:
            return
        visited.add(node)
        for inp in node.inputs.values():
            if inp.connection:
                self._compute_upstream(inp.connection.output_socket, visited, allow_hard)
        if getattr(node, 'is_soft_computation', False) or allow_hard:
            node.compute()
            for o in node.outputs.values():
                o._dirty = False

    def _on_get(self):
        inp = self.node.inputs.get("image")
        if not inp or not inp.connection:
            self._viewer_status.setPlainText("no input")
            return
        self._viewer_status.setPlainText("computing...")
        def work():
            visited = set()
            self._compute_upstream(inp.connection.output_socket, visited, allow_hard=True)
        worker = ComputeWorker(work)
        def done(res):
            self._viewer_status.setPlainText("idle")
            self._refresh_viewer()
        self._active_workers.append(worker)
        worker.finished_sig.connect(lambda r: self._cleanup_worker(worker, r, done))
        worker.start()

    def _on_fetch(self):
        inp = self.node.inputs.get("image")
        if not inp or not inp.connection:
            return
        # Soft upstream compute only
        visited = set()
        self._compute_upstream(inp.connection.output_socket, visited, allow_hard=False)
        out = inp.connection.output_socket
        img = getattr(out, "_cache", None)
        if img is None:
            return
        self._display_image(img)

    def _refresh_viewer(self):
        inp = self.node.inputs.get("image")
        if not inp or not inp.connection:
            return
        out = inp.connection.output_socket
        img = getattr(out, "_cache", None)
        if img:
            self._display_image(img)

    def _display_image(self, img):
        # Always convert to QImage for QPixmap
        qimg = None
        try:
            from PIL.ImageQt import ImageQt
            qimg_obj = ImageQt(img)
            if hasattr(qimg_obj, 'toqimage'):
                qimg = qimg_obj.toqimage()
            elif isinstance(qimg_obj, QImage):
                qimg = qimg_obj
            else:
                # fallback: try to convert via buffer
                buf = io.BytesIO()
                img.save(buf, "PNG")
                qimg = QImage.fromData(buf.getvalue())
        except Exception:
            buf = io.BytesIO()
            img.save(buf, "PNG")
            qimg = QImage.fromData(buf.getvalue())
        if qimg is not None:
            pix = QPixmap.fromImage(qimg)
            self._last_qpixmap = pix
            scaled_pix = pix.scaled(self.WIDTH - 10, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self._pix_item.setPixmap(scaled_pix)

    def _setup_render_button(self):
        run_btn = QPushButton("RUN")
        proxy = QGraphicsProxyWidget(self)
        proxy.setWidget(run_btn)
        y = self._height - 30
        proxy.setPos(5, y)
        # Ensure the button has the desired width so the proxy displays correctly
        run_btn.setFixedWidth(self.WIDTH - 10)
        self._run_status = QGraphicsTextItem("", self)
        self._run_status.setDefaultTextColor(QColor(200, 200, 200))
        self._run_status.setPos(5, y - 18)
        run_btn.clicked.connect(self._on_run)

    def _on_run(self):
        self._run_status.setPlainText("running...")
        def work():
            for inp in self.node.inputs.values():
                if inp.connection:
                    self._compute_upstream(inp.connection.output_socket, set())
            self.node.compute()
        worker = ComputeWorker(work)
        def done(res):
            self._run_status.setPlainText("done")
        self._active_workers.append(worker)
        worker.finished_sig.connect(lambda r: self._cleanup_worker(worker, r, done))
        worker.start()

    def _cleanup_worker(self, worker, res, callback):
        if worker in self._active_workers:
            self._active_workers.remove(worker)
        callback(res)

# Threaded compute worker for UI responsiveness
class ComputeWorker(QThread):
    finished_sig = pyqtSignal(object)
    def __init__(self, func, parent=None):
        super().__init__(parent)
        self.func = func

    def run(self):
        try:
            print("[ComputeWorker] Starting compute work...")
            result = self.func()
            print("[ComputeWorker] Compute work finished.")
            self.finished_sig.emit({"success": True, "result": result, "exc": None})
        except Exception:
            print("[ComputeWorker] Compute work failed.")
            self.finished_sig.emit({"success": False, "result": None, "exc": None})

