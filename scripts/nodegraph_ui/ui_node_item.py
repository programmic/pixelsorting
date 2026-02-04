
__all__ = ["NodeItemInput", "NodeItemProcessor", "NodeItemOutput"]
import io
from PyQt5.QtWidgets import (
    QGraphicsRectItem, QGraphicsTextItem, QGraphicsProxyWidget,
    QPushButton, QSpinBox, QDoubleSpinBox, QLineEdit, QCheckBox,
    QMainWindow, QLabel, QApplication, QGraphicsItem
)
from PyQt5.QtCore import QRectF, Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QImage, QColor
from PIL.ImageQt import ImageQt
from .ui_socket_item import SocketItem, RADIUS
from .nodes import ViewerNode, RenderToFileNode
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
        label = QGraphicsTextItem(display_name, self)
        label.setDefaultTextColor(QColor(180, 220, 180))
        label.setPos(10, 5)
        self._create_outputs()
        self.setRect(0, 0, self.WIDTH, self._height)

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
        label = QGraphicsTextItem(display_name, self)
        label.setDefaultTextColor(QColor(200, 200, 220))
        label.setPos(10, 5)
        self._create_sockets()
        self.setRect(0, 0, self.WIDTH, self._height)

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
        display_name = getattr(self.node, 'display_name', 'Output Node')
        label = QGraphicsTextItem(display_name, self)
        label.setDefaultTextColor(QColor(220, 180, 180))
        label.setPos(10, 5)
        self._create_inputs()
        self.setRect(0, 0, self.WIDTH, self._height)

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
        self._pix_item = QGraphicsRectItem(self)
        self._pix_item.setRect(5, self.MARGIN_TOP + 30, self.WIDTH - 10, 100)
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

    def _compute_upstream(self, out_socket, visited):
        node = out_socket.node
        if node in visited:
            return
        visited.add(node)
        for inp in node.inputs.values():
            if inp.connection:
                self._compute_upstream(inp.connection.output_socket, visited)
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
            self._compute_upstream(inp.connection.output_socket, visited)
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
        qimg = None
        try:
            qimg = ImageQt(img)
        except Exception:
            pass
        if qimg is None:
            buf = io.BytesIO()
            img.save(buf, "PNG")
            qimg = QImage.fromData(buf.getvalue())
        pix = QPixmap.fromImage(qimg)
        pix = pix.scaled(self.WIDTH - 10, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._pix_item.setBrush(pix)

    def _setup_render_button(self):
        run_btn = QPushButton("RUN")
        proxy = QGraphicsProxyWidget(self)
        proxy.setWidget(run_btn)
        y = self._height - 30
        proxy.setPos(self.WIDTH - 70, y)
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
            result = self.func()
            self.finished_sig.emit({"success": True, "result": result, "exc": None})
        except Exception:
            self.finished_sig.emit({"success": False, "result": None, "exc": None})

