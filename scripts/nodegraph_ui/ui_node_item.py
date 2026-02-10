
__all__ = ["NodeItemInput", "NodeItemProcessor", "NodeItemOutput"]
import io
from PyQt5.QtWidgets import (
    QGraphicsRectItem, QGraphicsTextItem, QGraphicsProxyWidget,
    QPushButton, QSpinBox, QDoubleSpinBox, QLineEdit, QCheckBox,
    QMainWindow, QLabel, QApplication, QGraphicsItem, QGraphicsPixmapItem, QComboBox, QGraphicsProxyWidget,
    QWidget, QDialog, QVBoxLayout, QScrollArea
)
from PyQt5.QtCore import QRectF, Qt, QTimer, QThread, pyqtSignal, QSize, QObject, QEvent
from PyQt5.QtGui import QPixmap, QImage, QColor, QBrush
from PIL.ImageQt import ImageQt
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtCore import QPoint
from PyQt5.QtGui import QTransform


# ViewerDialog: standalone zoomable/pannable dark-mode image viewer
class ViewerDialog(QDialog):
    def __init__(self, pixmap, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Viewer")
        self._orig_pix = pixmap
        self._scale = 1.0

        # layout
        layout = QVBoxLayout(self)
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        layout.addWidget(self.scroll)

        # container and label
        self.container = QWidget()
        self.container.setObjectName('zoom_container')
        self.label = QLabel(self.container)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setScaledContents(False)
        self.label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.label.move(0, 0)

        self.label.setPixmap(self._orig_pix)
        self.scroll.setWidget(self.container)
        # ensure the label receives mouse events for panning
        try:
            self.label.setAttribute(Qt.WA_TransparentForMouseEvents, False)
            self.label.setMouseTracking(True)
        except Exception:
            pass

        # dark styling
        try:
            self.setStyleSheet("QDialog{background-color: #222; color: #ddd}")
            self.scroll.viewport().setStyleSheet("background-color: #222;")
            self.label.setStyleSheet("background-color: transparent; color: #ddd")
        except Exception:
            pass

        # install pan+wheel filter on viewport and label (handles MMB panning and wheel-zoom)
        self._pan_filter = self._ViewportPanFilter(self.scroll, self)
        try:
            self.scroll.viewport().installEventFilter(self._pan_filter)
            # also install on the label and container so middle-button presses on the image are captured
            self.label.installEventFilter(self._pan_filter)
            self.container.installEventFilter(self._pan_filter)
        except Exception:
            pass

        # initialize size to 80% of available screen
        try:
            screen = QApplication.primaryScreen()
            geom = screen.availableGeometry() if screen is not None else None
            if geom is not None:
                w = int(geom.width() * 0.8)
                h = int(geom.height() * 0.8)
                self.resize(w, h)
        except Exception:
            pass

        # fit-to-window initial
        QTimer.singleShot(0, self._fit_to_viewport)

    def _fit_to_viewport(self):
        try:
            avail = self.scroll.viewport().size()
            if avail.width() > 0 and avail.height() > 0 and self._orig_pix is not None:
                ow = self._orig_pix.width()
                oh = self._orig_pix.height()
                scale_w = avail.width() / ow
                scale_h = avail.height() / oh
                fit_scale = min(scale_w, scale_h, 1.0)
                self._scale = fit_scale
                self._apply_scale()
        except Exception:
            pass

    def _apply_scale(self, center_ratio=(0.5, 0.5)):
        """Scale label pixmap and update container maintaining scroll position around center_ratio."""
        try:
            if self._orig_pix is None:
                return
            ow = self._orig_pix.width()
            oh = self._orig_pix.height()
            new_w = max(1, int(ow * self._scale))
            new_h = max(1, int(oh * self._scale))
            scaled = self._orig_pix.scaled(new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # compute previous viewport-relative center to restore after resize
            scr = self.scroll
            hbar = scr.horizontalScrollBar()
            vbar = scr.verticalScrollBar()
            vw = scr.viewport().width()
            vh = scr.viewport().height()

            # compute fractional position of center_ratio within content
            if self.container.width() > 0 and self.container.height() > 0:
                frac_x = (hbar.value() + center_ratio[0] * vw) / max(1, self.container.width())
                frac_y = (vbar.value() + center_ratio[1] * vh) / max(1, self.container.height())
            else:
                frac_x = 0.5
                frac_y = 0.5

            self.label.setPixmap(scaled)
            self.label.resize(scaled.size())
            # ensure container at least viewport size
            cw = max(scaled.width(), vw)
            ch = max(scaled.height(), vh)
            self.container.resize(cw, ch)
            lx = (cw - scaled.width()) // 2
            ly = (ch - scaled.height()) // 2
            self.label.move(lx, ly)

            # restore scrollbars to keep same fractional center
            try:
                hbar.setValue(int(frac_x * max(0, cw - vw)))
                vbar.setValue(int(frac_y * max(0, ch - vh)))
            except Exception:
                pass
            try:
                print(f"[ViewerDialog.zoom] scale={self._scale:.3f} scaled={new_w}x{new_h} container={cw}x{ch} viewport={vw}x{vh}")
            except Exception:
                pass
        except Exception:
            pass

    def zoom_at(self, factor, cursor_pos=None):
        # compute center ratio from cursor_pos in viewport coordinates
        try:
            scr = self.scroll
            if cursor_pos is None:
                center = (0.5, 0.5)
            else:
                vp = scr.viewport()
                rel_x = cursor_pos.x()
                rel_y = cursor_pos.y()
                vw = vp.width()
                vh = vp.height()
                center = (rel_x / max(1, vw), rel_y / max(1, vh))
            self._scale = max(0.05, min(10.0, self._scale * factor))
            self._apply_scale(center_ratio=center)
        except Exception:
            pass

    class _ViewportPanFilter(QObject):
        def __init__(self, scroll, parent_dialog):
            super().__init__()
            self.scroll = scroll
            self.parent = parent_dialog
            self.panning = False
            self.last_pos = None

        def eventFilter(self, obj, ev):
            try:
                t = ev.type()
                if t == QEvent.MouseButtonPress:
                    try:
                        print(f"[ViewerDialog.pan] MouseButtonPress on {type(obj).__name__} button={ev.button()} pos={ev.pos()} gpos={ev.globalPos()}")
                    except Exception:
                        pass
                    if ev.button() == Qt.MiddleButton:
                        self.panning = True
                        try:
                            self.last_pos = ev.globalPos()
                        except Exception:
                            self.last_pos = ev.pos()
                        try:
                            self.scroll.viewport().setCursor(Qt.ClosedHandCursor)
                        except Exception:
                            pass
                        return True
                elif t == QEvent.MouseMove:
                    if self.panning and self.last_pos is not None:
                        try:
                            print(f"[ViewerDialog.pan] MouseMove on {type(obj).__name__} pos={ev.pos()} gpos={ev.globalPos()} last={self.last_pos}")
                        except Exception:
                            pass
                        try:
                            cur = ev.globalPos()
                        except Exception:
                            cur = ev.pos()
                        delta = cur - self.last_pos
                        try:
                            hbar = self.scroll.horizontalScrollBar()
                            vbar = self.scroll.verticalScrollBar()
                            old_h = hbar.value()
                            old_v = vbar.value()
                            # If there is no scroll range (image smaller than viewport), move the label inside container instead
                            try:
                                if hbar.maximum() <= 0 and vbar.maximum() <= 0:
                                    lbl = getattr(self.parent, 'label', None)
                                    cont = getattr(self.parent, 'container', None)
                                    if lbl is not None and cont is not None:
                                        cur_x = lbl.x()
                                        cur_y = lbl.y()
                                        new_x = cur_x - int(delta.x())
                                        new_y = cur_y - int(delta.y())
                                        # clamp so image stays within a generous margin inside container
                                        max_x = cont.width() - lbl.width()
                                        max_y = cont.height() - lbl.height()
                                        new_x = max(0, min(new_x, max_x))
                                        new_y = max(0, min(new_y, max_y))
                                        lbl.move(new_x, new_y)
                                else:
                                    hbar.setValue(hbar.value() - int(delta.x()))
                                    vbar.setValue(vbar.value() - int(delta.y()))
                            except Exception:
                                # fallback to scrollbars
                                try:
                                    hbar.setValue(hbar.value() - int(delta.x()))
                                    vbar.setValue(vbar.value() - int(delta.y()))
                                except Exception:
                                    pass
                            try:
                                print(f"[ViewerDialog.pan] scrollbar h: {old_h} -> {hbar.value()}, v: {old_v} -> {vbar.value()}")
                            except Exception:
                                pass
                        except Exception:
                            pass
                        self.last_pos = cur
                        return True
                elif t == QEvent.Wheel:
                    # Intercept wheel to perform zoom centered on viewport cursor
                    try:
                        # prefer pixel-precise delta when available
                        delta = 0
                        try:
                            delta = ev.angleDelta().y()
                        except Exception:
                            try:
                                delta = ev.delta()
                            except Exception:
                                delta = 0
                        steps = delta / 120.0 if delta else 0
                        factor = 1.15 ** steps if steps else 1.0
                        # cursor position relative to viewport
                        try:
                            cursor_pos = ev.pos()
                        except Exception:
                            cursor_pos = None
                        try:
                            if getattr(self, 'parent', None) is not None:
                                self.parent.zoom_at(factor, cursor_pos=cursor_pos)
                        except Exception:
                            pass
                        try:
                            ev.accept()
                        except Exception:
                            pass
                        return True
                    except Exception:
                        pass
                elif t == QEvent.MouseButtonRelease:
                    try:
                        print(f"[ViewerDialog.pan] MouseButtonRelease on {type(obj).__name__} button={ev.button()}")
                    except Exception:
                        pass
                    if ev.button() == Qt.MiddleButton and self.panning:
                        self.panning = False
                        self.last_pos = None
                        try:
                            self.scroll.viewport().setCursor(Qt.ArrowCursor)
                        except Exception:
                            pass
                        return True
            except Exception:
                pass
            return False
from .ui_socket_item import SocketItem, RADIUS
from .nodes import ViewerNode, RenderToFileNode, SourceImageNode, DisplayDataNode
from .classes import SocketType, InputNode, OutputNode, ProcessorNode


# NodeItemInput: for InputNode
class NodeItemInput(QGraphicsRectItem):
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_offset = event.scenePos() - self.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            modifiers = event.modifiers()
            pos = event.scenePos()
            if modifiers & Qt.ControlModifier:
                # Snap to next 20th unit, respecting original offset
                offset = getattr(self, '_drag_offset', None)
                if offset is None:
                    offset = pos - self.pos()
                snapped_x = round((pos.x() - offset.x()) / 20) * 20
                snapped_y = round((pos.y() - offset.y()) / 20) * 20
                self.setPos(snapped_x, snapped_y)
                event.accept()
                return
        super().mouseMoveEvent(event)
    WIDTH = 160
    HEIGHT = 100
    SOCKET_SPACING = 20
    MARGIN_TOP = 25
    MARGIN_BOTTOM = 10

    def __init__(self, backend_node):
        super().__init__()
        self.node = backend_node
        # If this is a reroute node, render a compact socket-only UI.
        # Use multiple detection strategies for robustness: isinstance, class-name,
        # or display_name match.
        is_reroute = False
        try:
            from .classes import rerouteNode
            if isinstance(self.node, rerouteNode):
                is_reroute = True
        except Exception:
            is_reroute = False

        if not is_reroute:
            try:
                cname = type(self.node).__name__
                if 'reroute' in str(cname).lower():
                    is_reroute = True
            except Exception:
                pass
        if not is_reroute:
            try:
                if getattr(self.node, 'display_name', '').lower() == 'reroute':
                    is_reroute = True
            except Exception:
                pass

        if is_reroute:
            # compact appearance
            self.WIDTH = 36
            self.HEIGHT = 18
            self.node = backend_node
            self.input_items = {}
            self.output_items = {}
            self._height = self.HEIGHT
            # transparent background so it looks like a floating socket
            try:
                self.setBrush(QColor(0, 0, 0, 0))
                self.setPen(QColor(0, 0, 0, 0))
            except Exception:
                pass
            self.setFlag(self.ItemIsMovable)
            self.setFlag(self.ItemIsSelectable)
            # create two socket items: input (left) and output (right)
            try:
                in_item = SocketItem(self.node.inputs.get('in'), self, is_output=False)
                in_item.setPos(0, self.HEIGHT/2)
                self.input_items['in'] = in_item
            except Exception:
                pass
            try:
                out_item = SocketItem(self.node.outputs.get('out'), self, is_output=True)
                out_item.setPos(self.WIDTH, self.HEIGHT/2)
                self.output_items['out'] = out_item
            except Exception:
                pass
            self.setRect(0, 0, self.WIDTH, self._height)
            return
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
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_offset = event.scenePos() - self.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            modifiers = event.modifiers()
            pos = event.scenePos()
            if modifiers & Qt.ControlModifier:
                offset = getattr(self, '_drag_offset', None)
                if offset is None:
                    offset = pos - self.pos()
                snapped_x = round((pos.x() - offset.x()) / 20) * 20
                snapped_y = round((pos.y() - offset.y()) / 20) * 20
                self.setPos(snapped_x, snapped_y)
                event.accept()
                return
        super().mouseMoveEvent(event)
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
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_offset = event.scenePos() - self.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            modifiers = event.modifiers()
            pos = event.scenePos()
            if modifiers & Qt.ControlModifier:
                offset = getattr(self, '_drag_offset', None)
                if offset is None:
                    offset = pos - self.pos()
                snapped_x = round((pos.x() - offset.x()) / 20) * 20
                snapped_y = round((pos.y() - offset.y()) / 20) * 20
                self.setPos(snapped_x, snapped_y)
                event.accept()
                return
        super().mouseMoveEvent(event)
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
        # Use a QLabel inside a QGraphicsProxyWidget so QWidget events
        # (including mouseDoubleClickEvent) are reliably delivered.
        class ClickableLabel(QLabel):
            def __init__(self, on_double_click, parent=None):
                super().__init__(parent)
                self._on_double_click = on_double_click
                self.setAttribute(Qt.WA_TransparentForMouseEvents, False)

            def mouseDoubleClickEvent(self, event):
                print("[Viewer] Double-click detected on image viewer (ClickableLabel).")
                if self._on_double_click:
                    # forward the event to the callback
                    try:
                        self._on_double_click(event)
                    except Exception:
                        pass

        self._pix_label = ClickableLabel(self._on_pixmap_double_click)
        self._pix_label.setScaledContents(False)
        self._pix_label.setFixedSize(self.WIDTH - 10, 100)
        # Subclass QGraphicsProxyWidget to forward double-clicks from
        # the graphics scene level (guaranteed to receive them) to the
        # QLabel callback. Some platforms route events differently,
        # so handling at the proxy ensures reliability.
        class ClickableProxy(QGraphicsProxyWidget):
            def __init__(self, parent, on_double_click):
                super().__init__(parent)
                self._on_double_click = on_double_click

            def mouseDoubleClickEvent(self, event):
                try:
                    print("[Viewer] Double-click detected on proxy.")
                    if callable(self._on_double_click):
                        self._on_double_click(event)
                except Exception:
                    pass
                try:
                    event.accept()
                except Exception:
                    pass

        self._pix_proxy = ClickableProxy(self, self._on_pixmap_double_click)
        self._pix_proxy.setWidget(self._pix_label)
        self._pix_proxy.setPos(5, self.MARGIN_TOP + 30)
        # Ensure proxy and label reliably accept mouse events across platforms
        try:
            self._pix_proxy.setAcceptHoverEvents(True)
            self._pix_proxy.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)
        except Exception:
            try:
                self._pix_proxy.setAcceptedMouseButtons(Qt.LeftButton)
            except Exception:
                pass
        # Add an event filter as a robust fallback for double-click detection
        try:
            class _DblClickFilter(QObject):
                def __init__(self, cb):
                    super().__init__()
                    self.cb = cb

                def eventFilter(self, obj, ev):
                    try:
                        if ev.type() == QEvent.MouseButtonDblClick:
                            try:
                                if callable(self.cb):
                                    self.cb(ev)
                            except Exception:
                                pass
                            try:
                                ev.accept()
                            except Exception:
                                pass
                            return True
                    except Exception:
                        pass
                    return False

            _filter = _DblClickFilter(self._on_pixmap_double_click)
            self._pix_label.installEventFilter(_filter)
            self._pix_proxy.installEventFilter(_filter)
        except Exception:
            pass
        try:
            # Ensure the proxy and its widget accept left-button clicks and receive mouse events
            self._pix_proxy.setAcceptedMouseButtons(Qt.LeftButton)
            self._pix_label.setMouseTracking(True)
            self._pix_label.setAttribute(Qt.WA_TransparentForMouseEvents, False)
            # raise z-order so it receives events before overlapping items
            self._pix_proxy.setZValue(1)
        except Exception:
            pass
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
        # Prefer fetching a full-size image from the backend by forcing
        # an upstream hard-compute on the connected output socket.
        full_img = None
        try:
            inp = self.node.inputs.get("image")
            if inp and inp.connection:
                out_sock = inp.connection.output_socket
                try:
                    # allow hard computation to force full render
                    full_img = out_sock.get(allow_hard=True)
                except Exception:
                    full_img = getattr(out_sock, '_cache', None)
        except Exception:
            full_img = None

        if full_img is None:
            # fallback to last QPixmap cached by the preview (may be small)
            if not (hasattr(self, '_last_qpixmap') and self._last_qpixmap is not None):
                print("[Viewer] No QPixmap available for fullscreen display.")
                return
            full_pix = self._last_qpixmap
        else:
            # convert PIL Image (or similar) to QPixmap
            try:
                try:
                    from PIL.ImageQt import ImageQt
                    qimg_obj = ImageQt(full_img)
                    if hasattr(qimg_obj, 'toqimage'):
                        qimg = qimg_obj.toqimage()
                    elif isinstance(qimg_obj, QImage):
                        qimg = qimg_obj
                    else:
                        import io as _io
                        buf = _io.BytesIO()
                        full_img.save(buf, "PNG")
                        qimg = QImage.fromData(buf.getvalue())
                except Exception:
                    import io as _io
                    buf = _io.BytesIO()
                    full_img.save(buf, "PNG")
                    qimg = QImage.fromData(buf.getvalue())
                full_pix = QPixmap.fromImage(qimg)
            except Exception:
                full_pix = getattr(self, '_last_qpixmap', None)

        # Build a standalone viewer dialog (centralized implementation)
        try:
            dlg = ViewerDialog(full_pix)
            dlg.exec_()
        except Exception:
            # fallback: do nothing
            pass

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
        # Cache scaled QPixmap to avoid repeated rescaling of large images
        if not hasattr(self, '_scaled_pixmap_cache'):
            self._scaled_pixmap_cache = {'img_id': None, 'img_size': None, 'pixmap': None}
        cache = self._scaled_pixmap_cache
        img_id = id(img)
        img_size = getattr(img, 'size', None)
        target_size = (self.WIDTH - 10, 100)

        # Only rescale if image object or size changed
        if cache['img_id'] != img_id or cache['img_size'] != img_size:
            qimg = None
            try:
                from PIL.ImageQt import ImageQt
                qimg_obj = ImageQt(img)
                if hasattr(qimg_obj, 'toqimage'):
                    qimg = qimg_obj.toqimage()
                elif isinstance(qimg_obj, QImage):
                    qimg = qimg_obj
                else:
                    buf = io.BytesIO()
                    img.save(buf, "PNG")
                    qimg = QImage.fromData(buf.getvalue())
            except Exception:
                buf = io.BytesIO()
                img.save(buf, "PNG")
                qimg = QImage.fromData(buf.getvalue())
            if qimg is not None:
                pix = QPixmap.fromImage(qimg)
                scaled_pix = pix.scaled(*target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                cache['img_id'] = img_id
                cache['img_size'] = img_size
                cache['pixmap'] = scaled_pix
                self._last_qpixmap = pix
        if cache['pixmap'] is not None:
            try:
                # update proxy-contained QLabel
                if hasattr(self, '_pix_label') and self._pix_label is not None:
                    self._pix_label.setPixmap(cache['pixmap'])
                else:
                    # fallback to older pixmap item if present
                    try:
                        self._pix_item.setPixmap(cache['pixmap'])
                    except Exception:
                        pass
            except Exception:
                pass

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

