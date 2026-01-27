from PyQt5.QtWidgets import (
    QGraphicsTextItem, QGraphicsRectItem, QGraphicsProxyWidget,
    QGraphicsPixmapItem, QSpinBox, QDoubleSpinBox,
    QLineEdit, QMainWindow, QLabel, QGraphicsObject, QGraphicsItem,
    QComboBox, QPushButton, QFileDialog
)
from PyQt5.QtCore import QRectF, Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPen, QBrush, QColor, QPainter, QPixmap, QFontMetrics

from PIL.ImageQt import ImageQt
from PIL.Image import Image as PILImage

from .ui_socket_item import SocketItem, RADIUS
from .classes import SocketType
try:
    from .nodes import SourceImageNode, ViewerNode, RenderToFileNode
except Exception:
    SourceImageNode = None
    ViewerNode = None
    RenderToFileNode = None


# Combo that forces its popup/view width to match the combo width
class FixedWidthComboBox(QComboBox):
    def showPopup(self):
        try:
            vw = self.view()
            # force the popup width to the current widget width
            vw.setFixedWidth(max(40, int(self.width())))
        except Exception:
            pass
        try:
            super().showPopup()
        except Exception:
            # fallback to base behaviour if something goes wrong
            QComboBox.showPopup(self)


class PreviewPixmapItem(QGraphicsPixmapItem):
    def __init__(self, owner, parent=None):
        super().__init__(parent)
        self._owner = owner

    def mouseDoubleClickEvent(self, event):
        try:
            print("DEBUG: PreviewPixmapItem.mouseDoubleClickEvent")
            if hasattr(self._owner, '_open_preview'):
                try:
                    self._owner._open_preview()
                except Exception as e:
                    print("DEBUG: _open_preview failed:", e)
        except Exception as e:
            print("DEBUG: Preview mouseDoubleClickEvent error:", e)
        super().mouseDoubleClickEvent(event)


class ComputeWorker(QThread):
    finished_sig = pyqtSignal(object)

    def __init__(self, func, parent=None):
        super().__init__(parent)
        self.func = func

    def run(self):
        res = {'success': True, 'result': None, 'exc': None}
        try:
            res['result'] = self.func()
        except Exception:
            import traceback
            res['success'] = False
            res['exc'] = traceback.format_exc()
        try:
            self.finished_sig.emit(res)
        except Exception:
            pass


class ImagePreviewWindow(QMainWindow):
    def __init__(self, pixmap, title="Preview"):
        super().__init__()
        self.setWindowTitle(title)
        self._label = QLabel()
        self._label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self._label)
        # store original pixmap and show it scaled to the current window size
        self._orig = pixmap
        try:
            if pixmap is not None:
                scaled = self._orig.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self._label.setPixmap(scaled)
        except Exception:
            try:
                self._label.setPixmap(self._orig)
            except Exception:
                pass

    def resizeEvent(self, event):
        try:
            if getattr(self, '_orig', None) is not None:
                w = max(100, self.width() - 20)
                h = max(80, self.height() - 40)
                scaled = self._orig.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self._label.setPixmap(scaled)
        except Exception:
            pass
        super().resizeEvent(event)


class NodeItem(QGraphicsRectItem):
    WIDTH = 160
    HEIGHT = 100
    SOCKET_SPACING = 22
    # extra top margin so sockets sit below the title
    MARGIN_TOP = 34
    MARGIN_BOTTOM = 10

    def __init__(self, backend_node):
        super().__init__()
        self.node = backend_node
        self.input_items = {}
        self.output_items = {}
        self._active_workers = []

        # initial rect; will expand when sockets are created
        self.setRect(0, 0, self.WIDTH, self.HEIGHT)
        self.setBrush(QBrush(QColor(50, 50, 50)))
        self.setPen(QPen(Qt.NoPen))

        # allow nodes to be dragged and selected
        try:
            self.setFlag(QGraphicsItem.ItemIsMovable, True)
            self.setFlag(QGraphicsItem.ItemIsSelectable, True)
            self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        except Exception:
            pass

        title = QGraphicsTextItem(type(backend_node).__name__, self)
        title.setDefaultTextColor(QColor(220, 220, 220))
        title.setPos(6, 3)

        # create any embedded UI first so sockets are positioned below it
        try:
            self._create_embedded_ui()
        except Exception:
            pass

        self._create_sockets()


    class ToggleSwitchItem(QGraphicsObject):
        def __init__(self, width=46, height=20, parent=None):
            super().__init__(parent)
            self._w = width
            self._h = height
            self._checked = False
            self._callback = None
            self.setAcceptedMouseButtons(Qt.LeftButton)

        def boundingRect(self):
            return QRectF(0, 0, self._w, self._h)

        def paint(self, painter, option, widget=None):
            painter.setRenderHint(QPainter.Antialiasing)
            rect = QRectF(0, 0, self._w, self._h)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(1, 166, 150) if self._checked else QColor(190, 190, 190)))
            painter.drawRoundedRect(rect, self._h / 2, self._h / 2)
            knob_d = self._h - 6
            knob_x = rect.right() - knob_d - 3 if self._checked else rect.left() + 3
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(QPen(QColor(200, 200, 200)))
            painter.drawEllipse(QRectF(knob_x, rect.top() + 3, knob_d, knob_d))

        def mousePressEvent(self, event):
            self._checked = not self._checked
            self.update()
            cb = getattr(self, '_callback', None)
            if callable(cb):
                try:
                    cb(self._checked)
                except Exception:
                    pass

        def setChecked(self, v: bool):
            self._checked = bool(v)
            self.update()

        def isChecked(self) -> bool:
            return bool(self._checked)

        def setCallback(self, func):
            self._callback = func

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.rect().width(), self.rect().height())

    def paint(self, painter: QPainter, option, widget=None):
        painter.setBrush(self.brush())
        painter.setPen(self.pen())
        painter.drawRoundedRect(self.boundingRect(), 4, 4)

    def _create_sockets(self):
        # Build socket visuals and inline editors for inputs and outputs
        inputs = getattr(self.node, 'inputs', {}) or {}
        outputs = getattr(self.node, 'outputs', {}) or {}

        max_count = max(len(inputs), len(outputs), 1)
        height_needed = self.MARGIN_TOP + max_count * self.SOCKET_SPACING + self.MARGIN_BOTTOM
        self._height = max(self.HEIGHT, height_needed)
        self.setRect(0, 0, self.WIDTH, self._height)

        # Inputs (left side)
        y = self.MARGIN_TOP
        for name, sock in inputs.items():
            item = SocketItem(sock, self, is_output=False)
            # align socket center with node left edge (half-overlapping)
            item.setPos(0, y)
            self.input_items[name] = item

            label = QGraphicsTextItem(name, self)
            label.setDefaultTextColor(QColor(200, 200, 200))
            try:
                lh = label.boundingRect().height()
                label.setPos(12, y - lh / 2)
            except Exception:
                label.setPos(12, y - 8)
            item._label_item = label

            # ensure sockets and labels render above the node background
            try:
                item.setZValue(3)
                label.setZValue(2)
            except Exception:
                pass

            # inline editor if modifiable
                # inputs do not host inline editors; Value nodes host editable outputs

            y += self.SOCKET_SPACING

        # Outputs (right side)
        y = self.MARGIN_TOP
        for name, sock in outputs.items():
            item = SocketItem(sock, self, is_output=True)
            # align socket center with node right edge (half-overlapping)
            item.setPos(self.rect().width(), y)
            self.output_items[name] = item

            label = QGraphicsTextItem(name, self)
            label.setDefaultTextColor(QColor(200, 200, 200))
            lw = label.boundingRect().width()
            # use actual rect width in case node was resized
            try:
                node_w = int(self.rect().width())
            except Exception:
                node_w = self.WIDTH
            try:
                lh = label.boundingRect().height()
                label.setPos(node_w - lw - 18, y - lh / 2)
            except Exception:
                label.setPos(node_w - lw - 18, y - 8)
            item._label_item = label

            try:
                item.setZValue(3)
                label.setZValue(2)
            except Exception:
                pass

            # If this output socket is modifiable (Value nodes), create an inline editor
            try:
                if getattr(sock, 'is_modifiable', False):
                    # place editor to the left of the label inside the node
                    if sock.socket_type == SocketType.BOOLEAN:
                        w = self.ToggleSwitchItem()
                        v = getattr(self.node, name, False)
                        w.setChecked(bool(v))
                        def _on_bool_out(val, s=sock, n=name):
                            try:
                                setattr(self.node, n, bool(val))
                            except Exception:
                                pass
                            try:
                                s._cache = bool(val)
                                s._dirty = False
                            except Exception:
                                pass
                            try:
                                self.node.mark_dirty()
                            except Exception:
                                pass
                        w.setCallback(_on_bool_out)
                        w.setParentItem(self)
                        # position left of the label
                        try:
                            node_w = int(self.rect().width())
                        except Exception:
                            node_w = self.WIDTH
                        editor_x = node_w - lw - 18 - int(w._w) - 8
                        if editor_x < 8:
                            editor_x = 8
                        try:
                            h = int(w._h)
                            w.setPos(editor_x, y - h // 2)
                        except Exception:
                            w.setPos(editor_x, y - 10)
                        w.setZValue(2)
                        item._modifiable_widget = w
                    else:
                        if sock.socket_type == SocketType.INT:
                            widget = QSpinBox()
                        elif sock.socket_type == SocketType.FLOAT:
                            widget = QDoubleSpinBox()
                        else:
                            widget = QLineEdit()
                        proxy = QGraphicsProxyWidget()
                        proxy.setWidget(widget)
                        proxy.setParentItem(self)
                        try:
                            node_w = int(self.rect().width())
                        except Exception:
                            node_w = self.WIDTH
                        # choose width and position to the left of the label
                        try:
                            avail = max(40, int(node_w - 16 - lw - 8))
                            widget.setFixedWidth(min(120, avail))
                        except Exception:
                            pass
                        editor_x = node_w - lw - 18 - int(widget.width()) - 8
                        if editor_x < 8:
                            editor_x = 8
                        try:
                            ph = proxy.boundingRect().height()
                            proxy.setPos(editor_x, y - ph / 2)
                        except Exception:
                            proxy.setPos(editor_x, y - 12)
                        proxy.setZValue(2)
                        item._modifiable_widget = proxy
                        # initialize value
                        try:
                            init_val = getattr(self.node, name, None)
                            if init_val is not None:
                                if isinstance(widget, QSpinBox):
                                    widget.setValue(int(init_val))
                                elif isinstance(widget, QDoubleSpinBox):
                                    widget.setValue(float(init_val))
                                else:
                                    widget.setText(str(init_val))
                        except Exception:
                            pass
                        # wire changes back to node attribute and socket cache
                        try:
                            if isinstance(widget, QSpinBox):
                                widget.valueChanged.connect(lambda val, s=sock, n=name: (setattr(self.node, n, int(val)), setattr(s, '_cache', int(val)), setattr(s, '_dirty', False), self.node.mark_dirty()))
                            elif isinstance(widget, QDoubleSpinBox):
                                widget.valueChanged.connect(lambda val, s=sock, n=name: (setattr(self.node, n, float(val)), setattr(s, '_cache', float(val)), setattr(s, '_dirty', False), self.node.mark_dirty()))
                            else:
                                widget.textChanged.connect(lambda text, s=sock, n=name: (setattr(self.node, n, str(text)), setattr(s, '_cache', str(text)), setattr(s, '_dirty', False), self.node.mark_dirty()))
                        except Exception:
                            pass
            except Exception:
                pass

            y += self.SOCKET_SPACING

    def _create_embedded_ui(self):
        # SourceImageNode: show a combobox of available images
        try:
            if SourceImageNode is not None and isinstance(self.node, SourceImageNode):
                files = getattr(self.node, '_image_files', []) or []
                combo = FixedWidthComboBox()
                for f in files:
                    combo.addItem(f)
                try:
                    combo.setCurrentIndex(int(getattr(self.node, 'index', 0)))
                except Exception as e:
                    print(e)
                # compute the width needed to display the longest item and expand node if needed
                try:
                    cur_w = int(self.rect().width())
                    fm = QFontMetrics(combo.font())
                    max_text_w = 0
                    for f in files:
                        try:
                            w = fm.horizontalAdvance(str(f))
                        except Exception:
                            try:
                                w = fm.width(str(f))
                            except Exception:
                                w = 0
                        if w > max_text_w:
                            max_text_w = w
                    # add padding for combo arrow and margins
                    padding = 40
                    desired_w = max(cur_w, max_text_w + padding)
                    if desired_w > cur_w:
                        self.setRect(0, 0, desired_w, self.rect().height())
                except Exception:
                    pass
                combo.setFixedWidth(int(self.rect().width() - 41))
                try:
                    # prefer not to auto-adjust to contents so popup sizing is controlled
                    combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLength)
                except Exception:
                    pass
                try:
                    combo.view().setFixedWidth(int(self.rect().width() - 41))
                except Exception:
                    pass
                proxy = QGraphicsProxyWidget(self)
                proxy.setWidget(combo)
                try:
                    # ensure popup width matches the combo width (also handled in showPopup)
                    combo.view().setFixedWidth(int(self.rect().width() - 41))
                except Exception:
                    pass
                # place under the title
                proxy.setPos(8, 18)
                # make embedded UI sit below the node background but beneath sockets
                try:
                    proxy.setZValue(1)
                except Exception:
                    pass

                def _on_src(idx):
                    try:
                        self.node.index = int(idx)
                    except Exception as e:
                        print(e)
                    try:
                        if getattr(self.node, '_images', None):
                            i = int(getattr(self.node, 'index', 0))
                            if 0 <= i < len(self.node._images):
                                self.node.outputs['image']._cache = self.node._images[i]
                                self.node.outputs['image']._dirty = False
                    except Exception as e:
                        print(e)

                try:
                    combo.activated.connect(_on_src)
                except Exception as e:
                    print(e)

        except Exception as e:
            print(e)

        # ViewerNode: embed a small preview and Run/Fetch buttons
        try:
            if ViewerNode is not None and isinstance(self.node, ViewerNode):
                # preview pixmap item placed between title and buttons
                preview = PreviewPixmapItem(self, parent=self)
                preview.setZValue(4)
                # preview target size - leave small margins
                pv_w = int(self.rect().width() - 16)
                pv_h = 84
                # create a placeholder blank pixmap
                try:
                    blank = QPixmap(pv_w, pv_h)
                    blank.fill(QColor(40, 40, 40))
                    preview.setPixmap(blank)
                except Exception as e:
                    print(e)
                preview.setPos(8, 18)
                try:
                    preview.setAcceptedMouseButtons(Qt.LeftButton)
                    preview.setAcceptHoverEvents(True)
                except Exception:
                    pass
                preview.setScale(1.0)
                # store reference for preview windows
                try:
                    self._preview_windows = []
                except Exception:
                    pass
                # helper to update preview from the node's input image
                def _update_preview():
                    try:
                        try:
                            print(f"DEBUG: _update_preview called for node {type(self.node).__name__}")
                        except Exception:
                            pass
                        inp = self.node.inputs.get('image')
                        if inp is None:
                            try:
                                print("DEBUG: _update_preview - no 'image' input on node")
                            except Exception:
                                pass
                            return
                        im = inp.get()
                        if im is None:
                            try:
                                print("DEBUG: _update_preview - input.get() returned None")
                            except Exception:
                                pass
                            return
                        # avoid redundant updates
                        last = getattr(preview, '_last_image_id', None)
                        if last == id(im):
                            return
                        preview._last_image_id = id(im)
                        try:
                            qim = ImageQt(im)
                            pix = QPixmap.fromImage(qim)
                        except Exception:
                            return
                        # store full pixmap for opening resizable preview
                        preview._full_pix = pix
                        try:
                            print(f"DEBUG: _update_preview - setting preview for node {type(self.node).__name__}")
                        except Exception:
                            pass
                        try:
                            scaled = pix.scaled(pv_w, pv_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                            preview.setPixmap(scaled)
                        except Exception:
                            try:
                                preview.setPixmap(pix)
                            except Exception:
                                pass
                    except Exception:
                        pass

                # do NOT poll for input changes automatically; preview updates only when Run is pressed
                try:
                    self._preview_timer = None
                except Exception:
                    pass
                # ensure sockets are placed below preview by increasing top margin
                try:
                    orig_top = int(getattr(self, 'MARGIN_TOP', 34))
                    self.MARGIN_TOP = orig_top + pv_h + 12
                except Exception:
                    try:
                        self.MARGIN_TOP = self.MARGIN_TOP + pv_h + 12
                    except Exception:
                        pass
                # buttons
                btn_run = QPushButton('Run')
                btn_fetch = QPushButton('Fetch')
                p_run = QGraphicsProxyWidget(self)
                p_fetch = QGraphicsProxyWidget(self)
                p_run.setWidget(btn_run)
                p_fetch.setWidget(btn_fetch)
                # position buttons under preview (they will be re-positioned after sockets are created)
                p_run.setPos(8, self.MARGIN_TOP - 36)
                p_fetch.setPos(72, self.MARGIN_TOP - 36)
                p_run.setZValue(4)
                p_fetch.setZValue(4)

                def _call_run():
                    try:
                        try:
                            print(f"DEBUG: ViewerNode _call_run invoked for node {type(self.node).__name__}")
                        except Exception:
                            pass
                        win = None
                        s = self.scene()
                        if s is not None:
                            vs = s.views()
                            if vs:
                                win = vs[0].window()
                        if win is not None and hasattr(win, 'run_graph'):
                            try:
                                win.run_graph()
                            except Exception:
                                pass
                        # immediate update (also handled by timer)
                        # Run triggers an immediate preview update
                        try:
                            _update_preview()
                        except Exception:
                            pass
                    except Exception:
                        pass

                def _call_fetch():
                    try:
                        try:
                            print(f"DEBUG: ViewerNode _call_fetch invoked for node {type(self.node).__name__}")
                        except Exception:
                            pass
                        win = None
                        s = self.scene()
                        if s is not None:
                            vs = s.views()
                            if vs:
                                win = vs[0].window()
                        if win is not None and hasattr(win, 'fetch_graph'):
                            try:
                                win.fetch_graph()
                            except Exception:
                                pass
                        # Fetch does not update the embedded preview
                        pass
                    except Exception:
                        pass

                try:
                    btn_run.clicked.connect(_call_run)
                except Exception:
                    pass
                try:
                    btn_fetch.clicked.connect(_call_fetch)
                except Exception:
                    pass
                # RenderToFileNode: add Save button to save input image directly
                try:
                    if RenderToFileNode is not None and isinstance(self.node, RenderToFileNode):
                        btn_save = QPushButton('Save')
                        p_save = QGraphicsProxyWidget(self)
                        p_save.setWidget(btn_save)
                        # position the save button next to others
                        try:
                            p_save.setPos(136, self.MARGIN_TOP - 36)
                        except Exception:
                            p_save.setPos(136, self.MARGIN_TOP - 36)
                        p_save.setZValue(4)

                        def _call_save():
                            try:
                                print(f"DEBUG: RenderToFileNode _call_save for node {type(self.node).__name__}")
                            except Exception:
                                pass
                            try:
                                inp = self.node.inputs.get('image')
                                if inp is None:
                                    print("DEBUG: Save - no image input socket")
                                    return
                                img = inp.get()
                                if img is None:
                                    print("DEBUG: Save - input image is None")
                                    return
                                # prefer a filepath provided on the node
                                fp = None
                                try:
                                    fp = self.node.inputs.get('filepath') and self.node.inputs.get('filepath').get()
                                except Exception:
                                    fp = None
                                if not fp:
                                    # prompt user
                                    try:
                                        fp, _ = QFileDialog.getSaveFileName(None, "Save Image", "assets/printouts/output.png", "PNG Files (*.png);;All Files (*)")
                                    except Exception:
                                        fp = None
                                if not fp:
                                    print("DEBUG: Save cancelled or no filepath")
                                    return
                                try:
                                    img.save(fp)
                                    print(f"DEBUG: Saved image to {fp}")
                                except Exception as e:
                                    print("DEBUG: Failed to save image:", e)
                            except Exception as e:
                                print("DEBUG: _call_save error:", e)

                        try:
                            btn_save.clicked.connect(_call_save)
                        except Exception:
                            pass
                except Exception:
                    pass
                # expose _open_preview on this NodeItem so PreviewPixmapItem can call it
                def _open_preview():
                    try:
                        print(f"DEBUG: _open_preview called for node {type(self.node).__name__}")
                        pix = getattr(preview, '_full_pix', None)
                        if pix is None:
                            print("DEBUG: _open_preview - no pix available")
                            return
                        win = ImagePreviewWindow(pix, title=f"Preview - {type(self.node).__name__}")
                        try:
                            win.setAttribute(Qt.WA_DeleteOnClose)
                        except Exception:
                            pass
                        win.show()
                        try:
                            self._preview_windows.append(win)
                        except Exception:
                            pass
                    except Exception:
                        pass

                try:
                    # attach as instance method
                    self._open_preview = _open_preview
                except Exception:
                    pass
        except Exception:
            pass


# Provide a simple paint method for compatibility if desired
def _nodeitem_paint(self, painter, option, widget=None):
    try:
        painter.setBrush(self.brush())
        painter.setPen(self.pen())
        painter.drawRoundedRect(self.boundingRect(), 4, 4)
    except Exception:
        pass

