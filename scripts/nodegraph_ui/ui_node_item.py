# scripts/nodegraph_ui/ui_node_item.py

from PyQt5.QtWidgets import QGraphicsItem, QGraphicsTextItem, QGraphicsRectItem, QGraphicsProxyWidget, QComboBox, QGraphicsPixmapItem, QPushButton, QSpinBox, QDoubleSpinBox, QLineEdit, QCheckBox, QMainWindow, QLabel, QApplication, QWidget, QGraphicsObject
from PyQt5.QtCore import QRectF, QPointF, Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPen, QBrush, QColor, QPainter, QPixmap, QImage

from PIL.ImageQt import ImageQt
from PIL.Image import Image as PILImage
import io

from .ui_socket_item import SocketItem, RADIUS
from .nodes import SourceImageNode, ViewerNode, RenderToFileNode
from .classes import SocketType


class PreviewPixmapItem(QGraphicsPixmapItem):
    def __init__(self, owner, parent=None):
        super().__init__(parent)
        self._owner = owner
        try:
            self.setAcceptedMouseButtons(Qt.LeftButton)
            self.setAcceptHoverEvents(True)
        except Exception:
            pass

    def mouseDoubleClickEvent(self, event):
        try:
            owner_node = getattr(getattr(self, '_owner', None), 'node', None)
            print(f"[ui_node_item] preview double-clicked owner_node={type(owner_node).__name__ if owner_node is not None else None} id={id(owner_node) if owner_node is not None else None}")
            if hasattr(self._owner, '_open_preview'):
                self._owner._open_preview()
        except Exception:
            pass
        super().mouseDoubleClickEvent(event)


class ComputeWorker(QThread):
    """Run a callable in a background thread and emit result info when finished."""
    finished_sig = pyqtSignal(object)

    def __init__(self, func, parent=None):
        super().__init__(parent)
        self.func = func

    def run(self):
        res = { 'success': True, 'result': None, 'exc': None }
        try:
            r = self.func()
            res['result'] = r
        except Exception as e:
            import traceback
            res['success'] = False
            res['exc'] = traceback.format_exc()
        try:
            # emit a plain dict to avoid PyQt wrapping issues
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
        self._orig = pixmap
        # determine a reasonable initial window size (cap to 80% of screen)
        try:
            screen = QApplication.primaryScreen()
            geom = screen.availableGeometry()
            max_w = int(geom.width() * 0.8)
            max_h = int(geom.height() * 0.8)
        except Exception:
            max_w = 1600
            max_h = 1000

        init_w = min(max(200, self._orig.width()), max_w)
        init_h = min(max(200, self._orig.height()), max_h)
        # scale original pixmap to initial window size while keeping aspect
        try:
            scaled = self._orig.scaled(init_w, init_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self._label.setPixmap(scaled)
        except Exception:
            try:
                self._label.setPixmap(self._orig)
            except Exception:
                pass

        self.setMinimumSize(200, 200)
        self.resize(init_w, init_h)

    def resizeEvent(self, event):
        try:
            if self._orig is not None and not self._orig.isNull():
                w = self.centralWidget().width()
                h = self.centralWidget().height()
                try:
                    scaled = self._orig.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self._label.setPixmap(scaled)
                except Exception:
                    # fallback: set original pixmap
                    try:
                        self._label.setPixmap(self._orig)
                    except Exception:
                        pass
        except Exception:
            pass
        super().resizeEvent(event)

class NodeItem(QGraphicsRectItem):
    WIDTH = 160
    HEIGHT = 100
    SOCKET_SPACING = 20
    MARGIN_TOP = 25
    MARGIN_BOTTOM = 10

    def __init__(self, backend_node):
        super().__init__()
        self.node = backend_node
        # keep references to background workers so they aren't GC'd
        self._active_workers = []

        # Node background (draggable) - use self as the rect item
        super().__init__(0, 0, self.WIDTH, self.HEIGHT, parent=None)

        # Set background color based on node type
        node_type = getattr(self.node, 'node_type', 'unknown')
        if node_type == "input":
            bg_color = QColor(60, 90, 160)  # blue-ish
        elif node_type == "output":
            bg_color = QColor(90, 60, 60)   # red-ish
        elif node_type == "processor":
            bg_color = QColor(50, 50, 50)   # default dark
        else:
            bg_color = QColor(80, 80, 80)
        self.setBrush(QBrush(bg_color))
        self.setZValue(-1)  # keep background behind embedded widgets (they use higher z)
        # Enable selection so nodes can be clicked/selected
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        # Allow the node to be moved by dragging (left mouse button)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        # Notify on position changes
        self.setFlag(QGraphicsItem.ItemSendsScenePositionChanges, True)
        # Let the parent NodeItem receive mouse events rather than the bg child
        self.setAcceptHoverEvents(False)

        # Title
        title = QGraphicsTextItem(type(backend_node).__name__, self)
        title.setDefaultTextColor(QColor(220, 220, 220))  # light gray
        title.setPos(5, 2)
        # Ensure title text does not become selectable/focusable (prevents outline)
        title.setFlag(QGraphicsItem.ItemIsSelectable, False)
        title.setFlag(QGraphicsItem.ItemIsFocusable, False)
        title.setAcceptedMouseButtons(Qt.NoButton)
        title.setAcceptHoverEvents(False)

        # Remove border so no outline is visible
        self.setPen(QPen(Qt.NoPen))

        self.input_items = {}
        self.output_items = {}

        self._create_sockets()

        # Compute dynamic height based on number of sockets
        n_inputs = len(self.node.inputs) if hasattr(self.node, 'inputs') else 0
        n_outputs = len(self.node.outputs) if hasattr(self.node, 'outputs') else 0
        max_sockets = max(n_inputs, n_outputs, 1)
        computed_height = self.MARGIN_TOP + max_sockets * self.SOCKET_SPACING + self.MARGIN_BOTTOM
        # reserve extra space for viewer preview if needed
        try:
            if hasattr(self.node, 'node_type') and self.node.node_type == "output":
                computed_height += 40
            if isinstance(self.node, ViewerNode):
                computed_height += 120
        except Exception:
            pass
        self._height = max(self.HEIGHT, computed_height)
        # Update background rect to new height
        self.setRect(0, 0, self.WIDTH, self._height)

    # Small toggle widget drawn to look like a modern switch
    class ToggleSwitch(QCheckBox):
        def __init__(self, parent=None, width=48, height=24):
            super().__init__(parent)
            self._w = width
            self._h = height
            # Make the widget background transparent so only the custom painting is visible
            try:
                self.setAttribute(Qt.WA_TranslucentBackground, True)
                self.setAutoFillBackground(False)
                self.setStyleSheet("background: transparent; border: 0px;")
            except Exception:
                pass
            self.setFixedSize(QSize(self._w, self._h))
            self.setCursor(Qt.PointingHandCursor)

        def sizeHint(self):
            return QSize(self._w, self._h)

        def paintEvent(self, event):
            p = QPainter(self)
            p.save()
            p.setRenderHint(QPainter.Antialiasing)
            checked = self.isChecked()
            # colors
            track_on = QColor(1, 166, 150)  # teal-ish
            track_off = QColor(190, 190, 190)
            knob_color = QColor(255, 255, 255)

            # draw track
            radius = self._h / 2
            rect = self.rect()
            track_rect = rect.adjusted(0, 0, -1, -1)
            p.setPen(QPen(Qt.NoPen))
            p.setBrush(QBrush(track_on if checked else track_off))
            p.drawRoundedRect(track_rect, radius, radius)

            # draw knob
            knob_d = self._h - 4
            knob_y = 2
            knob_x = self._w - knob_d - 2 if checked else 2
            p.setBrush(QBrush(knob_color))
            p.setPen(QPen(QColor(200, 200, 200)))
            p.drawEllipse(knob_x, knob_y, knob_d, knob_d)
            p.restore()
            try:
                # lightweight debug hint when running in dev
                if getattr(self, '_debug_print', False):
                    print(f"[ui_node_item] ToggleSwitch.paintEvent checked={checked} pos={self.pos()} size=({self._w},{self._h})")
            except Exception:
                pass

    # Lightweight graphics-only toggle implemented as a QGraphicsObject to
    # avoid the overhead of embedding QWidget proxies in the scene.
    class ToggleSwitchItem(QGraphicsObject):
        def __init__(self, width=48, height=24, parent=None):
            super().__init__(parent)
            self._w = width
            self._h = height
            self._checked = False
            self.setAcceptHoverEvents(True)
            # accept mouse presses so the toggle can be interacted with directly
            try:
                self.setAcceptedMouseButtons(Qt.LeftButton)
            except Exception:
                pass
            try:
                self.setFlag(QGraphicsItem.ItemIsSelectable, False)
            except Exception:
                pass

        def boundingRect(self):
            return QRectF(0, 0, self._w, self._h)

        def paint(self, painter, option, widget=None):
            painter.save()
            painter.setRenderHint(QPainter.Antialiasing)
            checked = self._checked
            track_on = QColor(1, 166, 150)
            track_off = QColor(190, 190, 190)
            knob_color = QColor(255, 255, 255)

            radius = self._h / 2
            rect = self.boundingRect().adjusted(0, 0, -1, -1)
            # ensure no pen (outline) when drawing the track
            painter.setPen(QPen(Qt.NoPen))
            painter.setBrush(QBrush(track_on if checked else track_off))
            painter.drawRoundedRect(rect, radius, radius)

            knob_d = self._h - 4
            knob_y = 2
            knob_x = (self._w - knob_d - 2) if checked else 2
            painter.setBrush(QBrush(knob_color))
            painter.setPen(QPen(QColor(200, 200, 200)))
            painter.drawEllipse(knob_x, knob_y, knob_d, knob_d)
            painter.restore()
            # debug outline to visualize the ToggleSwitchItem bounds
            try:
                if getattr(self, '_debug_print', False):
                    try:
                        opt_pen = QPen(QColor(255, 0, 255, 200))
                        opt_pen.setStyle(Qt.DashLine)
                        saved = painter.save()
                        # draw overlay at higher z by creating a temporary QPainter on scene?
                        # fallback: emit print so logs show rect.
                        print(f"[ui_node_item] ToggleSwitchItem bounds={self.boundingRect()} checked={checked} parent={type(self.parentItem()).__name__ if self.parentItem() is not None else None}")
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                if getattr(self, '_debug_print', False):
                    print(f"[ui_node_item] ToggleSwitchItem.paint checked={checked} rect={rect} parent={type(self.parentItem()).__name__ if self.parentItem() is not None else None}")
            except Exception:
                pass

        def mousePressEvent(self, event):
            try:
                self._checked = not self._checked
                self.update()
                # call optional callback set by creator
                cb = getattr(self, '_callback', None)
                if callable(cb):
                    try:
                        cb(self._checked)
                    except Exception:
                        pass
                try:
                    if getattr(self, '_debug_print', False):
                        print(f"[ui_node_item] ToggleSwitch.mousePressEvent checked={self._checked} pos={self.pos()}")
                except Exception:
                    pass
            except Exception:
                pass

        def setChecked(self, v: bool):
            self._checked = bool(v)
            self.update()

        def isChecked(self) -> bool:
            return bool(self._checked)

        def setCallback(self, func):
            self._callback = func

            # If this is a SourceImageNode, embed a combobox to pick the loaded image
            try:
                if isinstance(self.node, SourceImageNode) and getattr(self.node, '_image_files', None) is not None:
                    combo = QComboBox()
                    for fname in self.node._image_files:
                        combo.addItem(fname)
                    combo.setCurrentIndex(getattr(self.node, 'index', 0))
                    combo.setFixedWidth(self.WIDTH - 10)

                    def _on_combo(idx, node=self.node):
                        try:
                            node.index = idx
                            # allow immediate compute/update if needed
                            node.compute()
                        except Exception:
                            pass

                    combo.currentIndexChanged.connect(_on_combo)
                    proxy = QGraphicsProxyWidget(self)
                    proxy.setWidget(combo)
                    proxy.setPos(5, 18)
                    proxy.setZValue(1)
                    self._combo_proxy = proxy
            except Exception:
                pass

            # If this is a ViewerNode, add a pixmap preview area that polls the input
            try:
                if isinstance(self.node, ViewerNode):
                    self._pix_item = PreviewPixmapItem(self, parent=self)
                    preview_h = 100
                    pix_x = 5
                    pix_y = self.MARGIN_TOP + 30
                    self._pix_item.setPos(pix_x, pix_y)
                    self._pix_item.setZValue(1)

                    # status text for debugging
                    self._viewer_status_text = QGraphicsTextItem('', self)
                    self._viewer_status_text.setDefaultTextColor(QColor(200, 200, 200))
                    status_x = pix_x
                    status_y = pix_y + preview_h + 6
                    self._viewer_status_text.setPos(status_x, status_y)

                    def refresh_viewer(self_ref=None):
                        # allow being called as bound method or standalone closure
                        self_obj = self if self_ref is None else self_ref
                        try:
                            inp = self_obj.node.inputs.get('image')
                            if inp is None:
                                try:
                                    self_obj._viewer_status_text.setPlainText('no input socket')
                                except Exception:
                                    pass
                                return
                            out = None
                            try:
                                out = inp.connection.output_socket if inp.connection is not None else None
                            except Exception:
                                out = None

                            # prefer using cached result to avoid triggering compute
                            cached = None
                            try:
                                cached = getattr(out, '_cache', None) if out is not None else None
                            except Exception:
                                cached = None

                            # skip heavy conversion work if the cached image instance hasn't changed
                            try:
                                cid = id(cached) if cached is not None else None
                                last_cid = getattr(self_obj, '_last_viewer_cache_id', None)
                                if cid is not None and cid == last_cid:
                                    # still update a light status line, but avoid re-creating pixmap
                                    try:
                                        if isinstance(cached, PILImage):
                                            s = getattr(cached, 'size', None)
                                            m = getattr(cached, 'mode', None)
                                            self_obj._viewer_status_text.setPlainText(f'PIL Image {s} mode={m} (cached)')
                                    except Exception:
                                        pass
                                    return
                                # store id for next tick
                                try:
                                    self_obj._last_viewer_cache_id = cid
                                except Exception:
                                    pass
                            except Exception:
                                pass

                            if cached is None:
                                try:
                                    self_obj._pix_item.setPixmap(QPixmap())
                                except Exception:
                                    pass
                                try:
                                    self_obj._viewer_status_text.setPlainText('input connected (no cache)')
                                except Exception:
                                    pass
                                return

                            img = cached

                            try:
                                qimg = None
                                # Try ImageQt first (safer for many PIL modes)
                                try:
                                    qimg = ImageQt(img)
                                    # PIL.ImageQt.ImageQt can be an object type not accepted
                                    # by QPixmap.fromImage across PyQt/PySide bindings —
                                    # prefer using PNG-bytes fallback instead of passing
                                    # ImageQt instances directly.
                                    from PIL.ImageQt import ImageQt as _ImageQtType
                                    if isinstance(qimg, _ImageQtType):
                                        qimg = None
                                except Exception as e:
                                    print(f"[ui_node_item] refresh_viewer: ImageQt failed: {e}")
                                    qimg = None

                                # Fallback: save to PNG bytes and load via QImage.fromData
                                if qimg is None:
                                    try:
                                        
                                        buf = io.BytesIO()
                                        img.save(buf, format='PNG')
                                        data = buf.getvalue()
                                        qimg = QImage.fromData(data)
                                    except Exception as e:
                                        print(f"[ui_node_item] refresh_viewer: PNG-bytes fallback failed: {e}")
                                        qimg = None

                                # Last resort: raw bytes path
                                if qimg is None:
                                    try:
                                        if hasattr(img, 'mode') and img.mode == 'RGB':
                                            data = img.tobytes('raw', 'RGB')
                                            qimg = QImage(data, img.width, img.height, QImage.Format_RGB888)
                                        else:
                                            img2 = img.convert('RGBA')
                                            data = img2.tobytes('raw', 'RGBA')
                                            try:
                                                qimg = QImage(data, img2.width, img2.height, QImage.Format_RGBA8888)
                                            except Exception:
                                                qimg = QImage(data, img2.width, img2.height, QImage.Format_ARGB32)
                                    except Exception as e:
                                        print(f"[ui_node_item] refresh_viewer: raw-bytes fallback failed: {e}")
                                        qimg = None

                                if qimg is None:
                                    print(f"[ui_node_item] refresh_viewer: failed to create QImage from PIL image; skipping preview update")
                                    try:
                                        if isinstance(img, PILImage):
                                            s = getattr(img, 'size', None)
                                            m = getattr(img, 'mode', None)
                                            self_obj._viewer_status_text.setPlainText(f'preview unavailable size={s} mode={m}')
                                        else:
                                            self_obj._viewer_status_text.setPlainText('preview unavailable (conversion failed)')
                                    except Exception:
                                        pass
                                    return

                                try:
                                    pix = QPixmap.fromImage(qimg)
                                except TypeError as e:
                                    try:
                                        buf = io.BytesIO()
                                        img.save(buf, format='PNG')
                                        data = buf.getvalue()
                                        qimg2 = QImage.fromData(data)
                                        pix = QPixmap.fromImage(qimg2)
                                    except Exception as e2:
                                        print(f"[ui_node_item] refresh_viewer: PNG-bytes fallback failed: {e2}")
                                        pix = None

                                w = int(self_obj.WIDTH - 10)
                                if pix is not None:
                                    pix = pix.scaled(w, preview_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                                    self_obj._pix_item.setPixmap(pix)
                                try:
                                    if isinstance(img, PILImage):
                                        s = getattr(img, 'size', None)
                                        m = getattr(img, 'mode', None)
                                        self_obj._viewer_status_text.setPlainText(f'PIL Image {s} mode={m}')
                                    else:
                                        self_obj._viewer_status_text.setPlainText(f'Got {type(img)}')
                                except Exception:
                                    pass
                                    print(f"[ui_node_item] refresh_viewer: preview update complete")
                            except Exception as e:
                                import traceback
                                print(f"[ui_node_item] refresh_viewer: failed to convert/display image: {e}")
                                traceback.print_exc()
                        except Exception:
                            pass

                    self._viewer_timer = QTimer()
                    self._viewer_timer.timeout.connect(lambda: refresh_viewer())
                    # increase polling interval to reduce CPU when many/large images exist
                    self._viewer_timer.start(1000)
                    # Buttons: GET (force compute) and FETCH (use cache only)
                    get_btn = QPushButton('GET')
                    fetch_btn = QPushButton('FETCH')
                    # size buttons to avoid overlap
                    try:
                        get_btn.setFixedWidth(50)
                        fetch_btn.setFixedWidth(50)
                    except Exception:
                        pass

                    def on_get():
                        try:
                            print(f"[ui_node_item] GET clicked for ViewerNode id={id(self.node)}")
                            inp = self.node.inputs.get('image')
                            if inp is None:
                                return

                            # recursive compute upstream nodes
                            visited = set()

                            def compute_upstream(out_socket):
                                node = out_socket.node
                                print(f"[ui_node_item] compute_upstream: computing {type(node).__name__} id={id(node)}")
                                if node in visited:
                                    return
                                visited.add(node)
                                # compute dependencies first
                                for ins in node.inputs.values():
                                    if ins.connection is not None:
                                        compute_upstream(ins.connection.output_socket)
                                try:
                                    node.compute()
                                    print(f"[ui_node_item] compute_upstream: completed compute for {type(node).__name__} id={id(node)}")
                                    # mark outputs as computed (clear dirty flag)
                                    for o in node.outputs.values():
                                        try:
                                            o._dirty = False
                                        except Exception:
                                            pass
                                except Exception as e:
                                    import traceback
                                    print(f"[ui_node_item] compute_upstream: exception computing {type(node).__name__} id={id(node)}: {e}")
                                    traceback.print_exc()

                            if inp.connection is None:
                                return

                            # run compute in background to avoid blocking UI
                            try:
                                if getattr(self, '_viewer_status_text', None) is not None:
                                    self._viewer_status_text.setPlainText('computing...')
                            except Exception:
                                pass

                            def worker_func():
                                try:
                                    compute_upstream(inp.connection.output_socket)
                                    return True
                                except Exception:
                                    raise

                            worker = ComputeWorker(worker_func)

                            def on_finished(res):
                                try:
                                    if res.get('success'):
                                        # schedule UI update on main thread
                                        try:
                                            from PyQt5.QtCore import QTimer as _QTimer
                                            _QTimer.singleShot(0, lambda: refresh_viewer(self))
                                        except Exception:
                                            try:
                                                refresh_viewer(self)
                                            except Exception:
                                                pass
                                    else:
                                        print(f"[ui_node_item] compute worker failed:\n{res.get('exc')}")
                                    try:
                                        if getattr(self, '_viewer_status_text', None) is not None:
                                            self._viewer_status_text.setPlainText('idle')
                                    except Exception:
                                        pass
                                except Exception:
                                    pass

                            try:
                                # keep reference so worker isn't GC'd
                                self._active_workers.append(worker)
                                def _cleanup(res):
                                    try:
                                        # remove reference
                                        try:
                                            self._active_workers.remove(worker)
                                        except Exception:
                                            pass
                                        on_finished(res)
                                    except Exception:
                                        pass

                                worker.finished_sig.connect(_cleanup)
                                worker.start()
                            except Exception:
                                # fallback to synchronous compute
                                try:
                                    compute_upstream(inp.connection.output_socket)
                                    refresh_viewer(self)
                                except Exception:
                                    pass
                        except Exception:
                            pass

                    def on_fetch():
                        try:
                            print(f"[ui_node_item] FETCH clicked for ViewerNode id={id(self.node)}")
                            inp = self.node.inputs.get('image')
                            if inp is None or inp.connection is None:
                                return
                            out = inp.connection.output_socket
                            # only use cached result; do not trigger compute
                            cached = getattr(out, '_cache', None)
                            if cached is None:
                                print(f"[ui_node_item] FETCH: no cached image available on output {out.name}")
                                return
                            try:
                                qim = None
                                try:
                                    qim = ImageQt(cached)
                                except Exception:
                                    qim = None

                                if qim is None:
                                    # fallback: save to PNG bytes and load into QImage
                                    try:
                                        buf = io.BytesIO()
                                        cached.save(buf, format='PNG')
                                        data = buf.getvalue()
                                        qimg = QImage.fromData(data)
                                        pix = QPixmap.fromImage(qimg)
                                    except Exception:
                                        pix = None
                                else:
                                    # If ImageQt returned an ImageQt instance, avoid passing
                                    # it directly to QPixmap.fromImage; use PNG-bytes instead.
                                    from PIL.ImageQt import ImageQt as _ImageQtType
                                    if isinstance(qim, _ImageQtType):
                                        try:
                                            buf = io.BytesIO()
                                            cached.save(buf, format='PNG')
                                            data = buf.getvalue()
                                            qimg = QImage.fromData(data)
                                            pix = QPixmap.fromImage(qimg)
                                        except Exception:
                                            pix = None
                                    else:
                                        try:
                                            pix = QPixmap.fromImage(qim)
                                        except Exception:
                                            pix = None

                                if pix is not None:
                                    w = int(self.WIDTH - 10)
                                    pix = pix.scaledToWidth(w)
                                    self._pix_item.setPixmap(pix)
                            except Exception as e:
                                import traceback
                                print(f"[ui_node_item] on_fetch: failed to display cached image: {e}")
                                traceback.print_exc()
                        except Exception:
                            pass

                    get_proxy = QGraphicsProxyWidget(self)
                    fetch_proxy = QGraphicsProxyWidget(self)
                    get_proxy.setWidget(get_btn)
                    fetch_proxy.setWidget(fetch_btn)
                    # place buttons under the status text
                    btn_y = status_y + 20
                    get_proxy.setPos(5, btn_y)
                    fetch_proxy.setPos(5 + get_btn.width() + 6, btn_y)
                    get_proxy.setZValue(2)
                    fetch_proxy.setZValue(2)
                    get_btn.clicked.connect(on_get)
                    fetch_btn.clicked.connect(on_fetch)
                    # preview support: open resizable window on double-click
                    def _open_preview():
                        try:
                            print(f"[ui_node_item] _open_preview called for node id={id(self.node)} type={type(self.node).__name__}")
                            # Try to obtain a full-resolution PIL image from upstream output
                            try:
                                inp = self.node.inputs.get('image')
                                upstream_img = None
                                if inp is not None and inp.connection is not None:
                                    out = inp.connection.output_socket
                                    # prefer cached full image
                                    upstream_img = getattr(out, '_cache', None)
                                    if upstream_img is None:
                                        try:
                                            upstream_img = out.get()
                                        except Exception:
                                            upstream_img = None
                                # fallback to current pixmap if no upstream image
                                if upstream_img is None:
                                    pix = self._pix_item.pixmap()
                                    if pix is None or pix.isNull():
                                        return
                                    title = f"Preview - {type(self.node).__name__}"
                                    win = ImagePreviewWindow(pix, title=title)
                                    self._preview_win = win
                                    win.show()
                                    return

                                # convert PIL Image to QPixmap for full-resolution preview
                                qimg = None
                                qimg = None
                                # Try ImageQt first
                                try:
                                    print(f"[ui_node_item] _open_preview: trying ImageQt conversion")
                                    qimg = ImageQt(upstream_img)
                                    from PIL.ImageQt import ImageQt as _ImageQtType
                                    if isinstance(qimg, _ImageQtType):
                                        qimg = None
                                except Exception as e:
                                    print(f"[ui_node_item] _open_preview: ImageQt failed: {e}")
                                    qimg = None

                                # Fallback via PNG bytes
                                if qimg is None:
                                    try:
                                        buf = io.BytesIO()
                                        upstream_img.save(buf, format='PNG')
                                        data = buf.getvalue()
                                        qimg = QImage.fromData(data)
                                    except Exception as e:
                                        print(f"[ui_node_item] _open_preview: PNG-bytes fallback failed: {e}")
                                        qimg = None

                                # Last resort: raw bytes
                                if qimg is None:
                                    try:
                                        if hasattr(upstream_img, 'mode') and upstream_img.mode == 'RGB':
                                            data = upstream_img.tobytes('raw', 'RGB')
                                            qimg = QImage(data, upstream_img.width, upstream_img.height, QImage.Format_RGB888)
                                        else:
                                            img2 = upstream_img.convert('RGBA')
                                            data = img2.tobytes('raw', 'RGBA')
                                            try:
                                                qimg = QImage(data, img2.width, img2.height, QImage.Format_RGBA8888)
                                            except Exception:
                                                qimg = QImage(data, img2.width, img2.height, QImage.Format_ARGB32)
                                    except Exception as e:
                                        print(f"[ui_node_item] _open_preview: raw-bytes fallback failed: {e}")
                                        qimg = None

                                if qimg is None:
                                    return

                                try:
                                    pix = QPixmap.fromImage(qimg)
                                except TypeError as e:
                                    print(f"[ui_node_item] _open_preview: QPixmap.fromImage rejected type {type(qimg)}: {e}, falling back to PNG bytes")
                                    try:
                                        buf = io.BytesIO()
                                        upstream_img.save(buf, format='PNG')
                                        data = buf.getvalue()
                                        qimg2 = QImage.fromData(data)
                                        pix = QPixmap.fromImage(qimg2)
                                    except Exception as e2:
                                        print(f"[ui_node_item] _open_preview: PNG-bytes fallback failed: {e2}")
                                        return
                                # create and show preview window
                                title = f"Preview - {type(self.node).__name__}"
                                win = ImagePreviewWindow(pix, title=title)
                                self._preview_win = win
                                win.show()
                            except Exception:
                                return
                        except Exception:
                            pass

                    try:
                        self._open_preview = _open_preview
                    except Exception:
                        pass
                    # ensure node background is tall enough to fit preview, status and buttons
                    try:
                        min_needed = btn_y + 28 + self.MARGIN_BOTTOM
                        if getattr(self, '_height', self.HEIGHT) < min_needed:
                            self._height = min_needed
                            self.setRect(0, 0, self.WIDTH, self._height)
                    except Exception:
                        pass
            except Exception:
                pass

            # If this is a RenderToFileNode, add a RUN button to trigger saving
            try:
                if isinstance(self.node, RenderToFileNode):
                    run_btn = QPushButton('RUN')
                    try:
                        run_btn.setFixedWidth(60)
                    except Exception:
                        pass

                    run_proxy = QGraphicsProxyWidget(self)
                    run_proxy.setWidget(run_btn)
                    # place button near bottom-right inside the node
                    try:
                        btn_x = int(self.WIDTH - run_btn.width() - 8)
                    except Exception:
                        btn_x = int(self.WIDTH - 68)
                    try:
                        btn_y = int(getattr(self, '_height', self.HEIGHT) - 30)
                    except Exception:
                        btn_y = int(self.HEIGHT - 30)
                    run_proxy.setPos(btn_x, btn_y)
                    run_proxy.setZValue(2)

                    # status text above the button
                    try:
                        self._run_status_text = QGraphicsTextItem('', self)
                        self._run_status_text.setDefaultTextColor(QColor(200, 200, 200))
                        self._run_status_text.setPos(6, btn_y - 18)
                    except Exception:
                        self._run_status_text = None

                    def on_run():
                        try:
                            # compute upstream dependencies for all connected inputs
                            visited = set()

                            def compute_upstream(out_socket):
                                node = out_socket.node
                                if node in visited:
                                    return
                                visited.add(node)
                                for ins in node.inputs.values():
                                    if ins.connection is not None:
                                        compute_upstream(ins.connection.output_socket)
                                try:
                                    node.compute()
                                    for o in node.outputs.values():
                                        try:
                                            o._dirty = False
                                        except Exception:
                                            pass
                                except Exception:
                                    import traceback
                                    traceback.print_exc()

                            def worker_func():
                                # compute upstream for all connected inputs
                                for ins in self.node.inputs.values():
                                    try:
                                        if getattr(ins, 'connection', None) is not None:
                                            compute_upstream(ins.connection.output_socket)
                                    except Exception:
                                        pass
                                # compute this node (saving to file)
                                try:
                                    self.node.compute()
                                except Exception:
                                    import traceback
                                    traceback.print_exc()
                                    raise
                                # attempt to return filepath for status
                                fp = None
                                try:
                                    fp_ins = self.node.inputs.get('filepath')
                                    if fp_ins is not None:
                                        if getattr(fp_ins, 'connection', None) is not None:
                                            fp_out = fp_ins.connection.output_socket
                                            fp = getattr(fp_out, '_cache', None)
                                        else:
                                            fp = getattr(fp_ins, '_cache', None)
                                except Exception:
                                    fp = None
                                return fp

                            try:
                                if getattr(self, '_run_status_text', None) is not None:
                                    self._run_status_text.setPlainText('running...')
                            except Exception:
                                pass

                            worker = ComputeWorker(worker_func)

                            def _on_done(res):
                                try:
                                    if res.get('success'):
                                        fp = res.get('result')
                                        msg = f"Saved: {fp}" if fp else "Run complete"
                                        try:
                                            if getattr(self, '_run_status_text', None) is not None:
                                                self._run_status_text.setPlainText(msg)
                                        except Exception:
                                            pass
                                    else:
                                        print(f"[ui_node_item] Render RUN worker failed:\n{res.get('exc')}")
                                        try:
                                            if getattr(self, '_run_status_text', None) is not None:
                                                self._run_status_text.setPlainText('Run failed (see console)')
                                        except Exception:
                                            pass
                                except Exception:
                                    pass

                            try:
                                # keep reference so worker isn't GC'd
                                self._active_workers.append(worker)
                                def _cleanup_run(res):
                                    try:
                                        try:
                                            self._active_workers.remove(worker)
                                        except Exception:
                                            pass
                                        _on_done(res)
                                    except Exception:
                                        pass

                                worker.finished_sig.connect(_cleanup_run)
                                worker.start()
                            except Exception:
                                # fallback synchronous run
                                try:
                                    fp = worker_func()
                                    msg = f"Saved: {fp}" if fp else "Run complete"
                                    if getattr(self, '_run_status_text', None) is not None:
                                        self._run_status_text.setPlainText(msg)
                                except Exception:
                                    import traceback
                                    traceback.print_exc()
                                    try:
                                        if getattr(self, '_run_status_text', None) is not None:
                                            self._run_status_text.setPlainText('Run failed (see console)')
                                    except Exception:
                                        pass
                        except Exception:
                            pass

                    try:
                        run_btn.clicked.connect(on_run)
                    except Exception:
                        pass
            except Exception:
                pass

            # Drag helpers
            self._drag_offset = None
        
        def boundingRect(self) -> QRectF:
                # Required for QGraphicsItem subclasses
                w = getattr(self, 'WIDTH', getattr(type(self), 'WIDTH', 160))
                h = getattr(self, '_height', getattr(type(self), 'HEIGHT', 100))
                return QRectF(0, 0, w, h)

        def paint(self, painter: QPainter, option, widget=None):
                # Draw normal background + border using this item's brush/pen
                w = getattr(self, 'WIDTH', getattr(type(self), 'WIDTH', 160))
                h = getattr(self, '_height', getattr(type(self), 'HEIGHT', 100))
                rect = QRectF(0, 0, w, h)
                try:
                    painter.setBrush(self.brush())
                    painter.setPen(self.pen())
                except Exception:
                    pass
                painter.drawRect(rect)

            # Do not draw any default selection outline here — selection outline removed


    def _create_sockets(self):
        y = self.MARGIN_TOP
        # inputs
        for name, sock in self.node.inputs.items():
            item = SocketItem(sock, self, is_output=False)
            item.setPos(0, y)
            self.input_items[name] = item
            # label next to input socket (to the right)
            label = QGraphicsTextItem(name, self)
            label.setDefaultTextColor(QColor(200, 200, 200))
            label.setFlag(QGraphicsItem.ItemIsSelectable, False)
            label.setFlag(QGraphicsItem.ItemIsFocusable, False)
            label.setAcceptedMouseButtons(Qt.NoButton)
            label.setAcceptHoverEvents(False)
            # position label a bit to the right of the socket
            padding = 6
            text_w = label.boundingRect().width()
            label.setPos(item.pos().x() + (RADIUS if 'RADIUS' in globals() else 6) + padding, y - label.boundingRect().height()/2)
            # store label reference if needed later
            item._label_item = label
            y += self.SOCKET_SPACING

        y = self.MARGIN_TOP
        # outputs
        for name, sock in self.node.outputs.items():
            item = SocketItem(sock, self, is_output=True)
            item.setPos(self.WIDTH, y)
            self.output_items[name] = item
            # label for output socket (to the left)
            label = QGraphicsTextItem(name, self)
            label.setDefaultTextColor(QColor(200, 200, 200))
            label.setFlag(QGraphicsItem.ItemIsSelectable, False)
            label.setFlag(QGraphicsItem.ItemIsFocusable, False)
            label.setAcceptedMouseButtons(Qt.NoButton)
            label.setAcceptHoverEvents(False)
            padding = 6
            text_w = label.boundingRect().width()
            # align label right-to-left so it sits left of the socket
            label.setPos(item.pos().x() - text_w - (RADIUS if 'RADIUS' in globals() else 6) - padding, y - label.boundingRect().height()/2)
            item._label_item = label
            # If the output socket is modifiable, add an inline editor to the left of the label
            try:
                if getattr(sock, 'is_modifiable', False):
                    widget = None
                    w = 60
                    if sock.socket_type == SocketType.INT:
                        widget = QSpinBox()
                        widget.setRange(-1000000000, 1000000000)
                        w = 60
                        # initialize from node attribute if present
                        try:
                            val = getattr(self.node, name)
                            if isinstance(val, int):
                                widget.setValue(val)
                        except Exception:
                            pass
                        def _on_int(v, s=sock, n=name):
                            try:
                                setattr(self.node, n, int(v))
                            except Exception:
                                pass
                            try:
                                s._cache = int(v)
                                s._dirty = False
                            except Exception:
                                pass
                            try:
                                for dep in self.node.dependents:
                                    dep.mark_dirty()
                            except Exception:
                                pass
                        widget.valueChanged.connect(_on_int)
                    elif sock.socket_type == SocketType.FLOAT:
                        widget = QDoubleSpinBox()
                        widget.setRange(-1e12, 1e12)
                        widget.setDecimals(4)
                        w = 80
                        try:
                            val = getattr(self.node, name)
                            if isinstance(val, float):
                                widget.setValue(val)
                        except Exception:
                            pass
                        def _on_float(v, s=sock, n=name):
                            try:
                                setattr(self.node, n, float(v))
                            except Exception:
                                pass
                            try:
                                s._cache = float(v)
                                s._dirty = False
                            except Exception:
                                pass
                            try:
                                for dep in self.node.dependents:
                                    dep.mark_dirty()
                            except Exception:
                                pass
                        widget.valueChanged.connect(_on_float)
                    elif sock.socket_type == SocketType.STRING:
                        widget = QLineEdit()
                        w = 120
                        try:
                            val = getattr(self.node, name)
                            if isinstance(val, str):
                                widget.setText(val)
                        except Exception:
                            pass
                        def _on_str(v, s=sock, n=name):
                            try:
                                setattr(self.node, n, v)
                            except Exception:
                                pass
                            try:
                                s._cache = v
                                s._dirty = False
                            except Exception:
                                pass
                            try:
                                for dep in self.node.dependents:
                                    dep.mark_dirty()
                            except Exception:
                                pass
                        widget.textChanged.connect(_on_str)
                    elif sock.socket_type == SocketType.BOOLEAN:
                        # Add a modern toggle switch for boolean sockets.
                        try:
                            ToggleItemCls = getattr(type(self), 'ToggleSwitchItem', None)
                            # Prefer the QWidget-based ToggleSwitch (embedded via
                            # QGraphicsProxyWidget) which paints reliably. If not
                            # available, fall back to the lightweight
                            # ToggleSwitchItem graphics object.
                            ToggleWidgetCls = getattr(type(self), 'ToggleSwitch', None)
                            ToggleItemCls = getattr(type(self), 'ToggleSwitchItem', None)
                            widget = None
                            def _on_bool(v, s=sock, n=name):
                                try:
                                    setattr(self.node, n, bool(v))
                                except Exception:
                                    pass
                                try:
                                    s._cache = bool(v)
                                    s._dirty = False
                                except Exception:
                                    pass
                                try:
                                    for dep in self.node.dependents:
                                        dep.mark_dirty()
                                except Exception:
                                    pass
                            try:
                                # Try QWidget toggle first
                                if ToggleWidgetCls is not None:
                                    wdg = ToggleWidgetCls()
                                    try:
                                        wdg.setFixedSize(QSize(48, 24))
                                        wdg._w = 48
                                        wdg._h = 24
                                    except Exception:
                                        pass
                                    widget = wdg
                                    # init state
                                    try:
                                        val = getattr(self.node, name)
                                    except Exception:
                                        val = getattr(sock, '_cache', None)
                                    try:
                                        widget.setChecked(bool(val))
                                    except Exception:
                                        pass

                                    # connect signal
                                    try:
                                        widget.toggled.connect(_on_bool)
                                    except Exception:
                                        pass
                                elif ToggleItemCls is not None:
                                    # Graphics-item fallback
                                    widget = ToggleItemCls(parent=self)
                                    try:
                                        val = getattr(self.node, name)
                                    except Exception:
                                        val = getattr(sock, '_cache', None)
                                    try:
                                        widget.setChecked(bool(val))
                                    except Exception:
                                        pass
                                    try:
                                        widget.setCallback(_on_bool)
                                    except Exception:
                                        try:
                                            widget._callback = _on_bool
                                        except Exception:
                                            pass
                                    try:
                                        widget.node = self.node
                                    except Exception:
                                        pass
                            except Exception:
                                widget = None
                        except Exception:
                            widget = None

                    if widget is not None:
                        # cap widths to avoid overflowing the node box
                        max_w = int(self.WIDTH * 0.45)
                        w = min(w, max_w)
                        # for ToggleSwitch widget keep its own fixed size; otherwise set width
                        try:
                            ToggleCls = getattr(type(self), 'ToggleSwitch', None)
                            if ToggleCls is not None and isinstance(widget, ToggleCls):
                                # ensure toggle keeps its intended size
                                try:
                                    widget.setFixedSize(QSize(getattr(widget, '_w', w), getattr(widget, '_h', 24)))
                                except Exception:
                                    pass
                            else:
                                widget.setFixedWidth(w)
                        except Exception:
                            pass
                        # store a pseudo-width used for layout calculations for non-toggle widgets
                        try:
                            ToggleCls = getattr(type(self), 'ToggleSwitch', None)
                            if not (ToggleCls is not None and isinstance(widget, ToggleCls)):
                                widget._w = w
                        except Exception:
                            pass

                        # Position widget: center horizontally on the node for booleans,
                        # otherwise align left of the label.
                        if sock.socket_type == SocketType.BOOLEAN:
                            # Position the graphics ToggleSwitchItem directly
                            try:
                                if widget is not None and isinstance(widget, self.ToggleSwitchItem):
                                    ww = getattr(widget, '_w', 48)
                                    hh = getattr(widget, '_h', 24)
                                    x_widget = int((self.WIDTH - ww) / 2)
                                    y_widget = int(y - 10 + 6)
                                    try:
                                        widget.setPos(x_widget, y_widget)
                                        widget.setZValue(2)
                                        item._modifiable_widget = widget
                                    except Exception:
                                        pass
                                else:
                                    # fallback: treat as proxyed widget
                                    try:
                                        widget.node = self.node
                                    except Exception:
                                        pass
                                    x_widget = int((self.WIDTH - getattr(widget, '_w', w)) / 2)
                                    y_widget = int(y - 10 + 6)
                                    try:
                                        proxy = QGraphicsProxyWidget(self)
                                        proxy.setWidget(widget)
                                        proxy.resize(getattr(widget, '_w', w), getattr(widget, '_h', 24))
                                        proxy.setAcceptedMouseButtons(Qt.LeftButton)
                                        proxy.setPos(x_widget, y_widget)
                                        proxy.setZValue(2)
                                        item._modifiable_widget = proxy
                                        item._modifiable_widget_widget = widget
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                        else:
                            x_widget = int(label.pos().x() - getattr(widget, '_w', w) - 8)
                            if x_widget < 6:
                                x_widget = 6
                            y_widget = int(y - 10 + 6)
                            try:
                                # Embed QWidget editors using a QGraphicsProxyWidget so they render
                                proxy = QGraphicsProxyWidget(self)
                                proxy.setWidget(widget)
                                # Do not let the editor proxy steal mouse presses — allow node dragging
                                try:
                                    proxy.setAcceptedMouseButtons(Qt.NoButton)
                                except Exception:
                                    pass
                                proxy.setPos(x_widget, y_widget)
                                proxy.setZValue(2)
                                # store reference to proxy (and widget if needed)
                                item._modifiable_widget = proxy
                                item._modifiable_widget_widget = widget
                                # also expose node on the widget for potential callbacks
                                try:
                                    widget.node = self.node
                                except Exception:
                                    pass
                            except Exception:
                                pass
            except Exception:
                pass
            y += self.SOCKET_SPACING

    # Debugging helpers: log mouse events to ensure events arrive
    def mousePressEvent(self, event):
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        try:
            # If double-click occurred over the preview pixmap, open preview
            try:
                sp = event.scenePos()
                pix = getattr(self, '_pix_item', None)
                if pix is not None:
                    try:
                        if pix.sceneBoundingRect().contains(sp):
                            print(f"[ui_node_item] NodeItem detected double-click on preview for node id={id(self.node)}")
                            try:
                                if hasattr(self, '_open_preview'):
                                    self._open_preview()
                            except Exception:
                                pass
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            pass
        super().mouseDoubleClickEvent(event)

    def itemChange(self, change, value):
        return super().itemChange(change, value)

    # Dragging is handled by QGraphicsItem default behavior (ItemIsMovable flag).
    # Custom drag handlers were removed to allow built-in movement and proper
    # interaction with child socket items.

# Ensure `NodeItem.paint` is available at the class level for PyQt's C++ wrapper.
# Some PyQt builds require the method to be present on the class object when
# the C++ type is instantiated; assign an explicit function to be safe.
def _nodeitem_paint(self, painter, option, widget=None):
    try:
        w = getattr(self, 'WIDTH', getattr(type(self), 'WIDTH', 160))
        h = getattr(self, '_height', getattr(type(self), 'HEIGHT', 100))
        rect = QRectF(0, 0, w, h)
        try:
            painter.setBrush(self.brush())
            painter.setPen(self.pen())
        except Exception:
            pass
        painter.drawRect(rect)
    except Exception:
        pass

    try:
        NodeItem.paint = _nodeitem_paint
    except Exception:
        pass
