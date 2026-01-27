from PyQt5.QtWidgets import QGraphicsEllipseItem, QGraphicsItem, QToolTip, QLabel
from PyQt5.QtCore import Qt, QVariantAnimation, QPoint, QTimer
from PyQt5.QtGui import QCursor
from PyQt5.QtGui import QColor

from .classes import SocketType, InputSocket

RADIUS = 6

SOCKET_COLORS = {
    SocketType.UNDEFINED: QColor(120, 120, 120),
    SocketType.FLOAT: QColor(40, 65, 185),
    SocketType.INT: QColor(146, 99, 142),
    SocketType.STRING: QColor(172, 78, 31),
    SocketType.BOOLEAN: QColor(254, 77, 57),
    SocketType.COLOR: QColor(255, 255, 255),
    SocketType.PIL_IMG: QColor(84, 37, 108),
    SocketType.PIL_IMG_MONOCH: QColor(24, 10, 31),
    SocketType.ENUM: QColor(150, 150, 175),
}


class SocketItem(QGraphicsEllipseItem):
    def __init__(self, backend_socket, parent, is_output):
        super().__init__(-RADIUS, -RADIUS, RADIUS * 2, RADIUS * 2, parent)

        self.socket = backend_socket
        self.is_output = is_output

        sock_type = backend_socket.socket_type
        color = SOCKET_COLORS.get(sock_type, SOCKET_COLORS[SocketType.UNDEFINED])
        self.setBrush(color)

        # We'll handle hover centrally from the scene (avoid built-in hover events)
        self.setAcceptHoverEvents(False)
        self.setFiltersChildEvents(False)
        # Disable selection and focus for sockets to avoid selection outlines
        self.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self.setFlag(QGraphicsItem.ItemIsFocusable, False)
        # Accept left button for clicks; also accept right button for reroute/disconnect
        self.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)

        # scale around center (ellipse centered at 0,0)
        self.setTransformOriginPoint(0, 0)

        # animation
        self._scale_anim = QVariantAnimation()
        self._scale_anim.setDuration(120)   # ms
        self._scale_anim.valueChanged.connect(self.setScale)

        self._base_scale = 1.0
        self._hover_scale = 1.6
        # Tooltip handling: delayed show so it doesn't flash while moving
        self._tooltip_timer = QTimer()
        self._tooltip_timer.setSingleShot(True)
        self._tooltip_timer.setInterval(350)
        self._tooltip_timer.timeout.connect(self._show_tooltip_immediate)
        self._tooltip_visible = False
        self._tooltip_widget = None
        self._tooltip_using_qtool = False

    def center(self):
        return self.scenePos()

    def mousePressEvent(self, event):
        # Left-click on outputs starts a new connection. Accept the event
        # so the parent NodeItem doesn't begin dragging when the socket is clicked.
        if event.button() == Qt.LeftButton and self.is_output:
            try:
                self.scene().start_connection(self)
            except Exception:
                pass
            event.accept()
            return

        # Right-click on inputs with an existing connection starts a reroute.
        # Accept the event to prevent parent drag.
        if event.button() == Qt.RightButton and not self.is_output:
            try:
                if getattr(self.socket, 'connection', None) is not None:
                    self.scene().start_reroute_from_input(self)
                    event.accept()
                    return
            except Exception:
                pass

        # For any other mouse presses, accept and stop propagation so the
        # socket region blocks node grabs.
        event.accept()
        return

    def _animate_scale(self, target):
        self._scale_anim.stop()
        self._scale_anim.setStartValue(self.scale())
        self._scale_anim.setEndValue(target)
        self._scale_anim.start()

    def _set_scale_immediate(self, target):
        # fallback immediate scale without animation
        self._scale_anim.stop()
        self.setScale(target)

    # New explicit hover methods called by the scene
    def on_hover_enter(self, screen_pos=None):
        self._animate_scale(self._hover_scale)
        self._hovering = True
        # start delayed tooltip; it will not show if hover leaves or a
        # connection is being dragged
        try:
            self._tooltip_timer.start()
        except Exception:
            pass

    def on_hover_leave(self):
        self._animate_scale(self._base_scale)
        self._hovering = False
        try:
            self._tooltip_timer.stop()
        except Exception:
            pass
        # hide custom tooltip widget if shown
        try:
            if self._tooltip_widget is not None and self._tooltip_visible:
                self._tooltip_widget.hide()
                self._tooltip_visible = False
        except Exception:
            pass
        try:
            if self._tooltip_using_qtool:
                QToolTip.hideText()
                self._tooltip_using_qtool = False
        except Exception:
            pass

    def on_hover_move(self, screen_pos=None):
        # keep tooltip following
        if not getattr(self, '_hovering', False):
            # ensure we enter hover state
            self._set_scale_immediate(self._hover_scale)
            self._hovering = True
        # while moving, restart the tooltip delay or update position if visible
        try:
            if self._tooltip_visible and self._tooltip_widget is not None:
                pos = QCursor.pos() + QPoint(10, 10)
                self._tooltip_widget.move(pos)
            else:
                self._tooltip_timer.start()
        except Exception:
            pass

    def hoverEnterEvent(self, event):
        self._animate_scale(self._hover_scale)
        self._hovering = True
        # show tooltip immediately at cursor
        try:
            # start delayed tooltip instead of showing immediately
            self._tooltip_timer.start()
        except Exception:
            pass
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._animate_scale(self._base_scale)
        self._hovering = False
        try:
            self._tooltip_timer.stop()
        except Exception:
            pass
        try:
            if self._tooltip_widget is not None and self._tooltip_visible:
                self._tooltip_widget.hide()
                self._tooltip_visible = False
        except Exception:
            pass
        super().hoverLeaveEvent(event)

    def hoverMoveEvent(self, event):
        # ensure hover scaling applies even if hoverEnter wasn't fired
        if not getattr(self, '_hovering', False):
            # immediate scale to hover state
            self._set_scale_immediate(self._hover_scale)
            self._hovering = True

        # update or restart tooltip handling while moving
        try:
            if self._tooltip_visible and self._tooltip_widget is not None:
                pos = QCursor.pos() + QPoint(10, 10)
                self._tooltip_widget.move(pos)
            else:
                self._tooltip_timer.start()
        except Exception:
            pass
        super().hoverMoveEvent(event)

    def _show_tooltip_immediate(self):
        # Do not show tooltip while user is dragging a connection
        try:
            s = self.scene()
            if s is not None and getattr(s, 'temp_connection', None) is not None:
                return
        except Exception:
            pass

        try:
            sname = getattr(self.socket, 'name', '')
            stype = SocketType(self.socket.socket_type).name if hasattr(self.socket, 'socket_type') else 'UNKNOWN'
            tooltip = f"{sname}: {stype}"
            if isinstance(self.socket, InputSocket) and self.socket.is_optional:
                tooltip += " (optional)"

            # If the backend node provides per-socket tooltips, include them
            try:
                node = getattr(self.socket, 'node', None)
                if node is not None:
                    # look for input or output tooltip dicts
                    tt = None
                    if isinstance(self.socket, InputSocket):
                        tt = getattr(node, 'tooltips_in', None)
                    else:
                        tt = getattr(node, 'tooltips_out', None)
                    if isinstance(tt, dict):
                        extra = tt.get(sname) or tt.get(str(sname))
                        if extra:
                            # append on new line for readability
                            tooltip = tooltip + "\n" + str(extra)
            except Exception:
                pass

            # create a QLabel-based tooltip widget so updates don't flicker
            if self._tooltip_widget is None:
                self._tooltip_widget = QLabel()
                self._tooltip_widget.setWindowFlags(Qt.ToolTip)
                self._tooltip_widget.setStyleSheet("QLabel{background: #333; color: #fff; padding:4px; border-radius:4px}")
            self._tooltip_widget.setText(tooltip)
            pos = QCursor.pos() + QPoint(10, 10)
            self._tooltip_widget.move(pos)
            self._tooltip_widget.adjustSize()
            self._tooltip_widget.show()
            # if the QLabel failed to appear (platform), fallback to QToolTip
            if not self._tooltip_widget.isVisible():
                QToolTip.showText(pos, tooltip)
                self._tooltip_using_qtool = True
                self._tooltip_visible = True
            else:
                self._tooltip_visible = True
                self._tooltip_using_qtool = False
        except Exception:
            pass
