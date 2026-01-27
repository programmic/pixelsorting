# scripts/nodegraph_ui/ui_connection_item.py

from PyQt5.QtWidgets import QGraphicsPathItem
from PyQt5.QtGui import QPainterPath, QPen, QBrush, QLinearGradient
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

from .ui_socket_item import SOCKET_COLORS

class ConnectionItem(QGraphicsPathItem):
    def __init__(self, start_socket):
        super().__init__()
        self.start_socket = start_socket
        # end_socket is the socket item at the destination (if connected).
        # end_pos is used while dragging (temporary) or if there is no end_socket.
        self.end_socket = None
        self.end_pos = start_socket.center()

    def update_end(self, pos):
        # While dragging, update a free-floating end position and clear any
        # end_socket reference so the visual follows the mouse.
        self.end_socket = None
        self.end_pos = pos
        self.update_path()

    def set_end_socket(self, socket_item):
        # Store a reference to the end socket so the connection follows that
        # socket when it moves.
        self.end_socket = socket_item
        self.update_path()

    def update_path(self):
        p1 = self.start_socket.center()
        # Prefer a live end socket position when available
        if self.end_socket is not None:
            p2 = self.end_socket.center()
        else:
            p2 = self.end_pos

        path = QPainterPath(p1)
        dx = (p2.x() - p1.x()) * 0.5

        path.cubicTo(
            p1.x() + dx, p1.y(),
            p2.x() - dx, p2.y(),
            p2.x(), p2.y()
        )

        # Determine colors for gradient: prefer socket colors when available
        try:
            start_color = SOCKET_COLORS.get(getattr(self.start_socket.socket, 'socket_type', None), QColor(200,200,200))
        except Exception:
            start_color = QColor(200,200,200)
        try:
            if self.end_socket is not None:
                end_color = SOCKET_COLORS.get(getattr(self.end_socket.socket, 'socket_type', None), QColor(160,160,160))
            else:
                end_color = QColor(160,160,160)
        except Exception:
            end_color = QColor(160,160,160)

        # Create a linear gradient along the connection
        grad = QLinearGradient(p1.x(), p1.y(), p2.x(), p2.y())
        grad.setColorAt(0.0, start_color)
        grad.setColorAt(1.0, end_color)

        pen = QPen(QBrush(grad), 4.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        self.setPen(pen)
        self.setPath(path)
