import math
from PyQt5.QtWidgets import (
    QDialog, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsItem,
    QVBoxLayout, QApplication
)
from PyQt5.QtCore import QTimer, Qt, QPoint, QRect
from PyQt5.QtGui import QPixmap

try:
    from PIL.ImageQt import ImageQt
except Exception:
    ImageQt = None


class PanGraphicsView(QGraphicsView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._panning = False
        self._pan_start = None
        self._is_programmatic_scroll = False  # KORREKTUR: Flag zur Unterscheidung von User- vs System-Scrolls
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.NoDrag)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning and self._pan_start is not None:
            delta = event.pos() - self._pan_start
            hbar = self.horizontalScrollBar()
            vbar = self.verticalScrollBar()
            
            hbar.setValue(hbar.value() - int(delta.x()))
            vbar.setValue(vbar.value() - int(delta.y()))
            
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

    def wheelEvent(self, event):
        try:
            delta = event.angleDelta().y()
            steps = delta / 120.0 if delta else 0
            factor = 1.15 ** steps
            self.scale(factor, factor)
            event.accept()
            return
        except Exception:
            pass
        super().wheelEvent(event)

    def scrollContentsBy(self, dx, dy):
        # KORREKTUR: Wenn das System zentriert, erlauben wir die Bewegung bedingungslos!
        if self._is_programmatic_scroll:
            super().scrollContentsBy(dx, dy)
            return

        scene = self.scene()
        items = scene.items() if scene else []
        pix_item = next((item for item in items if isinstance(item, QGraphicsPixmapItem)), None)

        if pix_item:
            item_rect_in_view = self.mapFromScene(pix_item.sceneBoundingRect()).boundingRect()
            viewport_rect = self.viewport().rect()

            future_rect = item_rect_in_view.translated(dx, dy)
            if not viewport_rect.intersects(future_rect):
                return  # Blockiere das Rausrutschen aus dem Bild beim manuellen Panning

        super().scrollContentsBy(dx, dy)


class FullscreenViewer(QDialog):
    """A fullscreen-capable image viewer with infinite canvas behaviour."""
    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)
        self.setWindowTitle('Fullscreen Viewer')
        self._scene = QGraphicsScene(self)
        self._scene.setSceneRect(-100000, -100000, 200000, 200000)

        self._view = PanGraphicsView(self._scene, self)
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)

        self._pix_item = None
        self._is_fullscreen = False

        screen = QApplication.primaryScreen()
        if screen is not None:
            geom = screen.availableGeometry()
            half_w = max(200, geom.width() // 2)
            half_h = max(200, geom.height() // 2)
            x = geom.x() + (geom.width() - half_w) // 2
            y = geom.y() + (geom.height() - half_h) // 2
            self._default_geometry = QRect(x, y, half_w, half_h)
            self.setGeometry(self._default_geometry)
        else:
            self._default_geometry = None

    def show_image(self, img):
        pix = None
        if isinstance(img, QPixmap):
            pix = img
        else:
            if ImageQt is not None:
                try:
                    qim = ImageQt(img)
                    pix = QPixmap.fromImage(qim)
                except Exception:
                    pass

        if pix is None:
            try:
                pix = QPixmap(img)
            except Exception:
                raise ValueError('Unsupported image type for FullscreenViewer.show_image')

        if self._pix_item is not None:
            self._scene.removeItem(self._pix_item)
            self._pix_item = None

        self._pix_item = QGraphicsPixmapItem(pix)
        self._pix_item.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self._scene.addItem(self._pix_item)
        self._pix_item.setPos(0, 0)

        QTimer.singleShot(50, self.center_and_fit_image)

    def center_and_fit_image(self):
        """Passt das Bild optimal in den aktuellen Rahmen ein und zentriert es."""
        if self._pix_item is not None:
            # KORREKTUR: Flag aktivieren, um die Sperre in scrollContentsBy temporär zu umgehen
            self._view._is_programmatic_scroll = True
            try:
                self._view.resetTransform()
                self._view.fitInView(self._pix_item, Qt.KeepAspectRatio)
                self._view.centerOn(self._pix_item)
            finally:
                # Flag nach der Berechnung wieder sicher ausschalten
                self._view._is_programmatic_scroll = False

    def show_fullscreen(self):
        self._is_fullscreen = True
        self.showFullScreen()
        self.raise_()
        self.activateWindow()
        QTimer.singleShot(50, self.center_and_fit_image)

    def exit_fullscreen(self):
        if self._is_fullscreen:
            self._is_fullscreen = False
            self.showNormal()
            if getattr(self, '_default_geometry', None) is not None:
                self.setGeometry(self._default_geometry)
            QTimer.singleShot(50, self.center_and_fit_image)

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Home:
            self.center_and_fit_image()
            if self._is_fullscreen:
                self.exit_fullscreen()
            ev.accept()
            return
        if ev.key() == Qt.Key_F11:
            if not self._is_fullscreen:
                self.show_fullscreen()
            else:
                self.exit_fullscreen()
            ev.accept()
            return
        super().keyPressEvent(ev)

    def toggle_scrollbars(self, visible: bool):
        policy = Qt.ScrollBarAlwaysOn if visible else Qt.ScrollBarAlwaysOff
        self._view.setHorizontalScrollBarPolicy(policy)
        self._view.setVerticalScrollBarPolicy(policy)

if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtGui import QPixmap
    app = QApplication(sys.argv)
    dlg = FullscreenViewer()
    pix = QPixmap(800, 600)
    pix.fill(Qt.darkGray)
    dlg.show_image(pix)
    dlg.show()
    sys.exit(app.exec_())
