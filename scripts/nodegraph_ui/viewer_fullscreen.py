"""
fullscreen viewer with infinite-canvas panning (MMB) and scroll-to-zoom.

Usage:
    from .viewer_fullscreen import FullscreenViewer
    fv = FullscreenViewer(parent)
    fv.show_image(qpixmap_or_pil_image)
    fv.show_fullscreen()

The image item is movable (left-drag) and may be pushed off-screen freely.
Middle mouse button pans the view. Wheel zooms centered on the cursor.
"""
from PyQt5.QtWidgets import (
    QDialog, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsItem,
    QVBoxLayout, QApplication
)
from PyQt5.QtCore import Qt, QPoint, QRect
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
        # keep mouse-driven zoom anchored at cursor
        try:
            self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        except Exception:
            pass
        # Smooth rendering
        try:
            self.setRenderHints(self.renderHints())
        except Exception:
            pass
        self.setDragMode(QGraphicsView.NoDrag)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            try:
                self.setCursor(Qt.ClosedHandCursor)
            except Exception:
                pass
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning and self._pan_start is not None:
            delta = event.pos() - self._pan_start
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
            try:
                self.setCursor(Qt.ArrowCursor)
            except Exception:
                pass
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        # Default: wheel zooms centered under cursor (no modifier required)
        try:
            delta = 0
            try:
                delta = event.angleDelta().y()
            except Exception:
                try:
                    delta = event.delta()
                except Exception:
                    delta = 0

            # Compute scale factor per notch (120 units)
            steps = delta / 120.0 if delta else 0
            factor = 1.15 ** steps
            # apply scale
            self.scale(factor, factor)
            event.accept()
            return
        except Exception:
            pass
        super().wheelEvent(event)


class FullscreenViewer(QDialog):
    """A fullscreen-capable image viewer with infinite canvas behaviour."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Fullscreen Viewer')
        self._scene = QGraphicsScene(self)
        # create a very large scene rect to make movement effectively unconstrained
        try:
            self._scene.setSceneRect(-100000, -100000, 200000, 200000)
        except Exception:
            pass

        self._view = PanGraphicsView(self._scene, self)
        # show no scrollbars by default to feel like infinite canvas
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)

        self._pix_item = None
        self._is_fullscreen = False

        # compute and apply default half-screen geometry
        try:
            screen = QApplication.primaryScreen()
            if screen is not None:
                geom = screen.availableGeometry()
                half_w = max(200, geom.width() // 2)
                half_h = max(200, geom.height() // 2)
                # center the floating window
                x = geom.x() + (geom.width() - half_w) // 2
                y = geom.y() + (geom.height() - half_h) // 2
                self._default_geometry = QRect(x, y, half_w, half_h)
                self.setGeometry(self._default_geometry)
            else:
                self._default_geometry = None
        except Exception:
            self._default_geometry = None

    def show_image(self, img):
        """Accept a QPixmap, QImage or a PIL.Image. Replace existing image."""
        pix = None
        if isinstance(img, QPixmap):
            pix = img
        else:
            # try PIL
            try:
                if ImageQt is not None:
                    qim = ImageQt(img)
                    pix = QPixmap.fromImage(qim)
            except Exception:
                pass

        if pix is None:
            # last effort: try casting from QPixmap-like
            try:
                pix = QPixmap(img)
            except Exception:
                raise ValueError('Unsupported image type for FullscreenViewer.show_image')

        # remove previous
        if self._pix_item is not None:
            try:
                self._scene.removeItem(self._pix_item)
            except Exception:
                pass
            self._pix_item = None

        self._pix_item = QGraphicsPixmapItem(pix)
        # allow free movement of the pixmap (left-drag) without clamping
        try:
            self._pix_item.setFlag(QGraphicsItem.ItemIsMovable, True)
            self._pix_item.setFlag(QGraphicsItem.ItemIsSelectable, True)
        except Exception:
            # some Qt wrappers require specific QGraphicsItem import; fallback: no flags
            pass

        # add to scene near center
        self._scene.addItem(self._pix_item)
        # place at scene origin (0,0)
        try:
            self._pix_item.setPos(0, 0)
        except Exception:
            pass

        # ensure view fits a comfortable starting scale (not forced)
        try:
            self._view.fitInView(self._pix_item, Qt.KeepAspectRatio)
        except Exception:
            pass

    def show_fullscreen(self):
        try:
            # enter true fullscreen
            self._is_fullscreen = True
            self.showFullScreen()
            self.raise_()
            self.activateWindow()
        except Exception:
            try:
                self.showMaximized()
            except Exception:
                try:
                    self.show()
                except Exception:
                    pass

    def exit_fullscreen(self):
        try:
            if self._is_fullscreen:
                self._is_fullscreen = False
                self.showNormal()
                # restore default geometry if known
                if getattr(self, '_default_geometry', None) is not None:
                    try:
                        self.setGeometry(self._default_geometry)
                    except Exception:
                        pass
        except Exception:
            pass

    def keyPressEvent(self, ev):
        try:
            if ev.key() == Qt.Key_Home:
                # restore default view state and exit fullscreen if necessary
                try:
                    self._view.resetTransform()
                except Exception:
                    pass
                try:
                    if self._pix_item is not None:
                        self._view.fitInView(self._pix_item, Qt.KeepAspectRatio)
                        self._view.centerOn(self._pix_item)
                except Exception:
                    pass
                try:
                    if self._is_fullscreen:
                        self.exit_fullscreen()
                except Exception:
                    pass
                ev.accept()
                return
            if ev.key() == Qt.Key_F11:
                try:
                    if not self._is_fullscreen:
                        self.show_fullscreen()
                    else:
                        self.exit_fullscreen()
                except Exception:
                    pass
                ev.accept()
                return
        except Exception:
            pass
        super().keyPressEvent(ev)
    def toggle_scrollbars(self, visible: bool):
        policy = Qt.ScrollBarAlwaysOn if visible else Qt.ScrollBarAlwaysOff
        self._view.setHorizontalScrollBarPolicy(policy)
        self._view.setVerticalScrollBarPolicy(policy)


if __name__ == '__main__':
    # quick manual test when running file directly
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
