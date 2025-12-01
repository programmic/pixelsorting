from __future__ import annotations
from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout, QPushButton, QHBoxLayout, QMessageBox, QApplication
from PySide6.QtCore import Qt, QPoint, QSize
from PySide6.QtGui import QPixmap, QImage
from PIL.ImageQt import ImageQt

class ImageEyeDropperDialog(QDialog):
    """Simple dialog that shows an image and lets the user click to pick a color.

    Usage:
        dlg = ImageEyeDropperDialog(pil_image)
        if dlg.exec():
            hexcolor = dlg.selected_hex
    """
    def __init__(self, pil_image, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pick Color From Image")
        self.selected_hex = None
        self.pil_image = pil_image
        self.img_qimage = ImageQt(pil_image.convert('RGBA'))
        # Keep original QImage/QPixmap for accurate coordinate mapping
        self.orig_qimage = QImage(self.img_qimage)
        self.orig_qpix = QPixmap.fromImage(self.orig_qimage)

        # Determine maximum display size: clamp to parent edit window if provided
        max_w = 800
        max_h = 600
        try:
            # Prefer the top-level window size (so passing a small settings widget
            # won't force the image to be tiny). Fall back to screen available
            # geometry if we can't determine a sensible parent size.
            if parent is not None:
                top = parent.window() if hasattr(parent, 'window') else parent
                geom = None
                try:
                    geom = top.geometry()
                except Exception:
                    try:
                        geom = top.frameGeometry()
                    except Exception:
                        geom = None
                if geom is not None and geom.width() > 200 and geom.height() > 200:
                    max_w = max(200, geom.width() - 200)
                    max_h = max(200, geom.height() - 200)
                else:
                    screen = QApplication.primaryScreen()
                    if screen:
                        geom = screen.availableGeometry()
                        max_w = max(300, geom.width() - 200)
                        max_h = max(300, geom.height() - 200)
            else:
                screen = QApplication.primaryScreen()
                if screen:
                    geom = screen.availableGeometry()
                    max_w = max(300, geom.width() - 200)
                    max_h = max(300, geom.height() - 200)
        except Exception:
            # fallback defaults above
            pass

        # Do not upscale tiny images — clamp target to orig size and max size
        orig_w = self.orig_qpix.width()
        orig_h = self.orig_qpix.height()
        target_w = min(orig_w, max_w)
        target_h = min(orig_h, max_h)
        self.qpix = self.orig_qpix.scaled(QSize(target_w, target_h), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.layout = QVBoxLayout(self)

        # Live preview of currently hovered/selected color
        self.color_preview = QLabel()
        self.color_preview.setFixedHeight(30)
        self.color_preview.setStyleSheet("border: 1px solid #111; background: #ffffff;")
        self.layout.addWidget(self.color_preview)

        self.lbl = QLabel()
        self.lbl.setAlignment(Qt.AlignCenter)
        self.lbl.setPixmap(self.qpix)
        # Enable mouse tracking so we receive mouse move events without pressing
        self.lbl.setMouseTracking(True)
        self.setMouseTracking(True)
        self.layout.addWidget(self.lbl)

        btn_row = QHBoxLayout()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(cancel)
        self.layout.addLayout(btn_row)

        # Use release instead of press; attach handlers to the label
        self.lbl.mouseMoveEvent = self._on_move
        self.lbl.mouseReleaseEvent = self._on_release

    def _on_move(self, ev):
        # Update live preview as mouse moves over the image
        pos = ev.pos()
        pixmap = self.lbl.pixmap()
        if pixmap is None:
            return
        lbl_w = self.lbl.width()
        lbl_h = self.lbl.height()
        disp_w = pixmap.width()
        disp_h = pixmap.height()
        # center offsets of the displayed pixmap inside the label
        offset_x = (lbl_w - disp_w) // 2
        offset_y = (lbl_h - disp_h) // 2
        # position within displayed pixmap
        x_disp = pos.x() - offset_x
        y_disp = pos.y() - offset_y
        if x_disp < 0 or y_disp < 0 or x_disp >= disp_w or y_disp >= disp_h:
            return
        # Map displayed coordinates back to original image coordinates
        orig_w = self.orig_qpix.width()
        orig_h = self.orig_qpix.height()
        x_orig = int(x_disp * (orig_w / disp_w))
        y_orig = int(y_disp * (orig_h / disp_h))
        if x_orig < 0 or y_orig < 0 or x_orig >= orig_w or y_orig >= orig_h:
            return
        color = self.orig_qimage.pixelColor(x_orig, y_orig)
        hexc = color.name()
        # update preview box
        self.color_preview.setStyleSheet(f"border: 1px solid #111; background: {hexc};")
        # store current hover color but do not accept yet
        self._hover_hex = hexc

    def _on_release(self, ev):
        # On mouse release select the current hovered color (if any)
        try:
            if hasattr(self, '_hover_hex') and self._hover_hex:
                self.selected_hex = self._hover_hex
                self.accept()
        except Exception:
            pass

    @staticmethod
    def pick_from_pil(pil_image, parent=None):
        dlg = ImageEyeDropperDialog(pil_image, parent)
        if dlg.exec():
            return dlg.selected_hex
        return None
