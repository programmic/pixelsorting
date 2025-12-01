from __future__ import annotations
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QSlider, QWidget
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

class HSVColorDialog(QDialog):
    def __init__(self, parent=None, initial_hex="#ffffff"):
        super().__init__(parent)
        self.setWindowTitle("HSV Color Picker")
        self.resize(360, 220)
        self.selected_hex = None

        self.layout = QVBoxLayout(self)

        self.preview = QLabel()
        self.preview.setFixedHeight(50)
        self.layout.addWidget(self.preview)

        # sliders: H (0-359), S (0-100), V (0-100)
        self.sliders = {}
        for name, rng in (('H', (0, 359)), ('S', (0, 100)), ('V', (0, 100))):
            row = QHBoxLayout()
            lbl = QLabel(name)
            lbl.setFixedWidth(20)
            s = QSlider(Qt.Horizontal)
            s.setMinimum(rng[0])
            s.setMaximum(rng[1])
            s.setSingleStep(1)
            row.addWidget(lbl)
            row.addWidget(s)
            self.layout.addLayout(row)
            self.sliders[name] = s

        btn_row = QHBoxLayout()
        ok = QPushButton("OK")
        cancel = QPushButton("Cancel")
        btn_row.addStretch()
        btn_row.addWidget(ok)
        btn_row.addWidget(cancel)
        self.layout.addLayout(btn_row)

        ok.clicked.connect(self._on_ok)
        cancel.clicked.connect(self.reject)

        # set initial color
        try:
            c = QColor(initial_hex)
            h, s, v, _ = c.getHsv()
            # QColor returns s,v in 0-255 scale; convert to 0-100
            self.sliders['H'].setValue(h if h >= 0 else 0)
            self.sliders['S'].setValue(int(s / 255.0 * 100))
            self.sliders['V'].setValue(int(v / 255.0 * 100))
        except Exception:
            self.sliders['H'].setValue(0)
            self.sliders['S'].setValue(0)
            self.sliders['V'].setValue(100)

        for s in self.sliders.values():
            s.valueChanged.connect(self._update_preview)

        self._update_preview()

    def _update_preview(self):
        h = self.sliders['H'].value()
        s = int(self.sliders['S'].value() / 100.0 * 255)
        v = int(self.sliders['V'].value() / 100.0 * 255)
        c = QColor()
        c.setHsv(h, s, v)
        self.preview.setStyleSheet(f"background-color: {c.name()}; border: 1px solid #111;")

    def _on_ok(self):
        h = self.sliders['H'].value()
        s = int(self.sliders['S'].value() / 100.0 * 255)
        v = int(self.sliders['V'].value() / 100.0 * 255)
        c = QColor()
        c.setHsv(h, s, v)
        self.selected_hex = c.name()
        self.accept()

    @staticmethod
    def get_color(parent=None, initial_hex="#ffffff"):
        dlg = HSVColorDialog(parent, initial_hex)
        res = dlg.exec()
        if res:
            return dlg.selected_hex
        return None
