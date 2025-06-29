# masterGUI.py
import sys
from PySide6.QtWidgets import *
from PySide6.QtCore import Qt
from PySide6.QtCore import Signal
from superqt import QDoubleSlider, QToggleSwitch


class SlotTableWidget(QWidget):
    slot_clicked = Signal(str)

    def __init__(self, slots: list[str], parent=None):
        super().__init__(parent)
        self.slots = slots
        self.slot_usage = {}

        self.layout = QHBoxLayout(self)
        self.layout.setSpacing(4)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.buttons = []
        for slot_name in slots:
            btn = QPushButton()
            btn.setFixedSize(20, 20)
            btn.setCheckable(False)
            btn.setFocusPolicy(Qt.NoFocus)
            btn.setToolTip(slot_name)
            self.layout.addWidget(btn)
            self.buttons.append((slot_name, btn))
            btn.clicked.connect(self._make_click_handler(slot_name))

        self.refresh_colors({})

    def _make_click_handler(self, slot_name):
        def handler():
            # Slot0 ist Original, nicht als Output klickbar, aber als Input OK (wird im GUI geprüft)
            self.slot_clicked.emit(slot_name)
        return handler

    def refresh_colors(self, slot_usage: dict[str, bool]):
        self.slot_usage = slot_usage
        for slot_name, btn in self.buttons:
            if slot_name == "slot0":
                btn.setStyleSheet("background-color: #3a75c4; border-radius: 3px;")  # blau für original
                btn.setEnabled(True)  # Jetzt klickbar, aber Auswahl wird später gefiltert
            elif slot_name not in slot_usage:
                btn.setStyleSheet("background-color: lightgray; border-radius: 3px;")
                btn.setEnabled(True)
            else:
                if slot_usage[slot_name]:
                    btn.setStyleSheet("background-color: #4caf50; border-radius: 3px;")  # grün belegt
                else:
                    btn.setStyleSheet("background-color: #e57373; border-radius: 3px;")  # rot leer
                btn.setEnabled(True)

class RenderPassSettingsWidget(QWidget):
    def __init__(self, settings_config: list[dict], parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.controls = {}

        for setting in settings_config:
            w = None
            label_text = setting.get("label", "")
            if label_text:
                label = QLabel(label_text)
                self.layout.addWidget(label)

            t = setting.get("type")

            if t == "radio":
                options = setting.get("options", [])
                w = QWidget()
                h_layout = QHBoxLayout(w)
                button_group = QButtonGroup(self)
                for option in options:
                    rb = QRadioButton(option)
                    h_layout.addWidget(rb)
                    button_group.addButton(rb)
                    if option == setting.get("default"):
                        rb.setChecked(True)
                self.controls[label_text] = button_group
                self.layout.addWidget(w)

            elif t == "switch":
                w = QCheckBox(label_text)
                w.setChecked(setting.get("default", False))
                self.controls[label_text] = w
                self.layout.addWidget(w)

            elif t == "slider":
                w = QSlider(Qt.Horizontal)
                w.setMinimum(setting.get("min", 0))
                w.setMaximum(setting.get("max", 100))
                w.setValue(setting.get("default", 0))
                self.controls[label_text] = w
                self.layout.addWidget(w)

            elif t == "multislider":
                w = QDoubleSlider(Qt.Horizontal)
                w.setMinimum(setting.get("min", 0.0))
                w.setMaximum(setting.get("max", 1.0))
                default = setting.get("default", (0.0, 1.0))
                if isinstance(default, (list, tuple)) and len(default) == 2:
                    w.setValue(*default)
                self.controls[label_text] = w
                self.layout.addWidget(w)

            else:
                w = QLabel(f"Unknown setting type: {t}")
                self.layout.addWidget(w)

    def get_values(self):
        result = {}
        for label, control in self.controls.items():
            if isinstance(control, QButtonGroup):
                checked = control.checkedButton()
                if checked:
                    result[label] = checked.text()
                else:
                    result[label] = None
            elif isinstance(control, QToggleSwitch):
                result[label] = control.isChecked()
            elif isinstance(control, QSlider):
                result[label] = control.value()
            elif isinstance(control, QDoubleSlider):
                result[label] = control.value()
            else:
                result[label] = None
        return result

    def set_values(self, values: dict):
        for label, val in values.items():
            control = self.controls.get(label)
            if control is None:
                continue
            if isinstance(control, QButtonGroup):
                for btn in control.buttons():
                    btn.setChecked(btn.text() == val)
            elif isinstance(control, QToggleSwitch):
                control.setChecked(bool(val))
            elif isinstance(control, QSlider):
                control.setValue(int(val))
            elif isinstance(control, QDoubleSlider):
                if isinstance(val, (list, tuple)) and len(val) == 2:
                    control.setValue(*val)

class RenderPassDefinition:
    def __init__(self, name: str, settings: dict):
        self.name = name
        self.settings = settings

class RenderPassWidget(QWidget):
    def __init__(self, renderpass_type: str, available_slots: list[str], on_select_slot):
        super().__init__()
        self.renderpass_type = renderpass_type
        self.available_slots = available_slots
        self.on_select_slot = on_select_slot
        self.selected_input = None
        self.selected_output = None

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(4, 4, 4, 4)

        # Top mit Drag Bar, Titel, Delete
        top = QHBoxLayout()
        self.drag_bar = QLabel("☰")
        self.drag_bar.setFixedWidth(20)
        self.drag_bar.setAlignment(Qt.AlignCenter)
        top.addWidget(self.drag_bar)

        self.title = QLabel(f"Renderpass: {renderpass_type}")
        self.title.setStyleSheet("font-weight: bold;")
        top.addWidget(self.title)
        top.addStretch()

        self.delete_btn = QPushButton("✖")
        self.delete_btn.setFixedWidth(30)
        self.delete_btn.clicked.connect(self._delete_self)
        top.addWidget(self.delete_btn)
        self.layout.addLayout(top)

        # Input/Output Labels klickbar
        self.io_layout = QHBoxLayout()
        self.input_label = QLabel("Input: <keiner>")
        self.input_label.setStyleSheet("background-color: lightgray; padding: 4px; border-radius: 4px;")
        self.output_label = QLabel("Output: <keiner>")
        self.output_label.setStyleSheet("background-color: lightgray; padding: 4px; border-radius: 4px;")

        self.input_label.mousePressEvent = self._on_input_click
        self.output_label.mousePressEvent = self._on_output_click

        self.io_layout.addWidget(self.input_label)
        self.io_layout.addWidget(self.output_label)
        self.layout.addLayout(self.io_layout)

        # Ersetze bisherigen Settings-Label durch neues Settings Widget
        settings_config = self.get_settings_config(renderpass_type)
        self.settings_widget = RenderPassSettingsWidget(settings_config)
        self.layout.addWidget(self.settings_widget)

        self.selection_mode = None

    def _on_input_click(self, _):
        self.selection_mode = 'input'
        self.input_label.setStyleSheet("background-color: #a0c4ff; padding: 4px; border-radius: 4px;")
        self.output_label.setStyleSheet("background-color: lightgray; padding: 4px; border-radius: 4px;")
        self.on_select_slot('input', self)

    def _on_output_click(self, _):
        self.selection_mode = 'output'
        self.output_label.setStyleSheet("background-color: #a0c4ff; padding: 4px; border-radius: 4px;")
        self.input_label.setStyleSheet("background-color: lightgray; padding: 4px; border-radius: 4px;")
        self.on_select_slot('output', self)

    def set_slot(self, slot_name: str):
        if self.selection_mode == 'input':
            self.selected_input = slot_name
            self.input_label.setText(f"Input: {slot_name}")
        elif self.selection_mode == 'output':
            self.selected_output = slot_name
            self.output_label.setText(f"Output: {slot_name}")

        # Reset Modus & Farben
        self.selection_mode = None
        self.input_label.setStyleSheet("background-color: lightgray; padding: 4px; border-radius: 4px;")
        self.output_label.setStyleSheet("background-color: lightgray; padding: 4px; border-radius: 4px;")

    def get_settings_config(self, renderpass_type):
        # Definiere hier die Settings-Konfigurationen je Renderpass-Typ
        if renderpass_type == "blur":
            return [
                {"label": "Blur Type", "type": "radio", "options": ["Gaussian", "Box", "Median"], "default": "Gaussian"},
                {"label": "Radius", "type": "slider", "min": 1, "max": 20, "default": 5},
                {"label": "Enabled", "type": "switch", "default": True},
            ]
        elif renderpass_type == "tone mapping":
            return [
                {"label": "Method", "type": "radio", "options": ["Reinhard", "ACES", "Filmic"], "default": "Reinhard"},
                {"label": "Exposure", "type": "slider", "min": 0, "max": 10, "default": 1},
                {"label": "Enabled", "type": "switch", "default": True},
            ]
        else:
            # Default, falls Typ unbekannt
            return [
                {"label": "Enabled", "type": "switch", "default": True},
            ]

    def get_settings(self):
        # Nutze das Settings Widget, um Werte auszulesen
        return self.settings_widget.get_values()

    def _delete_self(self):
        self.setParent(None)
        self.deleteLater()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Render Pass Manager")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.renderpasses_container = QVBoxLayout()
        self.layout.addLayout(self.renderpasses_container)

        self.available_slots = [f"slot{i}" for i in range(8)]

        self.add_pass_btn = QPushButton("Add Render Pass")
        self.add_pass_btn.clicked.connect(self.add_renderpass)
        self.layout.addWidget(self.add_pass_btn)

        self.current_selection_mode = None
        self.current_renderpass_widget = None

    def add_renderpass(self):
        # Füge Beispiel-Renderpass vom Typ "blur" hinzu, später Auswahl möglich
        rp = RenderPassWidget("blur", self.available_slots, self.select_slot_for_renderpass)
        self.renderpasses_container.addWidget(rp)

    def select_slot_for_renderpass(self, mode, renderpass_widget):
        self.current_selection_mode = mode
        self.current_renderpass_widget = renderpass_widget
        # Hier könnte ein Slot-Auswahl-Dialog erscheinen, wir nehmen hier einfach den ersten Slot zum Test
        # Um es dynamisch zu machen, kann man QSearchableListWidget oder ähnliches öffnen
        # Für Demo wählen wir nur "slot1"
        # In echt möchtest du evtl. ein Popup mit Liste der slots anzeigen
        slot_name = "slot1"
        renderpass_widget.set_slot(slot_name)
        self.current_selection_mode = None
        self.current_renderpass_widget = None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
