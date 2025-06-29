import sys
from PySide6.QtWidgets import *
from PySide6.QtCore import Qt
from PySide6.QtCore import Signal
from PySide6.QtGui import QColor
from superqt import QSearchableListWidget

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

        # Settings Platzhalter (einfach Label hier)
        self.settings_widget = QLabel(f"[Settings für {renderpass_type}]")
        self.settings_widget.setStyleSheet("font-style: italic; color: gray;")
        self.layout.addWidget(self.settings_widget)

        self.selection_mode = None

    def _on_input_click(self, event):
        self.selection_mode = 'input'
        self.input_label.setStyleSheet("background-color: #a0c4ff; padding: 4px; border-radius: 4px;")
        self.output_label.setStyleSheet("background-color: lightgray; padding: 4px; border-radius: 4px;")
        self.on_select_slot('input', self)

    def _on_output_click(self, event):
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

    def _delete_self(self):
        parent = self.parent()
        while parent and not isinstance(parent, QListWidget):
            parent = parent.parent()
        if parent:
            lw = parent
            for i in range(lw.count()):
                if lw.itemWidget(lw.item(i)) is self:
                    lw.takeItem(i)
                    break


    def _build_settings_ui(self, pass_type):
        settings = QWidget()
        layout = QFormLayout(settings)

        if pass_type == "Blur":
            self.radius_slider = QSlider(Qt.Horizontal)
            self.radius_slider.setMinimum(1)
            self.radius_slider.setMaximum(50)
            self.radius_slider.setValue(10)
            layout.addRow("Radius:", self.radius_slider)

        elif pass_type == "Threshold":
            self.threshold_slider = QSlider(Qt.Horizontal)
            self.threshold_slider.setRange(0, 255)
            self.threshold_slider.setValue(128)
            layout.addRow("Threshold:", self.threshold_slider)

        elif pass_type == "ColorOverlay":
            self.color_picker = QPushButton("Farbe wählen")
            self.color_picker.clicked.connect(self.pick_color)
            self.selected_color = QColor(255, 0, 0)
            layout.addRow("Farbe:", self.color_picker)

        else:
            layout.addRow(QLabel("Keine Einstellungen verfügbar"))

        return settings

    def pick_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.selected_color = color
            self.color_picker.setStyleSheet(f"background-color: {color.name()};")
            self.selected_color = color

    def get_settings(self):
        data = {
            "type": self.renderpass_type,
            "input_slot": self.input_slot.currentText(),
            "output_slot": self.output_slot.currentText()
        }
        if self.renderpass_type == "Blur":
            data["radius"] = self.radius_slider.value()
        elif self.renderpass_type == "Threshold":
            data["threshold"] = self.threshold_slider.value()
        elif self.renderpass_type == "ColorOverlay":
            data["color"] = self.selected_color.name()
        return data

    def _delete_self(self):
        list_widget = self.parentWidget().parentWidget().list_widget
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            if list_widget.itemWidget(item) == self:
                list_widget.takeItem(i)
                break

class SearchableReorderableListWidget(QSearchableListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        lw = self.list_widget  # Das interne QListWidget
        lw.setDragEnabled(True)
        lw.setAcceptDrops(True)
        lw.setDragDropMode(QListWidget.InternalMove)
        lw.setDefaultDropAction(Qt.MoveAction)

class CustomElementWidget(QWidget):
    def __init__(self, element_type: str):
        super().__init__()
        self.element_type = element_type
        self.layout = QHBoxLayout(self)

        # Drag-Handle
        #self.drag_bar = QLabel("☰")
        self.drag_bar = QLabel("=")
        self.drag_bar.setFixedWidth(20)
        self.drag_bar.setAlignment(Qt.AlignCenter)
        self.drag_bar.setStyleSheet("color: gray; font-size: 16px;")

        # Dynamisches Element
        if element_type == "Text":
            self.widget = QLineEdit("Ein Text")
        elif element_type == "Button":
            self.widget = QPushButton("Ein Button")
        elif element_type == "Checkbox":
            self.widget = QCheckBox("Eine Checkbox")
        else:
            self.widget = QLineEdit("Unbekannt")

        # Löschen-Button
        self.delete_btn = QPushButton("✖")
        self.delete_btn.setFixedWidth(30)
        self.delete_btn.setStyleSheet("color: red;")

        self.delete_btn.clicked.connect(self._delete_self)

        # Layout-Zusammenbau
        self.layout.addWidget(self.drag_bar)
        self.layout.addWidget(self.widget)
        self.layout.addWidget(self.delete_btn)
        self.layout.setContentsMargins(2, 2, 2, 2)
        self.layout.setSpacing(6)

    def _delete_self(self):
        # Find and remove the corresponding QListWidgetItem from the parent QListWidget
        parent = self.parent()
        while parent and not isinstance(parent, QListWidget):
            parent = parent.parent()
        if parent:
            for i in range(parent.count()):
                if parent.itemWidget(parent.item(i)) is self:
                    parent.takeItem(i)
                    break

class GUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Renderpass GUI mit Slots (kleine Quadrate oben)")

        self.available_slots = [f"slot{i}" for i in range(16)]
        self.slot_usage = {}

        main_layout = QHBoxLayout(self)

        # Links + Mitte als vertikaler Split (Slots oben, Renderpass-Liste unten)
        left_center = QVBoxLayout()

        # Oben: Slot-Visualisierung (klein)
        self.slot_table = SlotTableWidget(self.available_slots)
        left_center.addWidget(self.slot_table)

        # Darunter: Renderpass-Liste
        self.list_widget = SearchableReorderableListWidget()
        left_center.addWidget(self.list_widget, stretch=1)

        main_layout.addLayout(left_center, stretch=1)

        # Rechts: Liste zum schnellen Hinzufügen von Renderpasses
        self.pass_list = QListWidget()
        self.pass_list.addItems(["Blur", "Threshold", "ColorOverlay", "Invert", "Sharpen"])
        self.pass_list.setFixedWidth(150)
        main_layout.addWidget(self.pass_list)

        self.pass_list.itemClicked.connect(self.on_pass_selected)

        # Auswahlmodus für Slots
        self.current_selection_mode = None
        self.current_renderpass_widget = None

        # Slot-Klicks auswerten
        self.slot_table.slot_clicked.connect(self.on_slot_clicked)

        self.update_slot_usage()

    def on_pass_selected(self, item):
        renderpass_type = item.text()
        widget = RenderPassWidget(renderpass_type, self.available_slots, self.start_slot_selection)
        lw = self.list_widget.list_widget
        item = QListWidgetItem()
        item.setSizeHint(widget.sizeHint())
        lw.addItem(item)
        lw.setItemWidget(item, widget)

    def start_slot_selection(self, mode, widget):
        self.current_selection_mode = mode
        self.current_renderpass_widget = widget

    def on_slot_clicked(self, slot_name):
        if not self.current_renderpass_widget or not self.current_selection_mode:
            return

        # Slot0 darf nur als Input gewählt werden, nicht als Output
        if slot_name == "slot0" and self.current_selection_mode == 'output':
            # Ignorieren und evtl. Hinweis setzen (optional)
            QMessageBox.information(self, "Ungültige Auswahl",
                                    "Slot 0 (Original) kann nicht als Output verwendet werden.")
            return

        # Gültige Auswahl
        self.current_renderpass_widget.set_slot(slot_name)
        self.update_slot_usage()
        self.current_selection_mode = None
        self.current_renderpass_widget = None

    def update_slot_usage(self):
        self.slot_usage = {slot: False for slot in self.available_slots}
        lw = self.list_widget.list_widget
        for i in range(lw.count()):
            widget = lw.itemWidget(lw.item(i))
            if widget.selected_output in self.slot_usage:
                self.slot_usage[widget.selected_output] = True

        # Slot0 (original) ist immer belegt
        self.slot_usage["slot0"] = True

        self.slot_table.refresh_colors(self.slot_usage)

        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GUI()
    window.resize(600,800)
    window.show()
    sys.exit(app.exec())