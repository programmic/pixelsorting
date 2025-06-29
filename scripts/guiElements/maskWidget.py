# maskWidget.py

from PySide6.QtWidgets import *
from superqt import QToggleSwitch

class MaskWidget(QWidget):
    """
    A widget for selecting a mask slot.

    :param available_slots: A list of available slots.
    :param on_select_slot: Callback function when a slot is selected.
    :param parent: The parent widget.
    """
    def __init__(self, available_slots: list[str], on_select_slot, parent=None):
        super().__init__(parent)
        self.available_slots = available_slots
        self.on_select_slot = on_select_slot
        self.selected_slot = None
        
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.enabled = QToggleSwitch()
        self.slot_label = QLabel("Slot: ")
        self.slot_button = QPushButton("<select>")
        self.slot_button.clicked.connect(self._on_select_clicked)
        
        self.layout.addWidget(QLabel("Use Mask: "))
        self.layout.addWidget(self.enabled)
        self.layout.addWidget(self.slot_label)
        self.layout.addWidget(self.slot_button)
        self.layout.addStretch()
        
    def _on_select_clicked(self):
        self.on_select_slot('mask', self)
        
    def set_slot(self, slot_name: str):
        self.selected_slot = slot_name
        self.slot_button.setText(slot_name)
        
    def get_values(self):
        """
        Get the current values of the mask widget.

        :return: A dictionary containing the enabled state and selected slot.
        """
        return {
            "enabled": self.enabled.isChecked(),
            "slot": self.selected_slot
        }
