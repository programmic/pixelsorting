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
        self.on_select_slot('mask')
        
    def set_slot(self, slot_name: str):
        self.selected_slot = slot_name
        self.slot_button.setText(slot_name)
        
    def set_mask_slot(self, slot_name: str):
        """Alias for set_slot to maintain compatibility with renderPassWidget"""
        self.set_slot(slot_name)
        
    def get_values(self):
        """
        Get the current values of the mask widget.

        :return: A dictionary containing the enabled state and selected slot.
        """
        return {
            "enabled": self.enabled.isChecked(),
            "slot": self.selected_slot
        }

    def load_settings(self, settings_dict):
        """
        Load saved settings into the mask widget.

        :param settings_dict: Dictionary containing saved mask settings
        """
        if not settings_dict:
            return
            
        if 'enabled' in settings_dict:
            self.enabled.setChecked(bool(settings_dict['enabled']))
            
        if 'slot' in settings_dict:
            self.selected_slot = settings_dict['slot']
            self.slot_button.setText(str(self.selected_slot))
