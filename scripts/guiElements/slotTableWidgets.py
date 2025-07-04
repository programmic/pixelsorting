# slotTable.py

from PySide6.QtWidgets import *
from PySide6.QtCore import Qt, Signal

class SlotTableWidget(QWidget):
    """
    A widget that displays a table of slots as buttons.

    :param slots: A list of slot names to display.
    :param parent: The parent widget.
    """
    slot_clicked = Signal(str)

    def __init__(self, slots: list[str], parent=None):
        super().__init__(parent)
        self.slots = slots
        self.slot_usage = {}
        self.slot_images = {}  # Dictionary to store images by slot name
        self.toggled_slots = set()  # Set to keep track of toggled slots

        self.layout = QHBoxLayout(self)
        self.layout.setSpacing(4)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.buttons = []
        for slot_name in slots:
            btn = QPushButton()
            btn.setFixedSize(20, 20)
            btn.setCheckable(True)
            btn.setFocusPolicy(Qt.NoFocus)
            btn.setToolTip(slot_name)
            self.layout.addWidget(btn)
            self.buttons.append((slot_name, btn))
            btn.clicked.connect(self._make_click_handler(slot_name))

        self.refresh_colors({})

    def _make_click_handler(self, slot_name):
        def handler():
            self.slot_clicked.emit(slot_name)
        return handler

    def refresh_colors(self, slot_usage: dict[str, bool]):
        """
        Refresh the button colors based on slot usage.

        :param slot_usage: A dictionary indicating the usage of each slot.
        """
        self.slot_usage = slot_usage
        for slot_name, btn in self.buttons:
            if slot_name == "slot0":
                style = "background-color: #3a75c4; border-radius: 3px;"
                if btn.isChecked():
                    style += "border: 2px solid purple;"
                btn.setStyleSheet(style)
                btn.setEnabled(True)
            elif slot_name not in slot_usage:
                style = "background-color: lightgray; border-radius: 3px;"
                if btn.isChecked():
                    style += "border: 2px solid purple;"
                btn.setStyleSheet(style)
                btn.setEnabled(True)
            else:
                if slot_usage[slot_name]:
                    style = "background-color: #4caf50; border-radius: 3px;"
                else:
                    style = "background-color: #e57373; border-radius: 3px;"
                if btn.isChecked():
                    style += "border: 2px solid purple;"
                btn.setStyleSheet(style)
                btn.setEnabled(True)

    def get_image(self, slot_name):
        """
        Get the image stored in the given slot.

        :param slot_name: The name of the slot.
        :return: The image stored in the slot or None if not set.
        """
        return self.slot_images.get(slot_name, None)

    def set_image(self, slot_name, image):
        """
        Set the image for the given slot.

        :param slot_name: The name of the slot.
        :param image: The image to store.
        """
        self.slot_images[slot_name] = image
