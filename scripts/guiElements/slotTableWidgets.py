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
                btn.setStyleSheet("background-color: #3a75c4; border-radius: 3px;")
                btn.setEnabled(True)
            elif slot_name not in slot_usage:
                btn.setStyleSheet("background-color: lightgray; border-radius: 3px;")
                btn.setEnabled(True)
            else:
                if slot_usage[slot_name]:
                    btn.setStyleSheet("background-color: #4caf50; border-radius: 3px;")
                else:
                    btn.setStyleSheet("background-color: #e57373; border-radius: 3px;")
                btn.setEnabled(True)
