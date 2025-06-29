# masterGUI.py

import sys
from PySide6.QtWidgets import *
from PySide6.QtCore import Qt
from superqt import QSearchableListWidget

from guiElements.slotTableWidgets import *
from guiElements.maskWidget import *
from guiElements.renderPassSettingsWidget import *
from guiElements.renderPassWidget import *

class SearchableReorderableListWidget(QSearchableListWidget):
    """
    A searchable and reorderable list widget.

    :param parent: The parent widget.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.list_widget.setDragEnabled(True)
        self.list_widget.setAcceptDrops(True)
        self.list_widget.setDragDropMode(QListWidget.InternalMove)
        self.list_widget.setDefaultDropAction(Qt.MoveAction)

class GUI(QWidget):
    """
    The main GUI for the render pass application.

    :param parent: The parent widget.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Renderpass GUI with Mask Support")
        self.available_slots = [f"slot{i}" for i in range(16)]
        self.slot_usage = {}
        self.current_selection_mode = None
        self.current_widget = None

        main_layout = QHBoxLayout(self)
        
        # Left + Center (Slots + Renderpass list)
        left_center = QVBoxLayout()
        self.slot_table = SlotTableWidget(self.available_slots)
        left_center.addWidget(self.slot_table)
        
        self.list_widget = SearchableReorderableListWidget()
        left_center.addWidget(self.list_widget, stretch=1)
        main_layout.addLayout(left_center, stretch=1)
        
        # Right panel with pass list
        self.pass_list = QListWidget()
        self.pass_list.addItems(["Blur",
                                "Invert", 
                                "Simple Kuwahara",
                                "PixelSort",
                                "Mix By Percent",
                                "Mix Screen",
                                "Subtract",
                                "Adjust Brightness",
                                "Cristalline Growth"])
        self.pass_list.setFixedWidth(150)
        main_layout.addWidget(self.pass_list)

        self.pass_list.itemClicked.connect(self.on_pass_selected)
        self.slot_table.slot_clicked.connect(self.on_slot_clicked)
        self.update_slot_usage()

    def on_pass_selected(self, item):
        """
        Handle the selection of a render pass from the list.

        :param item: The selected QListWidgetItem.
        """
        renderpass_type = item.text()
        widget = RenderPassWidget(renderpass_type, self.available_slots, self.start_slot_selection, self.remove_render_pass)
        lw = self.list_widget.list_widget
        item = QListWidgetItem()
        item.setSizeHint(widget.sizeHint())
        lw.addItem(item)
        lw.setItemWidget(item, widget)

    def remove_render_pass(self, widget):
        """Remove the render pass widget and its list item from the list."""
        lw = self.list_widget.list_widget
        for i in range(lw.count()):
            item = lw.item(i)
            if lw.itemWidget(item) == widget:
                lw.takeItem(i)
                widget.setParent(None)
                widget.deleteLater()
                break

    def start_slot_selection(self, mode, widget):
        """
        Start the slot selection process.

        :param mode: The mode of selection (input or mask).
        :param widget: The widget that initiated the selection.
        """
        self.current_selection_mode = mode
        self.current_widget = widget

    def on_slot_clicked(self, slot_name):
        """
        Handle the selection of a slot from the slot table.

        :param slot_name: The name of the selected slot.
        """
        if not self.current_widget or not self.current_selection_mode:
            return
            
        # Special check for output on slot0
        if slot_name == "slot0" and self.current_selection_mode in ['output', 'mask']:
            QMessageBox.information(self, "Invalid Selection", 
                                  "Slot 0 cannot be used as output or mask")
            return
            
        self.current_widget.set_slot(slot_name)
        self.update_slot_usage()
        self.current_selection_mode = None
        self.current_widget = None

    def update_slot_usage(self):
        """
        Update the usage status of slots based on the current render passes.
        """
        self.slot_usage = {slot: False for slot in self.available_slots}
        lw = self.list_widget.list_widget
        
        for i in range(lw.count()):
            widget = lw.itemWidget(lw.item(i))
            if widget.selected_output in self.slot_usage:
                self.slot_usage[widget.selected_output] = True
                
            # Get mask slots from settings
            settings = widget.get_settings()
            if settings.get("enabled") and settings.get("slot") in self.slot_usage:
                self.slot_usage[settings.get("slot")] = True

        self.slot_usage["slot0"] = True  # Original is always occupied
        self.slot_table.refresh_colors(self.slot_usage)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GUI()
    window.resize(800, 900)
    window.show()
    sys.exit(app.exec())
