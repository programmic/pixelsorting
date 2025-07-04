# masterGUI

import sys
from PySide6.QtWidgets import *
from PySide6.QtCore import Qt
from superqt import QSearchableListWidget

from guiElements.slotTableWidgets import SlotTableWidget
from guiElements.renderPassWidget import RenderPassWidget
from PIL import Image

import renderHook

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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Renderpass GUI with Mask Support")
        self.available_slots = [f"slot{i}" for i in range(16)]
        self.slot_usage = {}
        self.current_selection_mode = None
        self.current_widget = None

        main_layout = QHBoxLayout(self)
        
        left_center = QVBoxLayout()
        self.slot_table = SlotTableWidget(self.available_slots)
        left_center.addWidget(self.slot_table)
        
        self.list_widget = SearchableReorderableListWidget()
        left_center.addWidget(self.list_widget, stretch=1)

        # Add Run Rendering button
        self.run_button = QPushButton("Run Rendering")
        self.run_button.clicked.connect(self.run_rendering)
        left_center.addWidget(self.run_button)

        main_layout.addLayout(left_center, stretch=1)
        
        right_side_layout = QVBoxLayout()
        self.pass_list = QListWidget()
        self.pass_name_mapping = {
            "Blur": "blur",
            "Invert": "invert",
            "Simple Kuwahara": "kuwaharaGPU",
            "PixelSort": "pixelSort",
            "Mix By Percent": "mixPercent",
            "Mix Screen": "mixScreen",
            "Subtract": "subtract",
            "Adjust Brightness": "adjustBrightness",
            "Cristalline Growth": "cristallineGrowth"
        }
        self.pass_list.addItems(self.pass_name_mapping.keys())
        self.pass_list.setFixedWidth(150)
        right_side_layout.addWidget(self.pass_list)

        self.select_image_button = QPushButton("Select Image")
        # Placeholder for button click event
        self.select_image_button.clicked.connect(self.select_image)
        right_side_layout.addWidget(self.select_image_button)

        main_layout.addLayout(right_side_layout)

        self.pass_list.itemClicked.connect(self.on_pass_selected)
        self.slot_table.slot_clicked.connect(self.on_slot_clicked)
        self.update_slot_usage()

    def on_pass_selected(self, item):
        display_name = item.text()
        internal_name = self.pass_name_mapping.get(display_name, display_name)
        renderpass_type = internal_name
        
        widget = RenderPassWidget(
            renderpass_type,
            self.available_slots,
            self.start_slot_selection,
            self.remove_render_pass
        )
        lw = self.list_widget.list_widget
        list_item = QListWidgetItem()
        list_item.setSizeHint(widget.sizeHint())
        lw.addItem(list_item)
        lw.setItemWidget(list_item, widget)

    def remove_render_pass(self, widget):
        """
        Remove the render pass widget and its list item from the list.
        """
        lw = self.list_widget.list_widget
        for i in range(lw.count()):
            item = lw.item(i)
            if lw.itemWidget(item) == widget:
                # Entferne das Item aus der QListWidget und lösche es sauber
                lw.takeItem(i)
                widget.setParent(None)
                widget.deleteLater()
                item = None
                self.update_slot_usage()
                break

    def start_slot_selection(self, mode, widget):
        self.current_selection_mode = mode
        self.current_widget = widget

    def on_slot_clicked(self, slot_name):
        if not self.current_widget or not self.current_selection_mode:
            return
            
        if slot_name == "slot0" and self.current_selection_mode in ['output', 'mask']:
            QMessageBox.information(self, "Invalid Selection", 
                                    "Slot 0 cannot be used as output or mask")
            return
            
        self.current_widget.set_slot(slot_name)
        self.update_slot_usage()
        self.current_selection_mode = None
        self.current_widget = None

    def update_slot_usage(self):
        self.slot_usage = {slot: False for slot in self.available_slots}
        lw = self.list_widget.list_widget
        
        for i in range(lw.count()):
            widget = lw.itemWidget(lw.item(i))
            if widget.selected_output in self.slot_usage:
                self.slot_usage[widget.selected_output] = True
                
            settings = widget.get_settings()
            if settings.get("enabled") and settings.get("slot") in self.slot_usage:
                self.slot_usage[settings.get("slot")] = True

        self.slot_usage["slot0"] = True  # Original always occupied
        self.slot_table.refresh_colors(self.slot_usage)

    def run_rendering(self):
        try:
            renderHook.run_all_render_passes(self)
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Rendering Error", str(e))

    def select_image(self):
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Select Image")
        file_dialog.setNameFilter("Images (*.png *.jpg *.bmp *.tiff)")
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                image_path = selected_files[0]
                try:
                    img = Image.open(image_path).convert("RGB")
                    self.slot_table.set_image("slot0", img)
                    QMessageBox.information(self, "Image Loaded", f"Image loaded into slot0:\n{image_path}")
                    self.update_slot_usage()
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to load image:\n{str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GUI()
    window.resize(800, 900)
    window.show()
    sys.exit(app.exec())
