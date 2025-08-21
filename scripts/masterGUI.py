# masterGUI

import sys
import subprocess
import sys
from PySide6.QtWidgets import *
from PySide6.QtCore import Qt, QThread, QTimer
from superqt import QSearchableListWidget
from utils import get_output_dir

from guiElements.modernSlotTableWidget import ModernSlotTableWidget
from guiElements.renderPassWidget import RenderPassWidget
from guiElements.importedImagesWidget import ImportedImagesListWidget
from renderWorker import RenderWorker
from PIL import Image

import renderHook
import json
import os

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
        self.currentWidget = None
        self.original_image_slots = set()
        self.imported_images = {}  # Dictionary to store imported images
        
        # Setup render thread and worker
        self.render_thread = None
        self.render_worker = None

        # Load renderPasses.json to get available passes
        try:
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up one level to project root
            project_root = os.path.dirname(script_dir)
            json_path = os.path.join(project_root, 'renderPasses.json')
            with open(json_path, 'r', encoding='utf-8') as f:
                self.render_passes_config = json.load(f)
                print(f"Successfully loaded {len(self.render_passes_config)} render passes from JSON")
        except Exception as e:
            print(f"Error loading renderPasses.json: {e}")
            self.render_passes_config = {}

        main_layout = QHBoxLayout(self)
        
        # Left side with slot table and status bar
        left_center = QVBoxLayout()
        
        # Slot table
        self.slotTable = ModernSlotTableWidget(self.available_slots)
        left_center.addWidget(self.slotTable)
        
        # Status bar for feedback
        self.status_bar = QLabel()
        self.status_bar.setStyleSheet("color: #666; padding: 5px;")
        left_center.addWidget(self.status_bar)
        
        def show_message(self, message):
            """Display a message in the status bar."""
            self.status_bar.setText(message)
            print(message)  # Also print to console for debugging
        
        self.show_message = show_message.__get__(self)
        
        self.listWidget = SearchableReorderableListWidget()
        left_center.addWidget(self.listWidget, stretch=1)

        # Add progress bar and status label
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(0)  # Indeterminate progress
        self.progress_bar.hide()
        left_center.addWidget(self.progress_bar)
        
        self.status_label = QLabel()
        self.status_label.hide()
        left_center.addWidget(self.status_label)
        
        # Add Run Rendering and Output Folder buttons
        button_layout = QHBoxLayout()
        
        self.run_button = QPushButton("Run Rendering")
        self.run_button.clicked.connect(self.runRendering)
        button_layout.addWidget(self.run_button)
        
        self.open_output_button = QPushButton("Open Output Folder")
        self.open_output_button.clicked.connect(self.openOutputFolder)
        button_layout.addWidget(self.open_output_button)
        
        left_center.addLayout(button_layout)

        # Add Save/Load buttons
        button_layout = QHBoxLayout()
        self.save_button = QPushButton("Save Project")
        self.save_button.clicked.connect(self.saveProject)
        button_layout.addWidget(self.save_button)
        
        self.load_button = QPushButton("Load Project")
        self.load_button.clicked.connect(self.loadProject)
        button_layout.addWidget(self.load_button)
        
        left_center.addLayout(button_layout)

        main_layout.addLayout(left_center, stretch=1)
        
        # Right side with tools
        rightSideLayout = QVBoxLayout()
        rightSideLayout.setSpacing(10)

        # Select Image Section
        self.selectImageButton = QPushButton("Select Image")
        self.selectImageButton.setStyleSheet("""
            QPushButton {
                padding: 8px;
                background: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #45a049;
            }
        """)
        self.selectImageButton.clicked.connect(self.select_image)
        rightSideLayout.addWidget(self.selectImageButton)

        # Imported Images List
        importLabel = QLabel("Imported Images:")
        importLabel.setStyleSheet("font-weight: bold; margin-top: 10px;")
        rightSideLayout.addWidget(importLabel)
        self.imported_images_list = ImportedImagesListWidget()
        self.imported_images_list.setMaximumHeight(150)
        rightSideLayout.addWidget(self.imported_images_list)

        # Pass selection
        passLabel = QLabel("Available render passes:")
        passLabel.setStyleSheet("font-weight: bold; margin-top: 10px;")
        rightSideLayout.addWidget(passLabel)
        self.passList = QListWidget()
        self.passList.setStyleSheet("""
            QListWidget {
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 4px;
            }
            QListWidget::item:hover {
                background: #f0f0f0;
            }
            QListWidget::item:selected {
                background: #e0e0e0;
                color: black;
            }
        """)
        self.pass_display_names = list(self.render_passes_config.keys())
        self.passList.addItems(self.pass_display_names)
        self.passList.setFixedWidth(200)
        rightSideLayout.addWidget(self.passList)

        main_layout.addLayout(rightSideLayout)

        self.passList.itemClicked.connect(self.on_pass_selected)
        self.slotTable.slot_clicked.connect(self.onSlotClicked)
        
        # Connect context menu signals
        self.slotTable.context_menu.slot_learn_requested.connect(self.onSlotLearnRequested)
        self.slotTable.context_menu.slot_clear_requested.connect(self.onSlotClearRequested)
        self.slotTable.context_menu.slot_preview_requested.connect(self.onSlotPreviewRequested)
        
        self.updateSlotUsage()

    def on_pass_selected(self, item):
        renderpass_type = item.text()
        widget = RenderPassWidget(
            renderpass_type,
            self.available_slots,
            self.startSlotSelection,
            self.removeRenderPass,
            self.updateSlotSource
        )
        lw = self.listWidget.list_widget
        listItem = QListWidgetItem()
        listItem.setSizeHint(widget.sizeHint())
        lw.addItem(listItem)
        lw.setItemWidget(listItem, widget)

    def removeRenderPass(self, widget):
        """
        Remove the render pass widget and its list item from the list.
        """
        lw = self.listWidget.list_widget
        for i in range(lw.count()):
            item = lw.item(i)
            if lw.itemWidget(item) == widget:
                # Entferne das Item aus der QListWidget und lösche es sauber
                lw.takeItem(i)
                widget.setParent(None)
                widget.deleteLater()
                item = None
                self.updateSlotUsage()
                break

    def import_image(self):
        """Handle importing a new image."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*.*)"
        )
        
        if file_path:
            try:
                image = Image.open(file_path)
                filename = os.path.basename(file_path)
                self.imported_images[filename] = image
                self.imported_images_list.addItem(filename)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
    
    def assign_image_to_slot(self, item):
        """Handle double-click on imported image to assign to slot."""
        filename = item.text()
        if filename in self.imported_images:
            self._assigning_imported_image = filename
            QMessageBox.information(self, "Assign Image", "Now click a slot to assign the image to it.")

    def startSlotSelection(self, mode, widget):
        self.currentSelectionMode = mode
        self.currentWidget = widget

    def onSlotClicked(self, slotName):
        if hasattr(self, '_assigning_imported_image'):
            # We're in the process of assigning an imported image
            if slotName in self.original_image_slots:
                # Don't allow overwriting slots with original images
                QMessageBox.warning(self, "Cannot Assign", "This slot already contains an original image.")
                return
                
            image = self.imported_images[self._assigning_imported_image]
            self.slotTable.set_image(slotName, image)
            self.original_image_slots.add(slotName)
            delattr(self, '_assigning_imported_image')
            self.imported_images_list.clearSelection()
            return
            
        if not self.currentWidget or not self.currentSelectionMode:
            return
            
        # Don't allow setting output to slots containing original images
        if self.currentSelectionMode == 'output' and slotName in self.original_image_slots:
            QMessageBox.warning(self, "Cannot Set Output", "Cannot output to a slot containing an original image.")
            return
            
        self.currentWidget.set_slot(slotName)
        self.updateSlotUsage()
        self.currentSelectionMode = None
        self.currentWidget = None

    def updateSlotUsage(self):
        self.slotUsage = {slot: False for slot in self.available_slots}
        lw = self.listWidget.list_widget
        
        for i in range(lw.count()):
            widget = lw.itemWidget(lw.item(i))
            if widget.selectedOutput in self.slotUsage:
                self.slotUsage[widget.selectedOutput] = True
                
            settings = widget.get_settings()
            if settings.get("enabled") and settings.get("slot") in self.slotUsage:
                self.slotUsage[settings.get("slot")] = True

        self.slotUsage["slot0"] = True  # Original always occupied
        self.slotTable.refreshColors(self.slotUsage)

    def updateSlotSource(self, slot_name, source_info):
        """Update the source information for a slot."""
        self.slotTable.setSlotSource(slot_name, source_info)

    def onSlotLearnRequested(self, slot_name):
        """Handle slot learning via right-click context menu."""
        self.show_message(f"Learning slot: {slot_name}")
        # Here you can implement the actual learning logic
        # For now, we'll just show a message
        QMessageBox.information(self, "Learn Slot", f"Learning slot: {slot_name}")

    def onSlotClearRequested(self, slot_name):
        """Handle slot clearing via right-click context menu."""
        if slot_name in self.original_image_slots:
            QMessageBox.warning(self, "Cannot Clear", "Cannot clear original image slot.")
            return
            
        self.slotTable.set_image(slot_name, None)
        if slot_name in self.slotTable.slot_sources:
            del self.slotTable.slot_sources[slot_name]
        self.updateSlotUsage()
        self.show_message(f"Cleared slot: {slot_name}")

    def onSlotPreviewRequested(self, slot_name):
        """Handle slot preview via right-click context menu."""
        try:
            # Force show preview immediately
            if self.slotTable.preview_widget is None:
                from guiElements.modernSlotPreviewWidget import ModernSlotPreviewWidget
                self.slotTable.preview_widget = ModernSlotPreviewWidget()
                
            image = self.slotTable.get_image(slot_name)
            source_info = self.slotTable.slot_sources.get(slot_name)
            
            if image is None and source_info is None:
                self.show_message(f"No content to preview in slot: {slot_name}")
                return
                
            self.slotTable.preview_widget.updateContent(image, source_info)
            
            # Position near the slot button
            for slot_name_btn, btn in self.slotTable.buttons:
                if slot_name_btn == slot_name:
                    pos = btn.mapToGlobal(btn.rect().center())
                    self.slotTable.preview_widget.show_at_position(self.slotTable, pos)
                    break
        except Exception as e:
            self.show_message(f"Error showing preview: {str(e)}")

    def runRendering(self):
        # Validate pipeline before starting render
        issues = renderHook.validate_render_pipeline(self)
        if issues:
            QMessageBox.warning(self, "Validation Issues", 
                              "Please fix the following issues before rendering:\n\n" + "\n".join(issues))
            return
            
        # Disable the run button during rendering
        self.run_button.setEnabled(False)
        self.progress_bar.show()
        self.status_label.show()
        self.status_label.setText("Initializing render...")

        # Create worker and thread
        self.render_worker = RenderWorker(self)
        self.render_thread = QThread()
        
        # Move worker to thread
        self.render_worker.moveToThread(self.render_thread)
        
        # Connect signals
        self.render_thread.started.connect(self.render_worker.run)
        self.render_worker.progress.connect(self.updateProgress)
        self.render_worker.error.connect(self.handleRenderError)
        self.render_worker.finished.connect(self.handleRenderFinished)
        self.render_worker.finished.connect(self.render_thread.quit)
        self.render_worker.finished.connect(self.render_worker.deleteLater)
        self.render_thread.finished.connect(self.render_thread.deleteLater)
        
        # Start rendering
        self.render_thread.start()
        
    def updateProgress(self, message):
        self.status_label.setText(message)
        
    def handleRenderError(self, error_message):
        QMessageBox.critical(self, "Rendering Error", error_message)
        self.progress_bar.hide()
        self.status_label.hide()
        self.run_button.setEnabled(True)
        
    def handleRenderFinished(self):
        self.progress_bar.hide()
        self.status_label.setText("Render completed successfully!")
        self.run_button.setEnabled(True)
        # Hide status after 3 seconds
        QTimer.singleShot(3000, self.status_label.hide)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image File", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*.*)")
            
        if file_path:
            try:
                image = Image.open(file_path).convert("RGB")
                filename = os.path.basename(file_path)
                self.imported_images[filename] = image
                self.imported_images_list.addImage(filename, image)
                self.show_message(f"Successfully imported image: {filename} - Drag to any slot to assign")
            except Exception as e:
                error_msg = f"Failed to load image: {str(e)}"
                self.show_message(error_msg)
                QMessageBox.critical(self, "Error", error_msg)

    def saveProject(self):
        """Save the current project configuration to a JSON file."""
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        import json
        
        fileDialog = QFileDialog(self)
        fileDialog.setWindowTitle("Save Project")
        fileDialog.setNameFilter("JSON Files (*.json)")
        fileDialog.setAcceptMode(QFileDialog.AcceptSave)
        
        if fileDialog.exec():
            file_path = fileDialog.selectedFiles()[0]
            if not file_path.endswith('.json'):
                file_path += '.json'
                
            try:
                project_data = {
                    "render_passes": [],
                    "slot_images": {},
                    "slot_usage": self.slotUsage
                }
                
                # Save render pass configurations
                lw = self.listWidget.list_widget
                for i in range(lw.count()):
                    widget = lw.itemWidget(lw.item(i))
                    if widget:
                        pass_data = {
                            "type": widget.renderpass_type,
                            "settings": widget.get_settings(),
                            "output_slot": widget.selectedOutput
                        }
                        project_data["render_passes"].append(pass_data)
                
                # Save slot images
                slot_images = {}
                for slot in self.available_slots:
                    if slot in self.slotTable.slot_images:
                        # For now, we'll just save the path if available
                        # In a real implementation, you might want to save images as base64
                        slot_images[slot] = "image_data_placeholder"
                
                project_data["slot_images"] = slot_images
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(project_data, f, indent=2, ensure_ascii=False)
                
                QMessageBox.information(self, "Project Saved", f"Project saved successfully to:\n{file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save project:\n{str(e)}")

    def openOutputFolder(self):
        """Open the output folder in the system's file explorer."""
        output_dir = get_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        
        if os.name == 'nt':  # Windows
            subprocess.run(['explorer', output_dir])
        elif os.name == 'posix':  # macOS and Linux
            if sys.platform == 'darwin':  # macOS
                subprocess.run(['open', output_dir])
            else:  # Linux
                subprocess.run(['xdg-open', output_dir])

    def loadProject(self):
        """Load a project configuration from a JSON file."""
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        import json
        
        fileDialog = QFileDialog(self)
        fileDialog.setWindowTitle("Load Project")
        fileDialog.setNameFilter("JSON Files (*.json)")
        
        if fileDialog.exec():
            file_path = fileDialog.selectedFiles()[0]
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    project_data = json.load(f)
                
                # Clear existing render passes
                lw = self.listWidget.list_widget
                lw.clear()
                
                # Load render passes
                for pass_data in project_data.get("render_passes", []):
                    renderpass_type = pass_data.get("type")
                    if renderpass_type in self.render_passes_config:
                        widget = RenderPassWidget(
                            renderpass_type,
                            self.available_slots,
                            self.startSlotSelection,
                            self.removeRenderPass
                        )
                        
                        # Load settings
                        if "settings" in pass_data:
                            widget.settingsWidget.set_values(pass_data["settings"])
                        
                        # Set output slot
                        if "output_slot" in pass_data:
                            widget.selectedOutput = pass_data["output_slot"]
                            widget.outputLabel.setText(f"Output: {widget.selectedOutput}")
                        
                        # Add to list
                        listItem = QListWidgetItem()
                        listItem.setSizeHint(widget.sizeHint())
                        lw.addItem(listItem)
                        lw.setItemWidget(listItem, widget)
                
                # Load slot usage
                if "slot_usage" in project_data:
                    self.slotUsage = project_data["slot_usage"]
                
                self.updateSlotUsage()
                QMessageBox.information(self, "Project Loaded", f"Project loaded successfully from:\n{file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load project:\n{str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GUI()
    window.resize(800, 900)
    window.show()
    sys.exit(app.exec())
