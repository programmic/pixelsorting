# masterGUI.py
from guiElements.preview_manager_instance import preview_manager

import sys
import subprocess
from PySide6.QtWidgets import *
from PySide6.QtCore import Qt, QThread, QTimer
from superqt import QSearchableListWidget
from utils import get_output_dir

from guiElements.modernSlotTableWidget import ModernSlotTableWidget
from guiElements.renderPassWidget import RenderPassWidget
from guiElements.importedImagesWidget import ImportedImagesListWidget
from renderWorker import RenderWorker
from PIL import Image
import random

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
        self.available_slots = [f"slot{i}" for i in range(16)]
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
        
        # --- Collapse/Expand All Buttons ---
        collapse_layout = QHBoxLayout()
        self.collapse_all_btn = QPushButton("Collapse All")
        self.collapse_all_btn.clicked.connect(self.collapse_all_renderpasses)
        collapse_layout.addWidget(self.collapse_all_btn)
        self.uncollapse_all_btn = QPushButton("Uncollapse All")
        self.uncollapse_all_btn.clicked.connect(self.uncollapse_all_renderpasses)
        collapse_layout.addWidget(self.uncollapse_all_btn)
        left_center.addLayout(collapse_layout)

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

        # Preload a default image from assets/images for faster testing
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            images_dir = os.path.join(project_root, 'assets', 'images')
            image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif', '.webp'))]
            if image_files:
                default_image_path = os.path.join(images_dir, random.choice(image_files))
            else:
                default_image_path = None
            if os.path.exists(default_image_path):
                from PIL import Image
                image = Image.open(default_image_path).convert("RGB")
                filename = os.path.basename(default_image_path)
                self.imported_images[filename] = image
                self.imported_images_list.addImage(filename, image)
        except Exception as e:
            print(f"[Warning] Could not preload default image: {e}")

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
        # Use display_name for available passes list
        self.pass_display_names = [v.get('display_name', k) for k, v in self.render_passes_config.items()]
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

    def collapse_all_renderpasses(self):
        lw = self.listWidget.list_widget
        for i in range(lw.count()):
            widget = lw.itemWidget(lw.item(i))
            if widget and hasattr(widget, 'collapsed') and not widget.collapsed:
                widget.toggle_collapsed()

    def uncollapse_all_renderpasses(self):
        lw = self.listWidget.list_widget
        for i in range(lw.count()):
            widget = lw.itemWidget(lw.item(i))
            if widget and hasattr(widget, 'collapsed') and widget.collapsed:
                widget.toggle_collapsed()
        
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

        # Preload a default image from assets/images for faster testing
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            images_dir = os.path.join(project_root, 'assets', 'images')
            image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif', '.webp'))]
            if image_files:
                default_image_path = os.path.join(images_dir, random.choice(image_files))
            else:
                default_image_path = None
            if os.path.exists(default_image_path):
                from PIL import Image
                image = Image.open(default_image_path).convert("RGB")
                filename = os.path.basename(default_image_path)
                self.imported_images[filename] = image
                self.imported_images_list.addImage(filename, image)
        except Exception as e:
            print(f"[Warning] Could not preload default image: {e}")

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

    def on_pass_selected(self, item: QListWidgetItem):
        display_label = item.text()
        # Map display label back to internal name
        internal_name = None
        for k, v in self.render_passes_config.items():
            if v.get('display_name', k) == display_label:
                internal_name = k
                break
        if not internal_name:
            self.show_message(f"No config found for {display_label}")
            return
        config = self.render_passes_config.get(internal_name, None)
        if not config:
            self.show_message(f"No config found for {display_label}")
            return
        # Pass settings to widget
        widget = RenderPassWidget(
            internal_name,
            self.available_slots,
            self.startSlotSelection,
            self.removeRenderPass,
            self.updateSlotSource,
            saved_settings=config.get('settings', {})
        )
        lw = self.listWidget.list_widget
        listItem = QListWidgetItem()
        listItem.setSizeHint(widget.sizeHint())
        lw.addItem(listItem)
        lw.setItemWidget(listItem, widget)

    def removeRenderPass(self, widget: RenderPassWidget):
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

    def startSlotSelection(self, mode: str, widget: RenderPassWidget):
        self.currentSelectionMode = mode
        self.currentWidget = widget

    def onSlotClicked(self, slotName: str):
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
                
            # Mark mask slot usage if mask is enabled
            try:
                mask = widget.maskWidget.get_values()
                if mask.get("enabled") and mask.get("slot") in self.slotUsage:
                    self.slotUsage[mask.get("slot")] = True
            except Exception:
                pass

        self.slotUsage["slot0"] = True  # Original always occupied
        self.slotTable.refreshColors(self.slotUsage)

    def updateSlotSource(self, slot_name: str, source_info: str):
        """Update the source information for a slot."""
        self.slotTable.setSlotSource(slot_name, source_info)

    def onSlotLearnRequested(self, slot_name: str):
        """Handle slot learning via right-click context menu."""
        self.show_message(f"Learning slot: {slot_name}")
        # Here you can implement the actual learning logic
        # For now, we'll just show a message
        QMessageBox.information(self, "Learn Slot", f"Learning slot: {slot_name}")

    def onSlotClearRequested(self, slot_name: str):
        """Handle slot clearing via right-click context menu."""
        if slot_name in self.original_image_slots:
            QMessageBox.warning(self, "Cannot Clear", "Cannot clear original image slot.")
            return
            
        self.slotTable.set_image(slot_name, None)
        if slot_name in self.slotTable.slot_sources:
            del self.slotTable.slot_sources[slot_name]
        self.updateSlotUsage()
        self.show_message(f"Cleared slot: {slot_name}")

    def onSlotPreviewRequested(self, slot_name: str):
        """Handle slot preview via right-click context menu."""
        try:
            # Force show preview immediately
            if self.slotTable.preview_widget is None:
                from guiElements.modernSlotPreviewWidgetMerged import ModernSlotPreviewWidget
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
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Image Files", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.tif *.webp);;All Files (*.*)")
            
        if file_paths:
            imported_count = 0
            failed_files = []
            
            for file_path in file_paths:
                try:
                    image = Image.open(file_path).convert("RGB")
                    filename = os.path.basename(file_path)
                    self.imported_images[filename] = image
                    self.imported_images_list.addImage(filename, image)
                    imported_count += 1
                except Exception as e:
                    failed_files.append(f"{os.path.basename(file_path)}: {str(e)}")
            
            # Show feedback
            if imported_count > 0:
                message = f"Successfully imported {imported_count} image(s)"
                if failed_files:
                    message += f", {len(failed_files)} failed"
                self.show_message(message)
                
            if failed_files:
                error_msg = f"Failed to import:\n" + "\n".join(failed_files)
                self.show_message(error_msg)
                QMessageBox.critical(self, "Import Error", error_msg)

    def saveProject(self):
        """Save the current project configuration to a JSON file."""
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        import json
        import os

        # Default to /saved directory in project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        saved_dir = os.path.join(project_root, "saved")
        os.makedirs(saved_dir, exist_ok=True)

        fileDialog = QFileDialog(self)
        fileDialog.setWindowTitle("Save Project")
        fileDialog.setNameFilter("JSON Files (*.json)")
        fileDialog.setAcceptMode(QFileDialog.AcceptSave)
        fileDialog.setDirectory(saved_dir)

        if fileDialog.exec():
            file_path = fileDialog.selectedFiles()[0]
            if not file_path.endswith('.json'):
                file_path += '.json'

            try:
                project_data = {
                    "render_passes": [],
                    "slot_images": {},
                    "slot_usage": self.slotUsage,
                    "slot_bindings": {}
                }

                # Save render pass configurations, using compact alias-to-param mapping
                lw = self.listWidget.list_widget
                for i in range(lw.count()):
                    widget = lw.itemWidget(lw.item(i))
                    if widget:
                        renderpass_type = widget.renderpass_type
                        config = self.render_passes_config.get(renderpass_type, {})
                        alias_map = config.get('alias_to_param', {})
                        settings = widget.get_settings()
                        print(f"[DEBUG][SAVE] RenderPass: {renderpass_type}")
                        print(f"[DEBUG][SAVE] Alias map: {alias_map}")
                        print(f"[DEBUG][SAVE] Widget settings: {settings}")
                        compact_settings = {alias: settings.get(alias_map.get(alias, alias)) for alias in alias_map}
                        print(f"[DEBUG][SAVE] Compact settings: {compact_settings}")
                        pass_data = {
                            "type": renderpass_type,
                            "settings": compact_settings,
                            "output_slot": widget.selectedOutput,
                            "alias_to_param": alias_map
                        }
                        if hasattr(widget, 'alias') and widget.alias:
                            pass_data["alias"] = widget.alias
                        project_data["render_passes"].append(pass_data)

                # Save slot images
                slot_images = {}
                for slot in self.available_slots:
                    if slot in self.slotTable.slot_images:
                        slot_images[slot] = "image_data_placeholder"
                project_data["slot_images"] = slot_images

                # Save slot bindings: for each slot, record its source and destination if available
                for slot in self.available_slots:
                    binding = {}
                    # Source: where does this slot get its image from?
                    if hasattr(self.slotTable, 'slot_sources') and slot in self.slotTable.slot_sources:
                        binding['source'] = self.slotTable.slot_sources[slot]
                    # Destination: which render pass outputs to this slot?
                    # Find any render pass that outputs to this slot
                    output_to_this = []
                    for i in range(lw.count()):
                        widget = lw.itemWidget(lw.item(i))
                        if widget and getattr(widget, 'selectedOutput', None) == slot:
                            output_to_this.append(widget.renderpass_type)
                    if output_to_this:
                        binding['destination'] = output_to_this
                    if binding:
                        project_data['slot_bindings'][slot] = binding

                def _normalize(obj):
                    # Recursively convert Enum values to their primitive .value
                    try:
                        from enum import Enum
                    except Exception:
                        Enum = None
                    if Enum is not None and isinstance(obj, Enum):
                        return obj.value
                    if isinstance(obj, dict):
                        return {k: _normalize(v) for k, v in obj.items()}
                    if isinstance(obj, list):
                        return [_normalize(v) for v in obj]
                    return obj

                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(_normalize(project_data), f, indent=2, ensure_ascii=False)

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

        # Default to /saved directory in project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        saved_dir = os.path.join(project_root, "saved")
        os.makedirs(saved_dir, exist_ok=True)

        fileDialog = QFileDialog(self)
        fileDialog.setWindowTitle("Load Project")
        fileDialog.setNameFilter("JSON Files (*.json)")
        fileDialog.setDirectory(saved_dir)

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
                    print(f"[DEBUG][LOAD] RenderPass: {renderpass_type}")
                    print(f"[DEBUG][LOAD] Pass data: {pass_data}")
                    if renderpass_type in self.render_passes_config:
                        print(f"[DEBUG][LOAD] Found config for {renderpass_type}")
                        widget = RenderPassWidget(
                            renderpass_type,
                            self.available_slots,
                            self.startSlotSelection,
                            self.removeRenderPass,
                            self.updateSlotSource,
                            saved_settings=pass_data if pass_data else None
                        )
                        if "alias" in pass_data:
                            setattr(widget, "alias", pass_data["alias"])
                        print(f"[DEBUG][LOAD] Created widget for {renderpass_type}")
                        listItem = QListWidgetItem()
                        listItem.setSizeHint(widget.sizeHint())
                        lw.addItem(listItem)
                        lw.setItemWidget(listItem, widget)
                # Load slot usage
                if "slot_usage" in project_data:
                    self.slotUsage = project_data["slot_usage"]

                # Restore slot sources from slot_bindings if present
                if "slot_bindings" in project_data:
                    for slot, binding in project_data["slot_bindings"].items():
                        if "source" in binding:
                            self.slotTable.setSlotSource(slot, binding["source"])

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
