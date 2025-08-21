from PySide6.QtWidgets import *
from PySide6.QtCore import Qt

from .maskWidget import *
from .renderPassSettingsWidget import *

import json
import os

class RenderPassWidget(QWidget):
    _settings_cache = None

    """
    Widget representing a single render pass with configurable inputs, output, and settings.
    Supports 1 or 2 input slots depending on pass type, plus mask functionality.

    Args:
        renderpass_type (str): Type of render pass (e.g., "Blur", "Mix By Percent")
        available_slots (list[str]): List of available slot names 
        onSelectSlot (callable): Callback when slot selection is initiated
        onDelete (callable): Callback to remove this widget from GUI
    """
    def __init__(self, renderpass_type: str, availableSlots: list[str], onSelectSlot, onDelete, onUpdateSlotSource=None, saved_settings: dict = None):
        super().__init__()
        self.renderpass_type = renderpass_type
        self.availableSlots = availableSlots
        self.onSelectSlot = onSelectSlot
        self.onDelete = onDelete  # Store the delete callback
        self.onUpdateSlotSource = onUpdateSlotSource  # Callback to update slot source information
        
        # Get settings configuration including input count
        settings_config = self.get_settings_config(renderpass_type)
        # Get category settings which contains input count
        category_settings = next((s for s in settings_config if "kategory" in s), None)
        # Determine number of inputs from category settings or default to 1
        self.numInputs = category_settings.get("num_inputs", 1) if category_settings else 1
        self.selectedInputs = [None] * self.numInputs
        self.selectedOutput = None
        
        # Store saved settings for later use
        self.savedSettings = saved_settings or {}

        self.mainLayout = QVBoxLayout(self)
        self.mainLayout.setContentsMargins(4, 4, 4, 4)

        # Create title bar with drag handle and delete button
        top = QHBoxLayout()
        self.dragBar = QLabel("☰")
        self.dragBar.setFixedWidth(20)
        self.dragBar.setAlignment(Qt.AlignCenter)
        top.addWidget(self.dragBar)

        self.title = QLabel(f"Renderpass: {renderpass_type}")
        self.title.setStyleSheet("font-weight: bold;")
        top.addWidget(self.title)
        top.addStretch()

        self.deleteBtn = QPushButton("✖")
        self.deleteBtn.setFixedWidth(30)
        self.deleteBtn.clicked.connect(self._delete_self)
        top.addWidget(self.deleteBtn)
        self.mainLayout.addLayout(top)

        # Input/output section
        self.ioLayout = QHBoxLayout()
        
        # Create input selection labels
        self.inputLabels = []
        for i in range(self.numInputs):
            label = QLabel(f"Input {i+1}: <none>")
            label.setStyleSheet("""
                background-color: lightgray;
                padding: 4px;
                border-radius: 4px;
                min-width: 80px;
            """)
            label.mousePressEvent = lambda e, idx=i: self._on_input_click(e, idx)
            self.ioLayout.addWidget(label)
            self.inputLabels.append(label)

        # Output selection
        self.outputLabel = QLabel("Output: <none>")
        self.outputLabel.setStyleSheet("""
            background-color: lightgray;
            padding: 4px;
            border-radius: 4px;
            min-width: 80px;
        """)
        self.outputLabel.mousePressEvent = self._on_output_click
        self.ioLayout.addWidget(self.outputLabel)
        
        self.mainLayout.addLayout(self.ioLayout)

        # Settings configuration
        settingsConfig = self.get_settings_config(renderpass_type)
        
        # Extract saved settings for this render pass
        saved_settings = {}
        if self.savedSettings and 'settings' in self.savedSettings:
            saved_settings = self.savedSettings['settings']
        
        self.settingsWidget = RenderPassSettingsWidget(settingsConfig, saved_settings)
        self.mainLayout.addWidget(self.settingsWidget)

        # Mask widget
        self.maskWidget = MaskWidget(availableSlots, self.start_slot_selection)
        self.mainLayout.addWidget(self.maskWidget)

        self.selectionMode = None
        self.currentSelectedInput = None

    def _cleanup(self):
        """Clean up child widgets and layouts to prevent leftover wrappers"""
        for child in self.findChildren(QWidget):
            child.setParent(None)
            child.deleteLater()
        if self.mainLayout is not None:
            while self.mainLayout.count():
                item = self.mainLayout.takeAt(0)
                if item.widget():
                    item.widget().setParent(None)
                    item.widget().deleteLater()

    def _delete_self(self):
        """Cleans up widget and calls callback to remove from GUI"""
        self._cleanup()
        if self.onDelete:
            self.onDelete(self)
        self.setParent(None)
        self.deleteLater()

    def _on_input_click(self, event, input_idx):
        self.selectionMode = 'input'
        self.currentSelectedInput = input_idx
        
        for i, label in enumerate(self.inputLabels):
            label.setStyleSheet(
                ("background-color: #a0c4ff;" if i == input_idx else "background-color: lightgray;")
                + "padding: 4px; border-radius: 4px; min-width: 80px;"
            )
        self.outputLabel.setStyleSheet("""
            background-color: lightgray;
            padding: 4px;
            border-radius: 4px;
            min-width: 80px;
        """)
        
        self.onSelectSlot('input', self)

    def _on_output_click(self, event):
        self.selectionMode = 'output'
        self.currentSelectedInput = None
        
        for label in self.inputLabels:
            label.setStyleSheet("""
                background-color: lightgray;
                padding: 4px;
                border-radius: 4px;
                min-width: 80px;
            """)
        self.outputLabel.setStyleSheet(
            "background-color: #a0c4ff; padding: 4px; border-radius: 4px; min-width: 80px;"
        )
        self.onSelectSlot('output', self)

    def set_slot(self, slot_name):
        # Store old output to clear its source info
        old_output = self.selectedOutput

        if self.selectionMode == 'input' and self.currentSelectedInput is not None:
            self.selectedInputs[self.currentSelectedInput] = slot_name
            self.inputLabels[self.currentSelectedInput].setText(f"Input {self.currentSelectedInput + 1}: {slot_name}")
        elif self.selectionMode == 'output':
            # Prevent selecting slot0 as output
            if slot_name == "slot0":
                self.outputLabel.setText("Output: <cannot use slot0>")
                return
                
            if old_output and self.onUpdateSlotSource:
                # Clear the old output slot source
                self.onUpdateSlotSource(old_output, None)
            self.selectedOutput = slot_name
            self.outputLabel.setText(f"Output: {slot_name}")
            if slot_name and self.onUpdateSlotSource:
                # Update the new output slot source
                source_info = f"{self.renderpass_type} Pass\nInputs: {', '.join(str(inp) for inp in self.selectedInputs if inp)}"
                self.onUpdateSlotSource(slot_name, source_info)
        elif self.selectionMode == 'mask':
            self.maskWidget.set_mask_slot(slot_name)
            
        self.selectionMode = None
        self.currentSelectedInput = None

    def get_settings(self):
        """
        Retrieve current settings from the settings widget.
        """
        settings = self.settingsWidget.get_values()
        return settings

    def load_settings(self, settings_dict):
        """
        Load saved settings into the widget.
        
        :param settings_dict: Dictionary containing saved settings
        """
        if not settings_dict:
            return
            
        # Load input/output slots
        if 'inputs' in settings_dict:
            inputs = settings_dict['inputs']
            for i, input_slot in enumerate(inputs):
                if i < len(self.selectedInputs):
                    self.selectedInputs[i] = input_slot
                    if i < len(self.inputLabels):
                        self.inputLabels[i].setText(f"Input {i+1}: {input_slot}")
        
        if 'output' in settings_dict:
            self.selectedOutput = settings_dict['output']
            self.outputLabel.setText(f"Output: {self.selectedOutput}")
        
        # Load mask settings
        if 'mask' in settings_dict:
            mask_settings = settings_dict['mask']
            self.maskWidget.load_settings(mask_settings)
        
        # Load render pass settings
        if 'settings' in settings_dict:
            # Ensure we have a proper dictionary for settings
            settings_data = settings_dict['settings']
            if isinstance(settings_data, dict):
                self.settingsWidget.set_values(settings_data)
            else:
                print(f"Warning: Invalid settings format for {self.renderpass_type}: {type(settings_data)}")

    def get_settings_config(self, renderpass_type):
        # Load settings from renderPasses.json once and cache
        if RenderPassWidget._settings_cache is None:
            try:
                # Get the directory where this script is located
                script_dir = os.path.dirname(os.path.abspath(__file__))
                # Go up two levels to project root (scripts/guiElements -> scripts -> project root)
                project_root = os.path.dirname(os.path.dirname(script_dir))
                json_path = os.path.join(project_root, 'renderPasses.json')
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        RenderPassWidget._settings_cache = json.load(f)
                except Exception as e:
                    print(f"Error loading renderPasses.json: {e}")
                    RenderPassWidget._settings_cache = {}
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

        settings_list = RenderPassWidget._settings_cache.get(renderpass_type)
        if settings_list and isinstance(settings_list, list):
            # Return the full list including category info
            return settings_list
        else:
            # Default fallback settings with category info
            return [
                {"kategory": "default", "num_inputs": 1},
                {"label": "Enabled", "type": "switch", "default": True}
            ]
    
    def start_slot_selection(self, mode):
        self.selectionMode = mode
        self.onSelectSlot(mode, self)
