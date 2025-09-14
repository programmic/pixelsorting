
from __future__ import annotations
from typing import Optional, Callable
from PySide6.QtWidgets import *
from PySide6.QtCore import Qt
from .renderPassSettingsWidget import *

import json
import os

class RenderPassWidget(QWidget):
    def _on_output_click(self, event):
        self.selectionMode = 'output'
        self.currentSelectedInput = None
        for label in self.inputLabels:
            label.setStyleSheet("background-color: #f8f9fa; color: #333; padding: 4px; border-radius: 4px; min-width: 90px; font-weight: 600;")
        self.outputLabel.setStyleSheet("background-color: #1976d2; color: #fff; font-weight: bold; padding: 4px; border-radius: 4px; min-width: 90px;")
        if self.onSelectSlot:
            self.onSelectSlot('output', self)
    """
    Widget representing a single render pass with configurable inputs, output, and settings.
    Supports 1 or 2 input slots depending on pass type.
    Args:
        renderpass_type (str): Type of render pass (e.g., "Blur", "Mix By Percent")
        available_slots (list[str]): List of available slot names 
        onSelectSlot (callable): Callback when slot selection is initiated
        onDelete (callable): Callback to remove this widget from GUI
    """
    _settings_cache = None

    def __init__(self, renderpass_type: str, availableSlots: list[str], onSelectSlot: Callable, onDelete: Callable, onUpdateSlotSource: Optional[Callable] = None, saved_settings: Optional[dict] = None):
        super().__init__()
        self.collapsed = False  # Ensure this is set before anything else
        self.renderpass_type = renderpass_type
        self.availableSlots = availableSlots
        self.onSelectSlot = onSelectSlot
        self.onDelete = onDelete  # Store the delete callback
        self.onUpdateSlotSource = onUpdateSlotSource  # Callback to update slot source information

        # Get settings configuration including input count
        settings_config = self.get_settings_config(renderpass_type)
        category_settings = next((s for s in settings_config if "kategory" in s), None)
        self.numInputs = category_settings.get("num_inputs", 1) if category_settings else 1
        self.selectedInputs = [None] * self.numInputs
        self.selectedOutput = None
        self.savedSettings = saved_settings or {}

        self.mainLayout = QVBoxLayout(self)
        self.mainLayout.setContentsMargins(4, 4, 4, 4)

        # Create title bar with drag handle, collapse button, and delete button
        top = QHBoxLayout()
        top.setContentsMargins(4, 4, 4, 4)
        top.setSpacing(4)
        self.dragBar = QLabel("☰")
        self.dragBar.setFixedWidth(20)
        self.dragBar.setAlignment(Qt.AlignCenter)
        top.addWidget(self.dragBar)

        self.title = QLabel(f"Renderpass: {renderpass_type}")
        self.title.setStyleSheet("font-weight: bold;")
        top.addWidget(self.title)
        top.addStretch()

        # Collapse/expand button
        self.collapseBtn = QPushButton("−")
        self.collapseBtn.setFixedWidth(24)
        self.collapseBtn.setToolTip("Collapse/Expand")
        self.collapseBtn.setCheckable(True)
        self.collapseBtn.setChecked(False)
        self.collapseBtn.clicked.connect(self.toggle_collapsed)
        top.addWidget(self.collapseBtn)

        self.deleteBtn = QPushButton("✖")
        self.deleteBtn.setFixedWidth(30)
        self.deleteBtn.clicked.connect(self._delete_self)
        top.addWidget(self.deleteBtn)
        self.mainLayout.addLayout(top)

        # --- Mask Layout (separate, less prominent) ---
        self.maskLayout = QHBoxLayout()
        self.maskLabel = QLabel("Mask: <none>")
        self.maskLabel.setStyleSheet("""
            background-color: #ededed;
            color: #888;
            padding: 2px 4px;
            border-radius: 4px;
            min-width: 70px;
            font-size: 11px;
        """)
        self.maskLabel.mousePressEvent = self._on_mask_click
        self.maskLayout.addWidget(self.maskLabel)
        self.maskLayout.addStretch()
        self.selectedMask = None
        self.maskLayout.setContentsMargins(4, 0, 4, 4)
        self.maskLayout.setSpacing(2)

        # --- IO Layout (inputs/outputs) ---
        self.ioLayout = QHBoxLayout()
        self.inputLabels = []
        for i in range(self.numInputs):
            label = QLabel(f"Input {i+1}: <none>")
            label.setStyleSheet("background-color: #f8f9fa; color: #333; padding: 4px; border-radius: 4px; min-width: 90px; font-weight: 600;")
            label.mousePressEvent = lambda event, idx=i: self._on_input_click(idx)
            self.ioLayout.addWidget(label)
            self.inputLabels.append(label)
        self.outputLabel = QLabel("Output: <none>")
        self.outputLabel.setStyleSheet("background-color: #f8f9fa; color: #333; padding: 4px; border-radius: 4px; min-width: 90px; font-weight: 600;")
        self.outputLabel.mousePressEvent = self._on_output_click
        self.ioLayout.addWidget(self.outputLabel)
        self.ioLayout.setContentsMargins(4, 0, 4, 4)
        self.ioLayout.setSpacing(2)

        # --- Collapsible content area ---
        self.contentWidget = QWidget()
        self.contentLayout = QVBoxLayout(self.contentWidget)
        self.contentLayout.setContentsMargins(0, 0, 0, 0)
        self.contentLayout.setSpacing(0)
        # Move IO and mask layouts into content area
        self.contentLayout.addLayout(self.ioLayout)
        self.contentLayout.addLayout(self.maskLayout)

        settingsConfig = self.get_settings_config(self.renderpass_type)
        saved_settings = {}
        if self.savedSettings and 'settings' in self.savedSettings:
            saved_settings = self.savedSettings['settings']
        self.settingsWidget = RenderPassSettingsWidget(settingsConfig, saved_settings)
        self.contentLayout.addWidget(self.settingsWidget)

        self.mainLayout.addWidget(self.contentWidget)
        self.selectionMode = None
        self.currentSelectedInput = None
        self.collapsed = False
    def toggle_collapsed(self):
        self.collapsed = not self.collapsed
        if hasattr(self, 'contentWidget'):
            self.contentWidget.setVisible(not self.collapsed)
        self.collapseBtn.setText("+" if self.collapsed else "−")
    def _on_mask_click(self, event):
        self.selectionMode = 'mask'
        self.currentSelectedInput = None
        for label in self.inputLabels:
            label.setStyleSheet("background-color: #f8f9fa; color: #333; padding: 4px; border-radius: 4px; min-width: 90px; font-weight: 600;")
        self.maskLabel.setStyleSheet("background-color: #1976d2; color: #fff; font-weight: bold; padding: 2px 4px; border-radius: 4px; min-width: 70px; font-size: 11px;")
        self.outputLabel.setStyleSheet("background-color: #f8f9fa; color: #333; padding: 4px; border-radius: 4px; min-width: 90px; font-weight: 600;")
        self.onSelectSlot('mask', self)

    def set_slot(self, slot_name):
        old_output = self.selectedOutput
        if self.selectionMode == 'mask':
            self.selectedMask = slot_name
            self.maskLabel.setText(f"Mask: {slot_name}")
        elif self.selectionMode == 'input' and self.currentSelectedInput is not None:
            self.selectedInputs[self.currentSelectedInput] = slot_name
            self.inputLabels[self.currentSelectedInput].setText(f"Input {self.currentSelectedInput + 1}: {slot_name}")
        elif self.selectionMode == 'output':
            # Prevent selecting slot0 as output
            if slot_name == "slot0":
                self.outputLabel.setText("Output: <cannot use slot0>")
                return
            if old_output and self.onUpdateSlotSource:
                self.onUpdateSlotSource(old_output, None)
            self.selectedOutput = slot_name
            self.outputLabel.setText(f"Output: {slot_name}")
            if slot_name and self.onUpdateSlotSource:
                source_info = f"{self.renderpass_type} Pass\nInputs: {', '.join(str(inp) for inp in self.selectedInputs if inp)}"
                self.onUpdateSlotSource(slot_name, source_info)
        self.selectionMode = None
        self.currentSelectedInput = None

    def get_settings(self):
        """
        Retrieve current settings from the settings widget.
        """
        settings = self.settingsWidget.get_values()
        settings['inputs'] = self.selectedInputs
        settings['output'] = self.selectedOutput
        settings['mask'] = self.selectedMask
        return settings

    def load_settings(self, settings_dict):
        """
        Load saved settings into the widget.
        :param settings_dict: Dictionary containing saved settings
        """
        if not settings_dict:
            return
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
        if 'mask' in settings_dict:
            self.selectedMask = settings_dict['mask']
            self.maskLabel.setText(f"Mask: {self.selectedMask}")
        if 'settings' in settings_dict:
            settings_data = settings_dict['settings']
            if isinstance(settings_data, dict):
                self.settingsWidget.set_values(settings_data)
            else:
                print(f"Warning: Invalid settings format for {self.renderpass_type}: {type(settings_data)}")

    def get_settings_config(self, renderpass_type):
        # Load settings from renderPasses.json once and cache
        if RenderPassWidget._settings_cache is None:
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
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
            return settings_list
        else:
            return [
                {"kategory": "default", "num_inputs": 1},
                {"label": "Enabled", "type": "switch", "default": True}
            ]

    def start_slot_selection(self, mode):
        self.selectionMode = mode
        self.onSelectSlot(mode, self)

    def _delete_self(self):
        """
        Callback for delete button. Calls the onDelete callback to remove this widget from the GUI.
        """
        if callable(self.onDelete):
            self.onDelete(self)
