# renderPassWidget.py

from PySide6.QtWidgets import *
from PySide6.QtCore import Qt

from guiElements.maskWidget import *
from guiElements.renderPassSettingsWidget import *

class RenderPassWidget(QWidget):
    """
    Widget representing a single render pass with configurable inputs, output, and settings.
    Supports 1 or 2 input slots depending on pass type, plus mask functionality.

    Args:
        renderpass_type (str): Type of render pass (e.g., "Blur", "Mix By Percent")
        available_slots (list[str]): List of available slot names 
        on_select_slot (callable): Callback when slot selection is initiated
    """
    def __init__(self, renderpass_type: str, available_slots: list[str], on_select_slot, on_delete):
        super().__init__()
        self.renderpass_type = renderpass_type
        self.available_slots = available_slots
        self.on_select_slot = on_select_slot
        self.on_delete = on_delete  # Store the delete callback
        
        # Determine number of input buttons needed
        self.num_inputs = 2 if renderpass_type in ["Mix By Percent", "Mix Screen", "Subtract"] else 1
        self.selected_inputs = [None] * self.num_inputs
        self.selected_output = None

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(4, 4, 4, 4)

        # Create title bar with drag handle and delete button
        top = QHBoxLayout()
        self.drag_bar = QLabel("☰")
        self.drag_bar.setFixedWidth(20)
        self.drag_bar.setAlignment(Qt.AlignCenter)
        top.addWidget(self.drag_bar)

        self.title = QLabel(f"Renderpass: {renderpass_type}")
        self.title.setStyleSheet("font-weight: bold;")
        top.addWidget(self.title)
        top.addStretch()

        self.delete_btn = QPushButton("✖")
        self.delete_btn.setFixedWidth(30)
        self.delete_btn.clicked.connect(self._delete_self)
        top.addWidget(self.delete_btn)
        self.layout.addLayout(top)

        # Input/output section
        self.io_layout = QHBoxLayout()
        
        # Create input selection labels
        self.input_labels = []
        for i in range(self.num_inputs):
            label = QLabel(f"Input {i+1}: <none>")
            label.setStyleSheet("""
                background-color: lightgray;
                padding: 4px;
                border-radius: 4px;
                min-width: 80px;
            """)
            label.mousePressEvent = lambda e, idx=i: self._on_input_click(e, idx)
            self.io_layout.addWidget(label)
            self.input_labels.append(label)

        # Output selection
        self.output_label = QLabel("Output: <none>")
        self.output_label.setStyleSheet("""
            background-color: lightgray;
            padding: 4px;
            border-radius: 4px;
            min-width: 80px;
        """)
        self.output_label.mousePressEvent = self._on_output_click
        self.io_layout.addWidget(self.output_label)
        
        self.layout.addLayout(self.io_layout)

        # Settings configuration
        settings_config = self.get_settings_config(renderpass_type)
        self.settings_widget = RenderPassSettingsWidget(settings_config)
        self.layout.addWidget(self.settings_widget)

        # Mask widget
        self.mask_widget = MaskWidget(available_slots, self.start_slot_selection)
        self.layout.addWidget(self.mask_widget)

        self.selection_mode = None
        self.current_selected_input = None

    def _delete_self(self):
        """Cleans up widget when deleted"""
        self.on_delete(self)  # Call the delete callback
        self.setParent(None)
        self.deleteLater()


    def _on_input_click(self, event, input_idx):
        """Handles click on input label to start slot selection"""
        self.selection_mode = 'input'
        self.current_selected_input = input_idx
        
        # Update highlights
        for i, label in enumerate(self.input_labels):
            label.setStyleSheet(
                "background-color: #a0c4ff;" if i == input_idx
                else "background-color: lightgray;"
                "padding: 4px; border-radius: 4px; min-width: 80px;" 
            )
        self.output_label.setStyleSheet("""
            background-color: lightgray;
            padding: 4px;
            border-radius: 4px;
            min-width: 80px;
        """)
        
        self.on_select_slot('input', self)

    def _on_output_click(self, event):
        """Handles click on output label to start slot selection"""
        self.selection_mode = 'output'
        self.current_selected_input = None
        
        # Reset all input highlights
        for label in self.input_labels:
            label.setStyleSheet("""
                background-color: lightgray;
                padding: 4px;
                border-radius: 4px;
                min-width: 80px;
            """)
        
        self.output_label.setStyleSheet("""
            background-color: #a0c4ff;
            padding: 4px;
            border-radius: 4px;
            min-width: 80px;
        """)
        
        self.on_select_slot('output', self)

    def set_slot(self, slot_name: str):
        """
        Assigns the currently selected slot to either:
        - The active input
        - The output
        - The mask (if enabled)
        """
        if self.selection_mode == 'input' and self.current_selected_input is not None:
            self.selected_inputs[self.current_selected_input] = slot_name
            self.input_labels[self.current_selected_input].setText(
                f"Input {self.current_selected_input+1}: {slot_name}"
            )
        elif self.selection_mode == 'output':
            self.selected_output = slot_name
            self.output_label.setText(f"Output: {slot_name}")
        elif self.mask_widget.enabled.isChecked():
            self.mask_widget.set_slot(slot_name)

        # Reset selection state
        self._reset_selection_ui()

    def _reset_selection_ui(self):
        """Resets all UI elements to non-selected state"""
        self.selection_mode = None
        self.current_selected_input = None
        
        for label in self.input_labels:
            label.setStyleSheet("""
                background-color: lightgray;
                padding: 4px;
                border-radius: 4px;
                min-width: 80px;
            """)
        
        self.output_label.setStyleSheet("""
            background-color: lightgray;
            padding: 4px;
            border-radius: 4px;
            min-width: 80px;
        """)

    def start_slot_selection(self, mode, widget):
        """Callback when mask slot selection is initiated"""
        if mode == 'mask':
            self.current_mask_widget = widget
            self.on_select_slot('mask', self)

    def get_settings_config(self, renderpass_type):
        """Returns configuration parameters for different render pass types"""
        if renderpass_type == "Blur":
            return [
                {"label": "Blur Type", "type": "radio", "options": ["Gaussian", "Box"], "default": "Gaussian"},
                {"label": "Radius", "type": "slider", "min": 1, "max": 20, "default": 5, "integer": True},
                {"label": "Enabled", "type": "switch", "default": True},
            ]
        elif renderpass_type == "Mix By Percent":
            return [
                {"label": "Mix Ratio", "type": "slider", "min": 0, "max": 100, "default": 50, "integer": True},
            ]
        elif renderpass_type == "Mask":
            return [
                {
                    "label": "Value",
                    "type": "multislider",
                    "min": 0.0,
                    "max": 255.0,
                    "default": [50.0, 120.0]
                },
            ]
        elif renderpass_type == "Simple Kuwahara":
            return [
                {"label": "Kernel", "type": "slider", "min": 2.0, "max": 64.0, "default": 8.0, "integer": True},
            ]
        elif renderpass_type == "PixelSort":
            return [
                {
                    "label": "Sort by",
                    "type": "dropdown",
                    "options": ['lum', 'hue', 'r', 'g', 'b'],
                    "default": "lum"
                },
                {
                    "label": "Rotate before processing",
                    "type": "radio",
                    "options": ['0', '90', '-90', '180'],
                    "default": "0"
                },
                {
                    "label": "Flip Horizontally",
                    "type": "switch",
                    "default": True
                },
                {
                    "label": "Flip Vertically",
                    "type": "switch",
                    "default": True
                },
                {
                    "label": "Use Chunk Splitting",
                    "type": "switch",
                    "default": True
                },
            ]
        else:
            return [
                {"label": "Enabled", "type": "switch", "default": True},
            ]

    def get_settings(self):
        """
        Returns all current settings including:
        - Input slots (as list if multiple inputs, single value otherwise)
        - Output slot
        - Mask settings
        - Other pass-specific settings
        """
        settings = self.settings_widget.get_values()
        settings["inputs"] = (
            self.selected_inputs if self.num_inputs > 1 
            else self.selected_inputs[0] if self.selected_inputs else None
        )
        return settings

    def _delete_self(self):
        """Cleans up widget when deleted"""
        self.setParent(None)
        self.deleteLater()
