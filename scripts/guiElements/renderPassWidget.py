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
        on_delete (callable): Callback to remove this widget from GUI
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

    def _cleanup(self):
        """Clean up child widgets and layouts to prevent leftover wrappers"""
        for child in self.findChildren(QWidget):
            child.setParent(None)
            child.deleteLater()
        if self.layout() is not None:
            while self.layout().count():
                item = self.layout().takeAt(0)
                if item.widget():
                    item.widget().setParent(None)
                    item.widget().deleteLater()

    def _delete_self(self):
        """Cleans up widget and calls callback to remove from GUI"""
        self._cleanup()
        if self.on_delete:
            self.on_delete(self)
        self.setParent(None)
        self.deleteLater()

    def _on_input_click(self, event, input_idx):
        self.selection_mode = 'input'
        self.current_selected_input = input_idx
        
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
        self.selection_mode = 'output'
        self.current_selected_input = None
        
        for label in self.input_labels:
            label.setStyleSheet("""
                background-color: lightgray;
                padding: 4px;
                border-radius: 4px;
                min-width: 80px;
            """)
        self.output_label.setStyleSheet(
            "background-color: #a0c4ff; padding: 4px; border-radius: 4px; min-width: 80px;"
        )
        self.on_select_slot('output', self)

    def set_slot(self, slot_name):
        if self.selection_mode == 'input' and self.current_selected_input is not None:
            self.selected_inputs[self.current_selected_input] = slot_name
            self.input_labels[self.current_selected_input].setText(f"Input {self.current_selected_input + 1}: {slot_name}")
        elif self.selection_mode == 'output':
            self.selected_output = slot_name
            self.output_label.setText(f"Output: {slot_name}")
        elif self.selection_mode == 'mask':
            self.mask_widget.set_mask_slot(slot_name)
        self.selection_mode = None
        self.current_selected_input = None

    def get_settings(self):
        """
        Retrieve current settings from the settings widget.
        """
        settings = self.settings_widget.get_settings()
        return settings

    def get_settings_config(self, renderpass_type):
        # This method should return settings configuration per pass type
        # For demo, returns a dummy dictionary
        return {"enabled": True, "slot": "slot1"}
    
    def start_slot_selection(self, mode):
        self.selection_mode = mode
        self.on_select_slot(mode, self)
