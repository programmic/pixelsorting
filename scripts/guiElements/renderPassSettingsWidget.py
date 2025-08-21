# renderPassSettingsWidget.py

from PySide6.QtWidgets import *
from PySide6.QtCore import Qt
from superqt import QDoubleSlider, QToggleSwitch, QRangeSlider


class RenderPassSettingsWidget(QWidget):
    """
    A widget for rendering pass settings.

    :param settings_config: A list of settings configurations.
    :param parent: The parent widget.
    """
    def __init__(self, settings_config: list[dict], saved_settings: dict = None, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.controls = {}
        self.settings_config = settings_config
        
        # Track consecutive switches to stack horizontally
        switch_group = QWidget()
        switch_layout = QHBoxLayout()
        switch_layout.setContentsMargins(0, 0, 0, 0)
        switch_group.setLayout(switch_layout)
        switch_count = 0

        # Filter out category info but keep other settings
        filtered_settings = [s for s in settings_config if "kategory" not in s]
        
        # Create a dictionary of default values from filtered settings
        default_values = {}
        for setting in filtered_settings:
            label = setting.get("label", "")
            if label:
                default_values[label] = setting.get("default")

        # Apply saved settings if provided
        if saved_settings:
            # Merge saved settings with defaults
            for label, default_val in default_values.items():
                if label in saved_settings:
                    default_values[label] = saved_settings[label]

        for setting in filtered_settings:
            control = None
            label_text = setting.get("label", "")
            t = setting.get("type")
            default_value = setting.get("default")

            if t == "switch":
                # If first switch in group, add the group widget to main layout
                if switch_count == 0:
                    self.layout.addWidget(switch_group)
                
                # Create switch with label
                widget = QWidget()
                hbox = QHBoxLayout()
                hbox.setContentsMargins(0, 0, 0, 0)
                
                label = QLabel(label_text)
                toggle = QToggleSwitch()
                toggle.setChecked(bool(default_value))
                
                hbox.addWidget(label)
                hbox.addWidget(toggle)
                hbox.addStretch()
                widget.setLayout(hbox)
                
                switch_layout.addWidget(widget)
                self.controls[label_text] = toggle
                switch_count += 1
                continue
            else:
                # If we had switches and now getting another control type
                if switch_count > 0:
                    switch_group = QWidget()
                    switch_layout = QHBoxLayout()
                    switch_layout.setContentsMargins(0, 0, 0, 0)
                    switch_group.setLayout(switch_layout)
                    switch_count = 0

            # Regular control processing (non-switch)
            if t == "radio":
                options = setting.get("options", [])
                widget = QWidget()
                h_layout = QHBoxLayout()
                h_layout.setContentsMargins(0, 0, 0, 0)
                button_group = QButtonGroup(self)
                
                label = QLabel(label_text)
                h_layout.addWidget(label)
                
                for option in options:
                    rb = QRadioButton(option)
                    h_layout.addWidget(rb)
                    button_group.addButton(rb)
                    if option == str(default_value):
                        rb.setChecked(True)
                
                self.controls[label_text] = button_group
                control = widget
                control.setLayout(h_layout)

            elif t == "dropdown":
                widget = QWidget()
                v_layout = QVBoxLayout()
                v_layout.setContentsMargins(0, 0, 0, 0)
                
                label = QLabel(label_text)
                combo = QComboBox()
                combo.addItems(setting.get("options", []))
                
                # Set default value
                default_text = str(default_value) if default_value is not None else ""
                index = combo.findText(default_text)
                if index >= 0:
                    combo.setCurrentIndex(index)
                elif default_text and default_text in setting.get("options", []):
                    combo.setCurrentText(default_text)
                
                v_layout.addWidget(label)
                v_layout.addWidget(combo)
                widget.setLayout(v_layout)
                self.controls[label_text] = combo
                control = widget

            elif t in ["slider", "multislider", "dualslider"]:
                widget = QWidget()
                v_layout = QVBoxLayout()
                v_layout.setContentsMargins(0, 0, 0, 0)
                
                label = QLabel(label_text)
                v_layout.addWidget(label)
                
                if t == "dualslider":
                    # Dual slider for range selection
                    slider = QRangeSlider(Qt.Horizontal)
                    slider.setMinimum(setting.get("min", 0))
                    slider.setMaximum(setting.get("max", 100))
                    
                    # Handle default values for dual slider
                    if isinstance(default_value, (list, tuple)) and len(default_value) == 2:
                        slider.setValue(default_value)
                    else:
                        # Use min/max as default range
                        slider.setValue((setting.get("min", 0), setting.get("max", 100)))
                    
                    # Create labels for range display
                    lower_label = QLabel(f"Min: {slider.value()[0]:.1f}")
                    upper_label = QLabel(f"Max: {slider.value()[1]:.1f}")
                    
                    def update_range_labels(values):
                        lower, upper = values
                        lower_label.setText(f"Min: {lower:.1f}")
                        upper_label.setText(f"Max: {upper:.1f}")
                    
                    slider.valueChanged.connect(update_range_labels)
                    
                    # Create + and - buttons for fine-tuning
                    minus_btn = QPushButton("-")
                    plus_btn = QPushButton("+")
                    minus_btn.setFixedSize(25, 25)
                    plus_btn.setFixedSize(25, 25)
                    
                    def decrement_range():
                        lower, upper = slider.value()
                        new_lower = max(lower - 1, slider.minimum())
                        new_upper = max(upper - 1, slider.minimum())
                        slider.setValue((new_lower, new_upper))
                    
                    def increment_range():
                        lower, upper = slider.value()
                        new_lower = min(lower + 1, slider.maximum())
                        new_upper = min(upper + 1, slider.maximum())
                        slider.setValue((new_lower, new_upper))
                    
                    minus_btn.clicked.connect(decrement_range)
                    plus_btn.clicked.connect(increment_range)
                    
                    h_slider_layout = QHBoxLayout()
                    h_slider_layout.addWidget(lower_label)
                    h_slider_layout.addWidget(slider)
                    h_slider_layout.addWidget(upper_label)
                    h_slider_layout.addWidget(minus_btn)
                    h_slider_layout.addWidget(plus_btn)
                    
                    slider_widget = QWidget()
                    slider_widget.setLayout(h_slider_layout)
                    v_layout.addWidget(slider_widget)
                    
                    self.controls[label_text] = slider
                    
                elif t == "multislider":
                    slider = QDoubleSlider(Qt.Horizontal)
                    slider.setMinimum(setting.get("min", 0))
                    slider.setMaximum(setting.get("max", 100))
                    slider.setValue(float(default_value) if default_value is not None else 0)
                    slider.setSingleStep(0.1)
                    
                    value_label = QLabel(str(slider.value()))
                    value_label.setFixedWidth(50)
                    slider.valueChanged.connect(lambda v, l=value_label: l.setText(str(v)))
                    
                    # Create + and - buttons for fine-tuning
                    minus_btn = QPushButton("-")
                    plus_btn = QPushButton("+")
                    minus_btn.setFixedSize(25, 25)
                    plus_btn.setFixedSize(25, 25)
                    
                    def decrement_value():
                        new_value = max(slider.value() - 0.1, slider.minimum())
                        slider.setValue(new_value)
                    
                    def increment_value():
                        new_value = min(slider.value() + 0.1, slider.maximum())
                        slider.setValue(new_value)
                    
                    minus_btn.clicked.connect(decrement_value)
                    plus_btn.clicked.connect(increment_value)
                    
                    h_slider_layout = QHBoxLayout()
                    h_slider_layout.addWidget(slider)
                    h_slider_layout.addWidget(value_label)
                    h_slider_layout.addWidget(minus_btn)
                    h_slider_layout.addWidget(plus_btn)
                    
                    slider_widget = QWidget()
                    slider_widget.setLayout(h_slider_layout)
                    v_layout.addWidget(slider_widget)
                    
                    self.controls[label_text] = slider
                else:
                    slider = QSlider(Qt.Horizontal)
                    slider.setMinimum(setting.get("min", 0))
                    slider.setMaximum(setting.get("max", 100))
                    slider.setValue(int(float(default_value)) if default_value is not None else 0)
                    
                    if setting.get("integer", False):
                        slider.setSingleStep(1)
                    
                    value_label = QLabel(str(slider.value()))
                    value_label.setFixedWidth(50)
                    slider.valueChanged.connect(lambda v, l=value_label: l.setText(str(v)))
                    
                    # Create + and - buttons for fine-tuning
                    minus_btn = QPushButton("-")
                    plus_btn = QPushButton("+")
                    minus_btn.setFixedSize(25, 25)
                    plus_btn.setFixedSize(25, 25)
                    
                    def decrement_value():
                        new_value = max(slider.value() - 1, slider.minimum())
                        slider.setValue(new_value)
                    
                    def increment_value():
                        new_value = min(slider.value() + 1, slider.maximum())
                        slider.setValue(new_value)
                    
                    minus_btn.clicked.connect(decrement_value)
                    plus_btn.clicked.connect(increment_value)
                    
                    h_slider_layout = QHBoxLayout()
                    h_slider_layout.addWidget(slider)
                    h_slider_layout.addWidget(value_label)
                    h_slider_layout.addWidget(minus_btn)
                    h_slider_layout.addWidget(plus_btn)
                    
                    slider_widget = QWidget()
                    slider_widget.setLayout(h_slider_layout)
                    v_layout.addWidget(slider_widget)
                    
                    self.controls[label_text] = slider
                
                control = widget
                control.setLayout(v_layout)

            if control is not None:
                self.layout.addWidget(control)

    def get_values(self):
        """
        Get the current values of the settings widget.

        :return: A dictionary containing the values of all controls.
        """
        result = {}
        for label, control in self.controls.items():
            if isinstance(control, QButtonGroup):
                checked = control.checkedButton()
                result[label] = checked.text() if checked else None
            elif isinstance(control, QToggleSwitch):
                result[label] = control.isChecked()
            elif isinstance(control, QSlider):
                result[label] = control.value()
            elif isinstance(control, QDoubleSlider):
                result[label] = control.value()
            elif isinstance(control, QRangeSlider):
                result[label] = control.value()
            elif isinstance(control, QComboBox):
                result[label] = control.currentText()
            else:
                result[label] = None
        return result

    def set_values(self, values: dict):
        """
        Set the values of the settings widget.

        :param values: A dictionary containing the values to set.
        """
        for label, val in values.items():
            control = self.controls.get(label)
            if control is None:
                continue
                
            try:
                if isinstance(control, QButtonGroup):
                    # Handle radio button groups
                    for btn in control.buttons():
                        if btn.text() == str(val):
                            btn.setChecked(True)
                            break
                elif isinstance(control, QToggleSwitch):
                    control.setChecked(bool(val))
                elif isinstance(control, QSlider):
                    control.setValue(int(float(val)))
                elif isinstance(control, QDoubleSlider):
                    control.setValue(float(val))
                elif isinstance(control, QRangeSlider):
                    if isinstance(val, (list, tuple)) and len(val) == 2:
                        control.setValue(val)
                elif isinstance(control, QComboBox):
                    # Handle both string and numeric values
                    str_val = str(val)
                    index = control.findText(str_val)
                    if index >= 0:
                        control.setCurrentIndex(index)
                    else:
                        # Try to find by data if text not found
                        for i in range(control.count()):
                            if control.itemText(i) == str_val:
                                control.setCurrentIndex(i)
                                break
                        else:
                            # Handle numeric values for dropdowns
                            try:
                                # Try to set as current text
                                control.setCurrentText(str_val)
                            except:
                                pass
                        
            except (ValueError, TypeError) as e:
                print(f"Error setting value for {label}: {e}")
