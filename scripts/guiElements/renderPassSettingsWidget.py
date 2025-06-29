# renderPassSettingsWidget.py

from PySide6.QtWidgets import *
from PySide6.QtCore import Qt
from superqt import QDoubleSlider, QToggleSwitch


class RenderPassSettingsWidget(QWidget):
    """
    A widget for rendering pass settings.

    :param settings_config: A list of settings configurations.
    :param parent: The parent widget.
    """
    def __init__(self, settings_config: list[dict], parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.controls = {}
        
        # Track consecutive switches to stack horizontally
        switch_group = QWidget()
        switch_layout = QHBoxLayout()
        switch_layout.setContentsMargins(0, 0, 0, 0)
        switch_group.setLayout(switch_layout)
        switch_count = 0

        for setting in settings_config:
            control = None
            label_text = setting.get("label", "")
            t = setting.get("type")

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
                toggle.setChecked(setting.get("default", False))
                
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
                self.layout.addWidget(label)
                
                for option in options:
                    rb = QRadioButton(option)
                    h_layout.addWidget(rb)
                    button_group.addButton(rb)
                    if option == setting.get("default"):
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
                combo.setCurrentText(setting.get("default", ""))
                
                v_layout.addWidget(label)
                v_layout.addWidget(combo)
                widget.setLayout(v_layout)
                self.controls[label_text] = combo
                control = widget

            elif t in ["slider", "multislider"]:
                widget = QWidget()
                v_layout = QVBoxLayout()
                v_layout.setContentsMargins(0, 0, 0, 0)
                
                label = QLabel(label_text)
                v_layout.addWidget(label)
                
                if t == "multislider":
                    slider = QDoubleSlider(Qt.Horizontal)
                    slider.setMinimum(setting.get("min", 0))
                    slider.setMaximum(setting.get("max", 100))
                    slider.setValue(setting.get("default", 0))
                    slider.setSingleStep(0.1)
                    
                    value_label = QLabel(str(slider.value()))
                    value_label.setFixedWidth(50)
                    slider.valueChanged.connect(lambda v, l=value_label: l.setText(str(v)))
                    
                    h_slider_layout = QHBoxLayout()
                    h_slider_layout.addWidget(slider)
                    h_slider_layout.addWidget(value_label)
                    
                    slider_widget = QWidget()
                    slider_widget.setLayout(h_slider_layout)
                    v_layout.addWidget(slider_widget)
                    
                    self.controls[label_text] = slider
                else:
                    slider = QSlider(Qt.Horizontal)
                    slider.setMinimum(setting.get("min", 0))
                    slider.setMaximum(setting.get("max", 100))
                    slider.setValue(setting.get("default", 0))
                    
                    if setting.get("integer", False):
                        slider.setSingleStep(1)
                    
                    value_label = QLabel(str(slider.value()))
                    value_label.setFixedWidth(50)
                    slider.valueChanged.connect(lambda v, l=value_label: l.setText(str(v)))
                    
                    h_slider_layout = QHBoxLayout()
                    h_slider_layout.addWidget(slider)
                    h_slider_layout.addWidget(value_label)
                    
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
            if isinstance(control, QButtonGroup):
                for btn in control.buttons():
                    btn.setChecked(btn.text() == val)
            elif isinstance(control, QToggleSwitch):
                control.setChecked(bool(val))
            elif isinstance(control, QSlider):
                control.setValue(int(val))
            elif isinstance(control, QDoubleSlider):
                control.setValue(float(val))
            elif isinstance(control, QComboBox):
                control.setCurrentText(val)
