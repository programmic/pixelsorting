# renderPassSettingsWidget.py

from __future__ import annotations
from typing import Optional
from PySide6.QtWidgets import *
from PySide6.QtCore import Qt
from superqt import QDoubleSlider, QToggleSwitch, QRangeSlider


class RenderPassSettingsWidget(QWidget):
    """
    A widget for rendering pass settings.

    :param settings_config: A list of settings configurations.
    :param parent: The parent widget.
    """
    def __init__(self, settings_config: list[dict], saved_settings: Optional[dict] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(4, 4, 4, 4)
        self.layout.setSpacing(4)
        self.controls = {}
        self._used_labels = set()
        self.settings_config = settings_config

        switch_group = QWidget()
        switch_layout = QHBoxLayout()
        switch_layout.setContentsMargins(4, 4, 4, 4)
        switch_layout.setSpacing(4)
        switch_group.setLayout(switch_layout)
        switch_count = 0

        filtered_settings = [s for s in settings_config if "kategory" not in s]
        default_values = {}
        for setting in filtered_settings:
            key = setting.get("alias") or setting.get("label", "")
            if key:
                default_values[key] = setting.get("default")

        if saved_settings:
            for key, default_val in default_values.items():
                if key in saved_settings:
                    default_values[key] = saved_settings[key]

        for idx, setting in enumerate(filtered_settings):
            # Only create controls for settings with a valid label
            label_text = setting.get("label")
            alias = setting.get("alias")
            key = alias or label_text
            if not label_text or not str(label_text).strip():
                print(f"Skipping setting {idx} (no label): {setting}")
                continue
            # Ensure uniqueness only for fallback labels (shouldn't be needed if config is correct)
            if key in self._used_labels:
                base_label = key
                count = 1
                while key in self._used_labels:
                    key = f"{base_label}_{count}"
                    count += 1
            self._used_labels.add(key)
            t = setting.get("type")
            default_value = setting.get("default")
            control = None

            # --- Support for image_input (mask) using slot system ---
            if t == "image_input":
                widget = QWidget()
                v_layout = QVBoxLayout()
                v_layout.setContentsMargins(0, 0, 0, 0)
                label = QLabel(label_text)
                label.setStyleSheet("color: #222; font-weight: bold; background: #f5f5f5; padding: 2px; border-radius: 3px;")
                v_layout.addWidget(label)
                combo = QComboBox()
                combo.setStyleSheet("background: #e0e0e0; color: #222; border-radius: 3px;")
                # Use availableSlots if provided in settings_config
                available_slots = setting.get("availableSlots")
                if not available_slots and hasattr(self, 'availableSlots'):
                    available_slots = self.availableSlots
                if not available_slots:
                    available_slots = []
                combo.addItem("<none>")
                for slot in available_slots:
                    combo.addItem(slot)
                v_layout.addWidget(combo)
                widget.setLayout(v_layout)
                self.controls[key] = combo
                control = widget
                self.layout.addWidget(control)
                continue

            if t == "switch":
                if switch_count == 0:
                    self.layout.addWidget(switch_group)
                widget = QWidget()
                hbox = QHBoxLayout()
                hbox.setContentsMargins(0, 0, 0, 0)
                label = QLabel(label_text)
                label.setStyleSheet("color: #222; font-weight: bold; background: #f5f5f5; padding: 2px; border-radius: 3px;")
                toggle = QToggleSwitch()
                toggle.setCheckable(True)
                toggle.setChecked(bool(default_value))
                print(f"[DEBUG] [CREATE] Switch '{label_text}' created, default={default_value}, checked={toggle.isChecked()}")
                print(f"[DEBUG] [META] toggle metaObject methodCount: {toggle.metaObject().methodCount()}")
                for i in range(toggle.metaObject().methodCount()):
                    m = toggle.metaObject().method(i)
                    print(f"[DEBUG] [META] toggle methodSignature: {m.methodSignature().data().decode()}")
                hbox.addWidget(label)
                hbox.addWidget(toggle)
                hbox.addStretch()
                widget.setLayout(hbox)
                switch_layout.addWidget(widget)
                self.controls[key] = toggle
                switch_count += 1
                # Add debug for toggling and dependency
                def make_toggle_handler(label, setting):
                    def handler(state):
                        print(f"[DEBUG] [TOGGLE] '{label}' toggled to {state}")
                        requires = setting.get("requires")
                        if requires and isinstance(requires, dict):
                            for req_label, req_value in requires.items():
                                # Try both alias and label for dependency lookup
                                target = self.controls.get(req_label)
                                if target is None:
                                    # Try to find by label if alias not found, or vice versa
                                    alt_key = None
                                    for s in self.settings_config:
                                        if s.get("alias") == req_label:
                                            alt_key = s.get("label")
                                            break
                                        if s.get("label") == req_label:
                                            alt_key = s.get("alias")
                                            break
                                    if alt_key:
                                        target = self.controls.get(alt_key)
                                if target is None:
                                    print(f"[DEBUG] [DEPENDENCY] Target '{req_label}' not found for '{label}'")
                                    continue
                                if isinstance(target, QToggleSwitch):
                                    print(f"[DEBUG] [DEPENDENCY] Setting switch '{req_label}' to {req_value} due to '{label}'")
                                    if target.isChecked() != bool(req_value):
                                        target.blockSignals(True)
                                        target.setChecked(bool(req_value))
                                        target.blockSignals(False)
                                        target.toggled.emit(bool(req_value))
                                        print(f"[DEBUG] [DEPENDENCY] Switch '{req_label}' set to {req_value} (triggered by '{label}')")
                                elif isinstance(target, dict) and 'slider' in target:
                                    slider_obj = target['slider']
                                    val = req_value
                                    print(f"[DEBUG] [DEPENDENCY] Setting slider '{req_label}' to {val} due to '{label}'")
                                    slider_obj.blockSignals(True)
                                    slider_obj.setValue(val)
                                    slider_obj.blockSignals(False)
                                    slider_obj.valueChanged.emit(val)
                                    if 'value_edit' in target:
                                        target['value_edit'].setText(f"{val:.2f}" if isinstance(slider_obj, QDoubleSlider) else str(val))
                                        print(f"[DEBUG] [DEPENDENCY] Slider '{req_label}' text set to {val}")
                                else:
                                    print(f"[DEBUG] [DEPENDENCY] Unhandled dependency type for '{req_label}' from '{label}' (type={type(target)})")
                    return handler
                if hasattr(toggle, "toggled"):
                    handler = make_toggle_handler(label_text, setting)
                    toggle.toggled.connect(handler)
                    toggle._handler_ref = handler
                    print(f"[DEBUG] [CONNECT] Connected 'toggled(bool)' for '{label_text}', receivers={toggle.receivers('toggled(bool)')}")
                else:
                    print(f"[DEBUG] [WARNING] toggle has no 'toggled' signal for '{label_text}'")
                continue
            # Modular slider creation: each slider is mapped independently by label
            if t == "slider":
                widget = QWidget()
                v_layout = QVBoxLayout()
                v_layout.setContentsMargins(0, 0, 0, 0)
                label = QLabel(label_text)
                label.setStyleSheet("color: #222; font-weight: bold; background: #f5f5f5; padding: 2px; border-radius: 3px;")
                v_layout.addWidget(label)
                is_integer = setting.get("integer", False)
                min_val = setting.get("min", 0)
                max_val = setting.get("max", 100)
                if is_integer:
                    slider = QSlider(Qt.Horizontal)
                    slider.setMinimum(int(min_val))
                    slider.setMaximum(int(max_val))
                    slider.setValue(int(float(default_value)) if default_value is not None else 0)
                    slider.setSingleStep(1)
                    value_edit = QLineEdit(str(slider.value()))
                else:
                    slider = QDoubleSlider(Qt.Horizontal)
                    slider.setMinimum(min_val)
                    slider.setMaximum(max_val)
                    slider.setValue(float(default_value) if default_value is not None else 0)
                    slider.setSingleStep(0.1)
                    value_edit = QLineEdit(f"{slider.value():.2f}")
                value_edit.setFixedWidth(60)
                value_edit.setAlignment(Qt.AlignCenter)

                # --- Bulletproof handlers ---
                def make_slider_handlers(slider, value_edit, label_text, is_integer, min_val, max_val):
                    def update_slider_from_text():
                        try:
                            new_value = int(value_edit.text()) if is_integer else float(value_edit.text())
                            if min_val <= new_value <= max_val:
                                slider.setValue(new_value)
                            else:
                                value_edit.setText(str(slider.value()) if is_integer else f"{slider.value():.2f}")
                        except ValueError:
                            value_edit.setText(str(slider.value()) if is_integer else f"{slider.value():.2f}")

                    def update_text_from_slider(value):
                        value_edit.setText(str(value) if is_integer else f"{value:.2f}")

                    return update_slider_from_text, update_text_from_slider

                handler_text, handler_slider = make_slider_handlers(slider, value_edit, label_text, is_integer, min_val, max_val)
                value_edit.editingFinished.connect(handler_text)
                slider.valueChanged.connect(handler_slider)

                # Keep references to avoid garbage collection
                slider._handler_text = handler_text
                slider._handler_slider = handler_slider
                value_edit._handler_text = handler_text
                value_edit._handler_slider = handler_slider

                # --- Fine-tune buttons ---
                minus_btn = QPushButton("-")
                plus_btn = QPushButton("+")
                minus_btn.setFixedSize(25, 25)
                plus_btn.setFixedSize(25, 25)

                def decrement_value():
                    new_value = max(slider.value() - (1 if is_integer else 0.1), slider.minimum())
                    slider.setValue(new_value)

                def increment_value():
                    new_value = min(slider.value() + (1 if is_integer else 0.1), slider.maximum())
                    slider.setValue(new_value)

                minus_btn.clicked.connect(decrement_value)
                plus_btn.clicked.connect(increment_value)

                h_slider_layout = QHBoxLayout()
                h_slider_layout.setContentsMargins(4, 4, 4, 4)
                h_slider_layout.setSpacing(4)
                h_slider_layout.addWidget(slider)
                h_slider_layout.addWidget(value_edit)
                h_slider_layout.addWidget(minus_btn)
                h_slider_layout.addWidget(plus_btn)

                slider_widget = QWidget()
                slider_widget.setLayout(h_slider_layout)
                v_layout.addWidget(slider_widget)
                widget.setLayout(v_layout)

                self.controls[label_text] = {'slider': slider, 'value_edit': value_edit, '_widget': widget}
                self.layout.addWidget(widget)
                continue
                widget.setLayout(v_layout)
                self.layout.addWidget(widget)
                # Store both slider and value_edit under the label for easy access
                self.controls[label_text] = {'slider': slider, 'value_edit': value_edit}
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
            if t == "switch":
                toggle = QToggleSwitch()
                toggle.setObjectName(label_text)
                self.controls[label_text] = toggle
                self.layout.addWidget(toggle)

                print(f"[DEBUG] Switch created: {label_text}")
                print(f"[DEBUG] toggle metaObject methodCount: {toggle.metaObject().methodCount()}")
                for i in range(toggle.metaObject().methodCount()):
                    method = toggle.metaObject().method(i)
                    print(f"[DEBUG] toggle methodSignature: {method.methodSignature().data().decode()}")

                # --- Handler factory so GC doesn't eat our lambda ---
                def make_toggle_handler(label, setting):
                    def handler(state):
                        print(f"[DEBUG] Handler entered for {label} with state={state}")
                        requires = setting.get("requires")
                        if requires and isinstance(requires, dict):
                            for req_label, req_value in requires.items():
                                target = self.controls.get(req_label)
                                if target is None:
                                    print(f"[DEBUG] Target {req_label} not found")
                                    continue
                                if isinstance(target, QToggleSwitch):
                                    if target.isChecked() != bool(req_value):
                                        target.blockSignals(True)
                                        target.setChecked(bool(req_value))
                                        target.blockSignals(False)
                                        target.toggled.emit(bool(req_value))
                                        print(f"[DEBUG] Set toggle {req_label} to {req_value}")
                                elif isinstance(target, dict) and 'slider' in target:
                                    slider_obj = target['slider']
                                    val = req_value
                                    slider_obj.blockSignals(True)
                                    slider_obj.setValue(val)
                                    slider_obj.blockSignals(False)
                                    slider_obj.valueChanged.emit(val)
                                    if 'value_edit' in target:
                                        target['value_edit'].setText(f"{val:.2f}" if isinstance(slider_obj, QDoubleSlider) else str(val))
                                        print(f"[DEBUG] Set slider {req_label} to {val}")
                    return handler

                # --- Safe connect ---
                if hasattr(toggle, "toggled"):
                    handler = make_toggle_handler(label_text, setting)
                    toggle.toggled.connect(handler)
                    toggle._handler_ref = handler  # keep a strong reference
                    print(f"[DEBUG] Connected 'toggled(bool)' for {label_text}, receivers={toggle.receivers(toggle.toggled)}")
                else:
                    print(f"[DEBUG] WARNING: toggle has no 'toggled' signal for {label_text}")
                continue

            elif t == "dropdown":
                widget = QWidget()
                v_layout = QVBoxLayout()
                v_layout.setContentsMargins(0, 0, 0, 0)
                
                label = QLabel(label_text)
                combo = QComboBox()
                combo.setStyleSheet("background: #e0e0e0; color: #222; border-radius: 3px;")
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
                label.setStyleSheet("color: #222; font-weight: bold; background: #f5f5f5; padding: 2px; border-radius: 3px;")
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
                    
                    # Create editable value fields for range display
                    lower_edit = QLineEdit(f"{slider.value()[0]:.1f}")
                    upper_edit = QLineEdit(f"{slider.value()[1]:.1f}")
                    lower_edit.setStyleSheet("background: #fff; color: #222; border: 1px solid #bbb; border-radius: 3px;")
                    upper_edit.setStyleSheet("background: #fff; color: #222; border: 1px solid #bbb; border-radius: 3px;")
                    lower_edit.setFixedWidth(60)
                    upper_edit.setFixedWidth(60)
                    lower_edit.setAlignment(Qt.AlignCenter)
                    upper_edit.setAlignment(Qt.AlignCenter)
                    
                    def update_range_from_text():
                        try:
                            new_lower = float(lower_edit.text())
                            new_upper = float(upper_edit.text())
                            if slider.minimum() <= new_lower <= new_upper <= slider.maximum():
                                slider.setValue((new_lower, new_upper))
                            else:
                                # Reset to current values if invalid
                                current_lower, current_upper = slider.value()
                                lower_edit.setText(f"{current_lower:.1f}")
                                upper_edit.setText(f"{current_upper:.1f}")
                        except ValueError:
                            # Reset to current values if invalid input
                            current_lower, current_upper = slider.value()
                            lower_edit.setText(f"{current_lower:.1f}")
                            upper_edit.setText(f"{current_upper:.1f}")
                    
                    def update_text_from_range(values):
                        lower, upper = values
                        lower_edit.setText(f"{lower:.1f}")
                        upper_edit.setText(f"{upper:.1f}")
                    
                    lower_edit.editingFinished.connect(update_range_from_text)
                    upper_edit.editingFinished.connect(update_range_from_text)
                    slider.valueChanged.connect(update_text_from_range)
                    
                    # Create + and - buttons for fine-tuning
                    minus_btn = QPushButton("-")
                    plus_btn = QPushButton("+")
                    minus_btn.setStyleSheet("background: #e0e0e0; color: #222; border-radius: 3px;")
                    plus_btn.setStyleSheet("background: #e0e0e0; color: #222; border-radius: 3px;")
                    minus_btn.setFixedSize(25, 25)
                    plus_btn.setFixedSize(25, 25)
                    
                    def decrement_lower():
                        lower, upper = slider.value()
                        new_lower = max(lower - 1, slider.minimum())
                        slider.setValue((new_lower, upper))

                    def increment_lower():
                        lower, upper = slider.value()
                        new_lower = min(lower + 1, upper)
                        slider.setValue((new_lower, upper))

                    def decrement_upper():
                        lower, upper = slider.value()
                        new_upper = max(upper - 1, lower)
                        slider.setValue((lower, new_upper))

                    def increment_upper():
                        lower, upper = slider.value()
                        new_upper = min(upper + 1, slider.maximum())
                        slider.setValue((lower, new_upper))

                    # Assign +/- buttons to lower and upper
                    minus_btn_lower = QPushButton("-")
                    plus_btn_lower = QPushButton("+")
                    minus_btn_upper = QPushButton("-")
                    plus_btn_upper = QPushButton("+")
                    for btn in [minus_btn_lower, plus_btn_lower, minus_btn_upper, plus_btn_upper]:
                        btn.setFixedSize(25, 25)
                        btn.setStyleSheet("background: #e0e0e0; color: #222; border-radius: 3px;")

                    minus_btn_lower.clicked.connect(decrement_lower)
                    plus_btn_lower.clicked.connect(increment_lower)
                    minus_btn_upper.clicked.connect(decrement_upper)
                    plus_btn_upper.clicked.connect(increment_upper)

                    h_slider_layout = QHBoxLayout()
                    h_slider_layout.setContentsMargins(4, 4, 4, 4)  # Standardized margins
                    h_slider_layout.setSpacing(4)  # Consistent spacing
                    h_slider_layout.addWidget(minus_btn_lower)
                    h_slider_layout.addWidget(lower_edit)
                    h_slider_layout.addWidget(plus_btn_lower)
                    h_slider_layout.addWidget(slider)
                    h_slider_layout.addWidget(minus_btn_upper)
                    h_slider_layout.addWidget(upper_edit)
                    h_slider_layout.addWidget(plus_btn_upper)
                    
                    # Only use the correct layout with independent controls
                    
                    slider_widget = QWidget()
                    slider_widget.setLayout(h_slider_layout)
                    v_layout.addWidget(slider_widget)
                    
                    self.controls[key] = slider
                    
                elif t == "multislider":
                    # Use QRangeSlider for multi-handle support
                    slider = QRangeSlider(Qt.Horizontal)
                    slider.setMinimum(setting.get("min", 0))
                    slider.setMaximum(setting.get("max", 100))
                    if isinstance(default_value, (list, tuple)) and len(default_value) == 2:
                        slider.setValue(tuple(default_value))
                    else:
                        slider.setValue((setting.get("min", 0), setting.get("max", 100)))
                    slider.setSingleStep(1 if setting.get("integer", False) else 0.1)

                    # Create editable value fields for each handle
                    value_edits = []
                    value_layout = QHBoxLayout()
                    for i, val in enumerate(slider.value()):
                        value_edit = QLineEdit(f"{val:.2f}")
                        value_edit.setStyleSheet("background: #fff; color: #222; border: 1px solid #bbb; border-radius: 3px;")
                        value_edit.setFixedWidth(60)
                        value_edit.setAlignment(Qt.AlignCenter)
                        value_layout.addWidget(value_edit)
                        value_edits.append(value_edit)
                        def make_update_slider_from_text(idx, value_edit):
                            def handler():
                                try:
                                    new_value = float(value_edit.text())
                                    values = list(slider.value())
                                    values[idx] = new_value
                                    # Clamp to min/max
                                    values[idx] = max(slider.minimum(), min(slider.maximum(), values[idx]))
                                    slider.setValue(tuple(values))
                                except ValueError:
                                    value_edit.setText(f"{slider.value()[idx]:.2f}")
                            return handler
                        value_edit.editingFinished.connect(make_update_slider_from_text(i, value_edit))
                    def update_texts_from_slider(values):
                        for i, value_edit in enumerate(value_edits):
                            value_edit.setText(f"{values[i]:.2f}")
                    slider.valueChanged.connect(update_texts_from_slider)
                    v_layout.addLayout(value_layout)

                    # Create minus and plus buttons for each handle
                    minus_btns = []
                    plus_btns = []
                    for i in range(2):
                        minus_btn = QPushButton("-")
                        plus_btn = QPushButton("+")
                        minus_btn.setStyleSheet("background: #e0e0e0; color: #222; border-radius: 3px;")
                        plus_btn.setStyleSheet("background: #e0e0e0; color: #222; border-radius: 3px;")
                        minus_btn.setFixedSize(25, 25)
                        plus_btn.setFixedSize(25, 25)
                        minus_btns.append(minus_btn)
                        plus_btns.append(plus_btn)
                        def make_decrement(idx):
                            def handler():
                                values = list(slider.value())
                                values[idx] = max(values[idx] - slider.singleStep(), slider.minimum())
                                slider.setValue(tuple(values))
                            return handler
                        def make_increment(idx):
                            def handler():
                                values = list(slider.value())
                                values[idx] = min(values[idx] + slider.singleStep(), slider.maximum())
                                slider.setValue(tuple(values))
                            return handler
                        minus_btn.clicked.connect(make_decrement(i))
                        plus_btn.clicked.connect(make_increment(i))

                    h_slider_layout = QHBoxLayout()
                    h_slider_layout.setContentsMargins(4, 4, 4, 4)  # Standardized margins
                    h_slider_layout.setSpacing(4)  # Consistent spacing
                    h_slider_layout.addWidget(minus_btns[0])
                    h_slider_layout.addWidget(value_edits[0])
                    h_slider_layout.addWidget(plus_btns[0])
                    h_slider_layout.addWidget(slider)
                    h_slider_layout.addWidget(minus_btns[1])
                    h_slider_layout.addWidget(value_edits[1])
                    h_slider_layout.addWidget(plus_btns[1])
                    
                    slider_widget = QWidget()
                    slider_widget.setLayout(h_slider_layout)
                    v_layout.addWidget(slider_widget)
                    
                    self.controls[label_text] = slider
                else:
                    # Determine if this is an integer or float slider
                    is_integer = setting.get("integer", False)
                    min_val = setting.get("min", 0)
                    max_val = setting.get("max", 100)
                    
                    if is_integer:
                        slider = QSlider(Qt.Horizontal)
                        slider.setMinimum(int(min_val))
                        slider.setMaximum(int(max_val))
                        slider.setValue(int(float(default_value)) if default_value is not None else 0)
                        slider.setSingleStep(1)
                        
                        # Create editable value field
                        value_edit = QLineEdit(str(slider.value()))
                        value_edit.setFixedWidth(60)
                        value_edit.setAlignment(Qt.AlignCenter)
                        
                        def update_slider_from_text():
                            try:
                                new_value = int(value_edit.text())
                                if min_val <= new_value <= max_val:
                                    slider.setValue(new_value)
                                else:
                                    value_edit.setText(str(slider.value()))
                            except ValueError:
                                value_edit.setText(str(slider.value()))
                        
                        def update_text_from_slider(value):
                            value_edit.setText(str(value))
                        
                        value_edit.editingFinished.connect(update_slider_from_text)
                        slider.valueChanged.connect(update_text_from_slider)
                        
                    else:
                        # Float slider - use QDoubleSlider for better precision
                        slider = QDoubleSlider(Qt.Horizontal)
                        slider.setMinimum(min_val)
                        slider.setMaximum(max_val)
                        # Only set float value if not a list
                        if not isinstance(default_value, list):
                            slider.setValue(float(default_value) if default_value is not None else 0)
                        else:
                            # If default_value is a list, use the first value
                            slider.setValue(float(default_value[0]) if default_value else 0)
                        slider.setSingleStep(0.1)
                        
                        # Create editable value field with float formatting
                        value_edit = QLineEdit(f"{slider.value():.2f}")
                        value_edit.setFixedWidth(60)
                        value_edit.setAlignment(Qt.AlignCenter)
                        
                        def update_slider_from_text():
                            try:
                                new_value = float(value_edit.text())
                                if min_val <= new_value <= max_val:
                                    slider.setValue(new_value)
                                else:
                                    value_edit.setText(f"{slider.value():.2f}")
                            except ValueError:
                                value_edit.setText(f"{slider.value():.2f}")
                        
                        def update_text_from_slider(value):
                            value_edit.setText(f"{value:.2f}")
                        
                        value_edit.editingFinished.connect(update_slider_from_text)
                        slider.valueChanged.connect(update_text_from_slider)
                    
                    # Create + and - buttons for fine-tuning
                    minus_btn = QPushButton("-")
                    plus_btn = QPushButton("+")
                    minus_btn.setFixedSize(25, 25)
                    plus_btn.setFixedSize(25, 25)
                    
                    def decrement_value():
                        if is_integer:
                            new_value = max(slider.value() - 1, slider.minimum())
                        else:
                            new_value = max(slider.value() - 0.1, slider.minimum())
                        slider.setValue(new_value)
                    
                    def increment_value():
                        if is_integer:
                            new_value = min(slider.value() + 1, slider.maximum())
                        else:
                            new_value = min(slider.value() + 0.1, slider.maximum())
                        slider.setValue(new_value)
                    
                    minus_btn.clicked.connect(decrement_value)
                    plus_btn.clicked.connect(increment_value)
                    
                    h_slider_layout = QHBoxLayout()
                    h_slider_layout.setContentsMargins(4, 4, 4, 4)  # Standardized margins
                    h_slider_layout.setSpacing(4)  # Consistent spacing
                    h_slider_layout.addWidget(slider)
                    h_slider_layout.addWidget(value_edit)
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
        Get the current values of the settings widget, always returning valid types (no blank strings or None).

        :return: A dictionary containing the values of all controls.
        """
        result = {}
        for key, control in self.controls.items():
            if isinstance(control, dict) and 'slider' in control:
                slider = control['slider']
                if isinstance(slider, QSlider):
                    try:
                        result[key] = int(slider.value())
                    except Exception:
                        result[key] = 0
                elif isinstance(slider, QDoubleSlider):
                    try:
                        result[key] = float(slider.value())
                    except Exception:
                        result[key] = 0.0
                else:
                    result[key] = slider.value() if hasattr(slider, 'value') else 0
            elif isinstance(control, QButtonGroup):
                checked = control.checkedButton()
                val = checked.text() if checked else ""
                result[key] = val
            elif isinstance(control, QToggleSwitch):
                val = control.isChecked()
                result[key] = bool(val)
            elif isinstance(control, QSlider):
                val = control.value()
                try:
                    result[key] = int(val)
                except Exception:
                    result[key] = 0
            elif isinstance(control, QDoubleSlider):
                val = control.value()
                try:
                    result[key] = float(val)
                except Exception:
                    result[key] = 0.0
            elif isinstance(control, QRangeSlider):
                val = control.value()
                if (isinstance(val, (list, tuple)) and len(val) == 2 and
                    all(isinstance(x, (int, float)) for x in val)):
                    result[key] = tuple(val)
                else:
                    result[key] = (0, 0)
            elif isinstance(control, QComboBox):
                val = control.currentText()
                if val is None or val.strip() == "":
                    val = control.itemText(0) if control.count() > 0 else ""
                result[key] = val
            elif hasattr(control, 'text'):
                val = control.text()
                if val is None or val.strip() == "":
                    result[key] = ""
                else:
                    result[key] = val
            else:
                result[key] = ""
        # return result (removed misplaced return outside function)
        return result

    def set_values(self, values: dict):
        """
        Set the values of the settings widget.

        :param values: A dictionary containing the values to set.
        """
        for key, val in values.items():
            control = self.controls.get(key)
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
                    # Check if this is actually a QDoubleSlider (which inherits from QSlider)
                    if hasattr(control, 'setValue') and hasattr(control, 'minimum') and hasattr(control, 'maximum'):
                        # Try to determine if this should be a float value
                        try:
                            float_val = float(val)
                            control.setValue(float_val)
                        except (ValueError, TypeError):
                            # Fall back to integer conversion for compatibility
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
                print(f"Error setting value for {key}: {e}")
