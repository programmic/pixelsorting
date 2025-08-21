from PySide6.QtWidgets import *
from PySide6.QtCore import Qt, Signal
from superqt import QDoubleSlider, QRangeSlider


class DualSliderWidget(QWidget):
    """A dual slider widget for selecting ranges with two knobs."""
    
    rangeChanged = Signal(float, float)
    
    def __init__(self, parent=None, minimum=0.0, maximum=100.0, 
                 lower_value=None, upper_value=None, decimals=1):
        super().__init__(parent)
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.lower_value = lower_value if lower_value is not None else self.minimum
        self.upper_value = upper_value if upper_value is not None else self.maximum
        self.decimals = decimals
        
        # Create the dual slider
        self.slider = QRangeSlider(Qt.Horizontal)
        self.slider.setMinimum(self.minimum)
        self.slider.setMaximum(self.maximum)
        self.slider.setValue((self.lower_value, self.upper_value))
        
        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Labels
        self.lower_label = QLabel(f"Min: {self.lower_value:.{self.decimals}f}")
        self.upper_label = QLabel(f"Max: {self.upper_value:.{self.decimals}f}")
        
        # Layout
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.lower_label)
        h_layout.addWidget(self.slider)
        h_layout.addWidget(self.upper_label)
        
        layout.addLayout(h_layout)
        
        # Connect signals
        self.slider.valueChanged.connect(self.on_range_changed)
        
    def on_range_changed(self, value):
        """Handle range changes."""
        self.lower_value, self.upper_value = value
        self.lower_label.setText(f"Min: {self.lower_value:.{self.decimals}f}")
        self.upper_label.setText(f"Max: {self.upper_value:.{self.decimals}f}")
        self.rangeChanged.emit(self.lower_value, self.upper_value)
        
    def get_range(self):
        """Get the current range."""
        return (self.lower_value, self.upper_value)
        
    def set_range(self, lower, upper):
        """Set the range."""
        self.slider.setValue((lower, upper))
        self.lower_value, self.upper_value = lower, upper
        self.lower_label.setText(f"Min: {lower:.{self.decimals}f}")
        self.upper_label.setText(f"Max: {upper:.{self.decimals}f}")
