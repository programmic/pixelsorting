# modernSlotPreviewWidget.py

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QGraphicsDropShadowEffect,
                              QHBoxLayout, QFrame)
from PySide6.QtCore import Qt, QPoint, QPropertyAnimation, QRect, QSize
from PySide6.QtGui import (QPixmap, QPainter, QPainterPath, QBrush, QColor, QPen,
                          QLinearGradient, QFont, QFontMetrics)
from PIL.ImageQt import ImageQt
import math


class ModernSlotPreviewWidget(QWidget):
    """
    A modern, non-flickering preview widget for slot images.
    
    Features:
    - Smooth fade-in/out animations
    - Proper positioning to avoid screen edges
    - High-DPI support
    - Modern styling with shadows
    """
    
    def __init__(self, parent=None):
        super().__init__(parent, Qt.ToolTip | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        # Remove WA_DeleteOnClose to prevent premature deletion
        
        # Animation support
        self._opacity = 0.0
        self._animation = QPropertyAnimation(self, b"windowOpacity")
        self._animation.setDuration(150)  # Smooth fade
        self._animation_active = False
        
        # Modern styling
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the modern UI."""
        self.setFixedSize(220, 220)
        
        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(8, 8, 8, 8)
        
        # Image label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #2d2d2d;
                border: 1px solid #555;
                border-radius: 4px;
            }
        """)
        self.image_label.setFixedSize(200, 160)
        
        # Info label
        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("""
            QLabel {
                color: #cccccc;
                font-size: 11px;
                padding: 4px;
                background-color: rgba(45, 45, 45, 180);
                border-radius: 4px;
            }
        """)
        
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.info_label)
        
        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 100))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)
        
    def updateContent(self, image=None, render_pass_info=None):
        """Update the preview content."""
        if image:
            # Convert PIL image to QPixmap
            qim = ImageQt(image)
            pixmap = QPixmap.fromImage(qim)
            
            # Scale to fit while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                190, 150,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.show()
        else:
            self.image_label.hide()
            
        if render_pass_info:
            self.info_label.setText(f"Will be produced by:\n{render_pass_info}")
            self.info_label.show()
        else:
            self.info_label.hide()
            
    def show_at_position(self, parent_widget, local_pos):
        """Show the preview at the correct position avoiding screen edges."""
        # Convert to global coordinates
        global_pos = parent_widget.mapToGlobal(local_pos)
        
        # Get screen geometry
        screen = parent_widget.screen()
        screen_rect = screen.availableGeometry()
        
        # Calculate position
        preview_width = self.width()
        preview_height = self.height()
        
        # Default position: to the right of cursor
        x = global_pos.x() + 15
        y = global_pos.y() - preview_height // 2
        
        # Adjust if off screen
        if x + preview_width > screen_rect.right():
            x = global_pos.x() - preview_width - 15
            
        if y < screen_rect.top():
            y = screen_rect.top() + 10
            
        if y + preview_height > screen_rect.bottom():
            y = screen_rect.bottom() - preview_height - 10
            
        # Position and show with animation
        self.move(x, y)
        self.show()
        
        # Fade in
        self._animation.setStartValue(0.0)
        self._animation.setEndValue(1.0)
        self._animation_active = True
        self._animation.start()
        
    def hide(self):
        """Hide with fade out animation."""
        if not self.isVisible() or self._animation_active:
            return
            
        self._animation_active = True
        self._animation.setStartValue(1.0)
        self._animation.setEndValue(0.0)
        
        # Disconnect any existing connections first
        try:
            self._animation.finished.disconnect()
        except:
            pass
            
        self._animation.finished.connect(self._close_after_animation)
        self._animation.start()
        
    def _close_after_animation(self):
        """Close after fade out with safety checks."""
        try:
            # Check if widget still exists
            if self and self.isVisible():
                super().hide()
                # Don't call close() to prevent deletion issues
                # Let parent manage lifecycle
        except (RuntimeError, AttributeError):
            # Widget already deleted, ignore
            pass
        finally:
            self._animation_active = False
            
    def paintEvent(self, event):
        """Custom paint for rounded corners."""
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Draw background
            path = QPainterPath()
            path.addRoundedRect(self.rect(), 8, 8)
            
            # Gradient background
            gradient = QLinearGradient(0, 0, 0, self.height())
            gradient.setColorAt(0, QColor(35, 35, 35, 240))
            gradient.setColorAt(1, QColor(25, 25, 25, 240))
            
            painter.fillPath(path, QBrush(gradient))
            
            # Border
            painter.setPen(QPen(QColor(100, 100, 100, 200), 1))
            painter.drawPath(path)
        except (RuntimeError, AttributeError):
            # Widget already deleted, ignore paint errors
            pass
            
    def sizeHint(self):
        """Provide proper size hint."""
        return QSize(220, 220)
        
    def __del__(self):
        """Cleanup when widget is destroyed."""
        try:
            if hasattr(self, '_animation'):
                self._animation.stop()
                self._animation.finished.disconnect()
        except:
            pass
