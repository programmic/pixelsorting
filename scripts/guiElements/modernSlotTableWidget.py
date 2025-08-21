# modernSlotTableWidget.py

from PySide6.QtWidgets import (QWidget, QHBoxLayout, QPushButton, QGraphicsDropShadowEffect,
                              QStyle, QStyleOption, QStylePainter)
from PySide6.QtCore import Qt, Signal, QPoint, QSize, QTimer, QRect, QEvent
from PySide6.QtGui import (QPixmap, QPainter, QBrush, QColor, QPen, QLinearGradient,
                          QPainterPath, QFont, QFontMetrics, QPalette)
from .modernSlotPreviewWidget import ModernSlotPreviewWidget
from .slotContextMenu import SlotContextMenu
from .previewManager import preview_manager
import weakref
from PIL.ImageQt import ImageQt
import math


class ModernSlotButton(QPushButton):
    """A modern styled button for slot display with proper scaling and animations."""
    
    def __init__(self, slot_name, parent=None):
        super().__init__(parent)
        self.slot_name = slot_name
        self.setFixedSize(32, 32)  # Scaled size for better visibility
        self.setCursor(Qt.PointingHandCursor)
        self.setMouseTracking(True)  # Enable mouse tracking for hover
        
        # Animation properties
        self._hover_progress = 0.0
        self._animation_timer = QTimer()
        self._animation_timer.setInterval(16)  # 60fps
        self._animation_timer.timeout.connect(self._animate_hover)
        
        # State tracking
        self._has_image = False
        self._is_used = False
        self._is_output = False
        self._is_input = False
        
        # Remove default styling
        self.setStyleSheet("")
        
    def set_state(self, has_image=False, is_used=False, is_output=False, is_input=False):
        """Update the visual state of the button."""
        self._has_image = has_image
        self._is_used = is_used
        self._is_output = is_output
        self._is_input = is_input
        self.update()
        
    def enterEvent(self, event):
        """Start hover animation."""
        super().enterEvent(event)
        self._animation_timer.start()
        
    def leaveEvent(self, event):
        """End hover animation."""
        super().leaveEvent(event)
        self._animation_timer.stop()
        self._hover_progress = 0.0
        self.update()
        
    def _animate_hover(self):
        """Animate hover effect."""
        if self._hover_progress < 1.0:
            self._hover_progress = min(1.0, self._hover_progress + 0.1)
            self.update()
        
    def paintEvent(self, event):
        """Custom paint event with modern styling."""
        painter = QStylePainter(self)
        option = QStyleOption()
        option.initFrom(self)
        
        # Get base colors based on state
        if self.slot_name == "slot0":
            base_color = QColor("#3a75c4")  # Blue for input
        elif self.slot_name == "slot15":
            base_color = QColor("#ffb74d")  # Orange for output
        elif self._is_input:
            base_color = QColor("#3a75c4")  # Blue for input
        elif self._is_output:
            base_color = QColor("#ffb74d")  # Orange for output
        elif self._has_image:
            base_color = QColor("#4caf50")  # Green for used with image
        elif self._is_used:
            base_color = QColor("#4caf50")  # Green for used
        else:
            base_color = QColor("#e57373")  # Red for unused
        
        # Create gradient for depth
        gradient = QLinearGradient(0, 0, 0, self.height())
        if self.isDown():
            gradient.setColorAt(0, base_color.darker(120))
            gradient.setColorAt(1, base_color.darker(140))
        elif self.underMouse() or self._hover_progress > 0:
            hover_intensity = self._hover_progress * 0.2
            gradient.setColorAt(0, base_color.lighter(100 + int(hover_intensity * 50)))
            gradient.setColorAt(1, base_color)
        else:
            gradient.setColorAt(0, base_color)
            gradient.setColorAt(1, base_color.darker(110))
        
        # Draw rounded rectangle
        path = QPainterPath()
        path.addRoundedRect(QRect(1, 1, self.width() - 2, self.height() - 2), 4, 4)
        
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillPath(path, QBrush(gradient))
        
        # Draw border
        border_color = base_color.darker(150)
        if self._has_image:
            border_color = QColor("#2e7d32")  # Darker green for image border
        
        pen = QPen(border_color)
        pen.setWidth(2 if self._has_image else 1)
        painter.setPen(pen)
        painter.drawPath(path)
        
        # Draw slot number
        font = QFont()
        font.setPixelSize(10)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor("white"))
        
        # Center text
        slot_num = self.slot_name.replace("slot", "")
        metrics = QFontMetrics(font)
        text_rect = metrics.boundingRect(slot_num)
        x = (self.width() - text_rect.width()) // 2
        y = (self.height() + text_rect.height()) // 2 - 2
        
        painter.drawText(x, y, slot_num)
        
        # Draw indicator for special states
        if self._has_image:
            # Small image indicator
            indicator_rect = QRect(self.width() - 8, 2, 6, 6)
            painter.setBrush(QBrush(QColor("#2e7d32")))
            painter.setPen(QPen(QColor("#1b5e20")))
            painter.drawEllipse(indicator_rect)


class ModernSlotTableWidget(QWidget):
    """
    A modern widget that displays slots with improved rendering and interaction.
    
    :param slots: A list of slot names to display.
    :param parent: The parent widget.
    """
    slot_clicked = Signal(str)
    image_dropped = Signal(str, object)  # slot_name, image
    
    def __init__(self, slots: list[str], parent=None):
        super().__init__(parent)
        self.slots = slots
        self.slot_usage = {}
        self.slot_images = {}
        self.slot_sources = {}
        self.original_input_image = None
        
        # Context menu for right-click
        self.context_menu = SlotContextMenu(self)
        
        # Modern layout with proper spacing
        self.layout = QHBoxLayout(self)
        self.layout.setSpacing(2)
        self.layout.setContentsMargins(4, 4, 4, 4)
        
        # Create modern buttons
        self.buttons = []
        for slot_name in slots:
            btn = ModernSlotButton(slot_name)
            self.layout.addWidget(btn)
            self.buttons.append((slot_name, btn))
            btn.clicked.connect(self._make_click_handler(slot_name))
            btn.setContextMenuPolicy(Qt.CustomContextMenu)
            btn.customContextMenuRequested.connect(
                lambda pos, slot=slot_name: self._show_context_menu(slot, pos)
            )
            btn.installEventFilter(self)
            
        # Preview system - use centralized preview manager
        self.preview_timer = QTimer()

        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self._show_preview)
        self.current_preview_slot = None
        self._last_mouse_pos = None
        
        # Enable drag and drop
        self.setAcceptDrops(True)
        self.setMouseTracking(True)  # Enable mouse tracking for hover preview
        
        # Cache for images
        self._image_cache = {}
        
    def _make_click_handler(self, slot_name):
        def handler():
            self.slot_clicked.emit(slot_name)
        return handler
        
    def _show_context_menu(self, slot_name, pos):
        """Show context menu for slot right-click."""
        self.context_menu.show_menu(slot_name, self.mapToGlobal(pos))
        
    def refreshColors(self, slot_usage: dict[str, bool]):
        """Update button colors based on slot usage."""
        self.slot_usage = slot_usage
        
        for slot_name, btn in self.buttons:
            has_image = slot_name in self.slot_images
            is_used = slot_usage.get(slot_name, False)
            is_output = slot_name == "slot15"
            is_input = slot_name == "slot0"
            
            btn.set_state(
                has_image=has_image,
                is_used=is_used,
                is_output=is_output,
                is_input=is_input
            )
            
    def enterEvent(self, event):
        """Handle mouse enter for preview system."""
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        """Handle mouse leave for preview system."""
        super().leaveEvent(event)
        preview_manager.hide_preview(delay=100)  # Small delay to prevent flicker
        
    def mouseMoveEvent(self, event):
        """Handle mouse movement for preview positioning."""
        super().mouseMoveEvent(event)
        
        # Find which button is under cursor
        for slot_name, btn in self.buttons:
            if btn.underMouse():
                if self.current_preview_slot != slot_name:
                    self.preview_timer.stop()  # Stop previous timer
                    self.current_preview_slot = slot_name
                    self.preview_timer.start(500)  # Delay to prevent flicker
                return
                
        self.current_preview_slot = None

        preview_manager.hide_preview(delay=100)
        
    def _show_preview(self):
        """Show the preview widget with proper positioning."""
        if not self.current_preview_slot:
            return
            
        slot_name = self.current_preview_slot
        
        # Create content provider for preview manager
        class SlotContentProvider:
            def __init__(self, table_widget, slot_name):
                self.table_widget = table_widget
                self.slot_name = slot_name
                
            def get_image(self):
                return self.table_widget.get_image(self.slot_name)
                
            def get_info(self):
                return self.table_widget.slot_sources.get(self.slot_name)
        
        provider = SlotContentProvider(self, slot_name)
        cursor_pos = self.mapFromGlobal(self.cursor().pos())
        preview_manager.show_preview(provider, cursor_pos, self)
        
    def _hide_preview(self):
        """Hide the preview widget."""
        preview_manager.hide_preview()
                
    def get_image(self, slot_name):
        """Get image from slot with caching."""
        if slot_name == "slot0" and self.original_input_image is not None:
            return self.original_input_image
            
        return self.slot_images.get(slot_name)
        
    def set_image(self, slot_name, image):
        """Set image for slot with caching."""
        if slot_name == "slot0":
            self.original_input_image = image
            self.slot_images[slot_name] = image
        else:
            self.slot_images[slot_name] = image
            
        # Clear source info since we have actual image
        if slot_name in self.slot_sources:
            del self.slot_sources[slot_name]
            
        self.refreshColors(self.slot_usage)
        
    def setSlotSource(self, slot_name, render_pass_info):
        """Set render pass source for slot."""
        self.slot_sources[slot_name] = render_pass_info
        
        # Clear image since it's now pending
        if slot_name in self.slot_images:
            del self.slot_images[slot_name]
            
        self.refreshColors(self.slot_usage)
        
    def dragEnterEvent(self, event):
        """Accept drag events with visual feedback."""
        if event.mimeData().hasText():
            event.acceptProposedAction()
            self.setStyleSheet("ModernSlotTableWidget { border: 2px dashed #4caf50; }")
        else:
            event.ignore()
            
    def dragLeaveEvent(self, event):
        """Reset visual feedback on drag leave."""
        super().dragLeaveEvent(event)
        self.setStyleSheet("")
        
    def dropEvent(self, event):
        """Handle drop events with proper positioning."""
        drop_pos = event.pos()
        
        # Find which button was dropped on
        for slot_name, btn in self.buttons:
            if btn.geometry().contains(drop_pos):
                filename = event.mimeData().text()
                
                # Find parent with imported_images
                parent = self.parent()
                while parent and not hasattr(parent, 'imported_images'):
                    parent = parent.parent()
                    
                if parent and hasattr(parent, 'imported_images') and filename in parent.imported_images:
                    image = parent.imported_images[filename]
                    self.set_image(slot_name, image)
                    self.image_dropped.emit(slot_name, image)
                    event.acceptProposedAction()
                else:
                    event.ignore()
                break
                
        self.setStyleSheet("")
        
    def sizeHint(self):
        """Provide proper size hint for layout."""
        return QSize(len(self.slots) * 34 + 8, 40)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Leave:
            preview_manager.hide_preview(delay=100)
        return False