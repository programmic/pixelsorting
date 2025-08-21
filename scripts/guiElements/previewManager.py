from PySide6.QtCore import QObject, QTimer, QPoint, QThread
from PySide6.QtWidgets import QApplication
from .modernSlotPreviewWidget import ModernSlotPreviewWidget
import weakref


class PreviewManager(QObject):
    """
    Centralized manager for hover preview widgets to prevent lifecycle issues
    and race conditions.
    """
    
    _instance = None
    
    def __init__(self):
        super().__init__()
        self._preview_widget = None
        self._hide_timer = QTimer()
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._delayed_hide)
        self._current_widget_ref = None
        self._is_showing = False
        
    @classmethod
    def instance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    def get_preview_widget(self):
        """Get or create the preview widget."""
        if self._preview_widget is None:
            self._preview_widget = ModernSlotPreviewWidget()
            
        # Check if widget is still valid
        try:
            self._preview_widget.isVisible()
        except (RuntimeError, AttributeError):
            # Widget was deleted, recreate
            self._preview_widget = ModernSlotPreviewWidget()
            
        return self._preview_widget
        
    def show_preview(self, content_provider, position, parent_widget=None):
        """Show preview with content from provider."""
        # Cancel any pending hide
        self._hide_timer.stop()
        
        preview = self.get_preview_widget()
        
        # Update content
        if hasattr(content_provider, 'get_image') and hasattr(content_provider, 'get_info'):
            image = content_provider.get_image()
            info = content_provider.get_info()
            preview.updateContent(image, info)
        
        # Position and show - use parent_widget if provided, otherwise use QApplication
        if parent_widget is None:
            parent_widget = QApplication.instance().activeWindow()
        
        if parent_widget is not None:
            preview.show_at_position(parent_widget, position)
        else:
            # Fallback: show at screen position without parent
            preview.move(position.x() + 15, position.y() - preview.height() // 2)
            preview.show()
            
        self._is_showing = True
        
    def hide_preview(self, delay=0):
        """Hide preview with optional delay."""
        if delay > 0:
            self._hide_timer.start(delay)
        else:
            self._delayed_hide()
            
    def _delayed_hide(self):
        """Actually hide the preview."""
        if self._preview_widget and self._preview_widget.isVisible():
            try:
                self._preview_widget.hide()
                self._is_showing = False
            except (RuntimeError, AttributeError):
                # Widget was deleted
                self._preview_widget = None
                
    def is_showing(self):
        """Check if preview is currently showing."""
        return self._is_showing
        
    def set_current_widget(self, widget):
        """Set the current widget that triggered the preview."""
        self._current_widget_ref = weakref.ref(widget) if widget else None
        
    def get_current_widget(self):
        """Get the current widget that triggered the preview."""
        if self._current_widget_ref:
            try:
                return self._current_widget_ref()
            except:
                return None
        return None
        
    def cleanup(self):
        """Clean up resources."""
        if self._preview_widget:
            try:
                self._preview_widget.hide()
                self._preview_widget.deleteLater()
            except:
                pass
        self._preview_widget = None
        self._hide_timer.stop()


# Global instance
preview_manager = PreviewManager.instance()
