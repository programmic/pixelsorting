from PySide6.QtWidgets import QListWidget, QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, Signal, QMimeData, QPoint, QUrl
from PySide6.QtGui import QDrag, QPixmap, QDropEvent, QDragEnterEvent
from PIL.ImageQt import ImageQt
from PIL import Image
from .previewManager import preview_manager
import os

class ImportedImagesListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)  # Enable drop support
        self.images = {}  # Store PIL images by filename
        self.setStyleSheet("""
            QListWidget {
                border: 1px solid #ccc;
                border-radius: 4px;
                background: white;
            }
            QListWidget::item {
                padding: 4px;
                border-bottom: 1px solid #eee;
                color: black;
            }
            QListWidget::item:hover {
                background: #f0f0f0;
            }
            QListWidget::item:selected {
                background: #e0e0e0;
                color: black;
            }
        """)
        
    def addImage(self, filename, image):
        """Add an image to the list and store it."""
        self.images[filename] = image
        self.addItem(filename)
        
    def mousePressEvent(self, event):
        """Handle drag start."""
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            item = self.itemAt(event.pos())
            if item and item.text() in self.images:
                # Create drag
                drag = QDrag(self)
                mime_data = QMimeData()
                mime_data.setText(item.text())
                drag.setMimeData(mime_data)
                
                # Show message about drag starting
                parent = self.parent()
                while parent and not hasattr(parent, 'show_message'):
                    parent = parent.parent()
                if parent and hasattr(parent, 'show_message'):
                    parent.show_message(f"Dragging image '{item.text()}'")
                
                # Create preview pixmap for drag
                image = self.images[item.text()]
                pixmap = QPixmap.fromImage(ImageQt(image))
                scaled_pixmap = pixmap.scaled(32, 32, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                drag.setPixmap(scaled_pixmap)
                drag.setHotSpot(QPoint(16, 16))
                
                # Start drag
                drag.exec_(Qt.CopyAction)

    def mouseMoveEvent(self, event):
        """Handle preview on hover."""
        super().mouseMoveEvent(event)
        item = self.itemAt(event.pos())
        
        if item and item.text() in self.images:
            # Create content provider for preview manager
            class ImageContentProvider:
                def __init__(self, images, filename):
                    self.images = images
                    self.filename = filename
                    
                def get_image(self):
                    return self.images.get(self.filename)
                    
                def get_info(self):
                    return f"Image: {self.filename}"
            
            provider = ImageContentProvider(self.images, item.text())
            pos = self.mapToGlobal(event.pos())
            preview_manager.show_preview(provider, pos)
        else:
            preview_manager.hide_preview(delay=100)

    def leaveEvent(self, event):
        """Handle mouse leave."""
        super().leaveEvent(event)
        preview_manager.hide_preview(delay=100)
        
    def _hide_preview(self):
        """Hide the preview widget."""
        preview_manager.hide_preview()

    def dragEnterEvent(self, event):
        """Handle drag enter events for file drops."""
        if event.mimeData().hasUrls():
            # Check if any URLs are image files
            urls = event.mimeData().urls()
            for url in urls:
                if url.isLocalFile():
                    file_path = url.toLocalFile()
                    if self._is_valid_image_file(file_path):
                        event.acceptProposedAction()
                        return
        event.ignore()

    def dragMoveEvent(self, event):
        """Handle drag move events."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Handle drop events for file drops."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            imported_count = 0
            failed_files = []
            
            for url in urls:
                if url.isLocalFile():
                    file_path = url.toLocalFile()
                    if self._is_valid_image_file(file_path):
                        try:
                            image = Image.open(file_path)
                            filename = os.path.basename(file_path)
                            self.images[filename] = image
                            self.addItem(filename)
                            imported_count += 1
                        except Exception as e:
                            failed_files.append(f"{filename}: {str(e)}")
            
            # Show feedback
            parent = self.parent()
            while parent and not hasattr(parent, 'show_message'):
                parent = parent.parent()
            
            if imported_count > 0:
                message = f"Successfully imported {imported_count} image(s)"
                if failed_files:
                    message += f", {len(failed_files)} failed"
                if parent and hasattr(parent, 'show_message'):
                    parent.show_message(message)
            
            if failed_files and parent and hasattr(parent, 'show_message'):
                parent.show_message(f"Failed to import: {', '.join(failed_files)}")
                
            event.acceptProposedAction()
        else:
            event.ignore()

    def _is_valid_image_file(self, file_path):
        """Check if a file is a valid image file."""
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif', '.webp'}
        _, ext = os.path.splitext(file_path.lower())
        return ext in valid_extensions
