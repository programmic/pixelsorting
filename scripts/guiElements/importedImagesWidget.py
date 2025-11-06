from PySide6.QtWidgets import QListWidget, QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, Signal, QMimeData, QPoint, QUrl
from PySide6.QtGui import QDrag, QPixmap, QDropEvent, QDragEnterEvent
from PIL.ImageQt import ImageQt
from PIL import Image
from guiElements.previewManager import preview_manager
import os

class ImportedImagesListWidget(QListWidget):
    def dragEnterEvent(self, event):
        """Highlight widget on drag enter if valid."""
        if event.mimeData().hasUrls() or event.mimeData().hasText() or event.mimeData().hasFormat("image/png"):
            event.acceptProposedAction()
            self.setStyleSheet(self.styleSheet() + "\nQListWidget { border: 2px solid #4caf50; background: #e8f5e9; }")
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        """Keep highlight during drag move if valid."""
        if event.mimeData().hasUrls() or event.mimeData().hasText() or event.mimeData().hasFormat("image/png"):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        """Remove highlight on drag leave."""
        self.setStyleSheet(self.styleSheet().replace("\nQListWidget { border: 2px solid #4caf50; background: #e8f5e9; }", ""))
        event.accept()

    def dropEvent(self, event):
        """Handle drops: accept image files, URLs, or raw image bytes.

        Adds dropped images to this widget and to the top-level GUI
        `imported_images` dictionary when available.
        """
        self.setStyleSheet(self.styleSheet().replace("\nQListWidget { border: 2px solid #4caf50; background: #e8f5e9; }", ""))
        mime = event.mimeData()
        handled = False

        # 1. URLs (file paths)
        if mime.hasUrls():
            urls = mime.urls()
            for url in urls:
                local = url.toLocalFile()
                if local and os.path.isfile(local) and self._is_valid_image_file(local):
                    try:
                        img = Image.open(local)
                        filename = os.path.basename(local)
                        # Ensure unique key
                        key = filename
                        i = 1
                        while key in self.images:
                            key = f"{os.path.splitext(filename)[0]}_{i}{os.path.splitext(filename)[1]}"
                            i += 1
                        self.addImage(key, img)
                        # Also register in main GUI if possible
                        parent = self.parent()
                        while parent and not hasattr(parent, 'imported_images'):
                            parent = parent.parent()
                        if parent and hasattr(parent, 'imported_images'):
                            parent.imported_images[key] = img
                        handled = True
                    except Exception as e:
                        print(f"Error opening dropped image file: {e}")

        # 2. Raw image data (e.g., image/png)
        if not handled and mime.hasFormat('image/png'):
            try:
                data = mime.data('image/png')
                from io import BytesIO
                img = Image.open(BytesIO(data))
                # Create a synthetic name
                key = f"dropped_image_{len(self.images) + 1}.png"
                self.addImage(key, img)
                parent = self.parent()
                while parent and not hasattr(parent, 'imported_images'):
                    parent = parent.parent()
                if parent and hasattr(parent, 'imported_images'):
                    parent.imported_images[key] = img
                handled = True
            except Exception as e:
                print(f"Error loading image bytes from drop: {e}")

        # 3. Text (filename) fallback
        if not handled and mime.hasText():
            text = mime.text().strip()
            if os.path.isfile(text) and self._is_valid_image_file(text):
                try:
                    img = Image.open(text)
                    filename = os.path.basename(text)
                    self.addImage(filename, img)
                    parent = self.parent()
                    while parent and not hasattr(parent, 'imported_images'):
                        parent = parent.parent()
                    if parent and hasattr(parent, 'imported_images'):
                        parent.imported_images[filename] = img
                    handled = True
                except Exception as e:
                    print(f"Error opening image from text drop: {e}")

        if handled:
            event.acceptProposedAction()
        else:
            event.ignore()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)  # Enable drop support
        # Use per-pixel scrolling for smoother UX
        try:
            from PySide6.QtWidgets import QAbstractItemView
            self.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
            self.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        except Exception:
            pass
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
        """Handle drag start with richer MIME data."""
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            item = self.itemAt(event.pos())
            if item and item.text() in self.images:
                drag = QDrag(self)
                mime_data = QMimeData()
                filename = item.text()
                mime_data.setText(filename)
                # Set a file URL for compatibility
                import os
                from PySide6.QtCore import QUrl
                file_path = os.path.abspath(filename)
                mime_data.setUrls([QUrl.fromLocalFile(file_path)])
                # Add image data as bytes (PNG format)
                from io import BytesIO
                image = self.images[filename]
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                mime_data.setData("image/png", buffer.getvalue())
                drag.setMimeData(mime_data)
                # Show message about drag starting
                parent = self.parent()
                while parent and not hasattr(parent, 'show_message'):
                    parent = parent.parent()
                if parent and hasattr(parent, 'show_message'):
                    parent.show_message(f"Dragging image '{filename}'")
                # Create preview pixmap for drag
                pixmap = QPixmap.fromImage(ImageQt(image))
                scaled_pixmap = pixmap.scaled(32, 32, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                drag.setPixmap(scaled_pixmap)
                drag.setHotSpot(QPoint(16, 16))
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


    def _is_valid_image_file(self, file_path):
        """Check if a file is a valid image file."""
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif', '.webp'}
        _, ext = os.path.splitext(file_path.lower())
        return ext in valid_extensions
