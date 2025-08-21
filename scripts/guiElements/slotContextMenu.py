from PySide6.QtWidgets import QMenu
from PySide6.QtGui import QAction
from PySide6.QtCore import Signal, QObject


class SlotContextMenu(QObject):
    """Context menu for slot right-click operations."""
    
    slot_learn_requested = Signal(str)
    slot_clear_requested = Signal(str)
    slot_preview_requested = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
    def show_menu(self, slot_name, global_pos):
        """Show context menu for a slot."""
        menu = QMenu()
        
        # Learn slot action
        learn_action = QAction("Learn Slot", menu)
        learn_action.triggered.connect(lambda: self.slot_learn_requested.emit(slot_name))
        menu.addAction(learn_action)
        
        # Clear slot action
        clear_action = QAction("Clear Slot", menu)
        clear_action.triggered.connect(lambda: self.slot_clear_requested.emit(slot_name))
        menu.addAction(clear_action)
        
        # Preview slot action
        preview_action = QAction("Preview Slot", menu)
        preview_action.triggered.connect(lambda: self.slot_preview_requested.emit(slot_name))
        menu.addAction(preview_action)
        
        menu.exec(global_pos)
