from __future__ import annotations
from PySide6.QtCore import QObject, Signal
import renderHook
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from masterGUI import GUI

class RenderWorker(QObject):
    finished = Signal()
    progress = Signal(str)
    error = Signal(str)

    def __init__(self, gui_instance: 'GUI'):
        super().__init__()
        self.gui = gui_instance

    def run(self):
        try:
            renderHook.run_all_render_passes(self.gui, self.progress.emit)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit()
