from PySide6.QtCore import QObject, Signal
import renderHook

class RenderWorker(QObject):
    finished = Signal()
    progress = Signal(str)
    error = Signal(str)

    def __init__(self, gui_instance):
        super().__init__()
        self.gui = gui_instance

    def run(self):
        try:
            renderHook.run_all_render_passes(self.gui, self.progress.emit)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit()
