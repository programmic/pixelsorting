import sys
from PIL import Image
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QPushButton,
    QFileDialog,
    QSpacerItem,
    QSizePolicy
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, QObject, Signal, QThread
import superqt
import os
import main

class ImageDropLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText("Ziehe ein Bild hierher")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("QLabel { border: 2px dashed #aaa; padding: 20px; }")
        self.setAcceptDrops(True)
        self.image_pixmap = None  

    def dragEnterEvent(self, event): 
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) == 1 and urls[0].toLocalFile().lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                event.acceptProposedAction()
            else:
                event.ignore()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            image_path = urls[0].toLocalFile()
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                self.image_pixmap = pixmap  
                self.setPixmap(pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            else:
                self.setText("Ungültiges Bildformat")

class Worker(QObject):
    finished = Signal()
    progress = Signal(str)  # Optional für Fortschritt-Logging

    def __init__(self, image, settings):
        super().__init__()
        self.image = image
        self.settings = settings

    def run(self):
        import main  # sicherstellen, dass du nicht im globalen Namespace importierst
        main.process_image(
            image=self.image,
            contrastLimLower=self.settings['contrastLimLower'],
            contrastLimUpper=self.settings['contrastLimUpper'],
            sortMode=self.settings['sortMode'],
            inverse=self.settings['inverse'],
            useVerticalSplitting=self.settings['useVerticalSplitting'],
            rotateImage=self.settings['rotateImage'],
            save_output=self.settings['save_output'],
            show_output=self.settings['show_output'],
            exportPath=self.settings['export_path']
        )
        self.finished.emit()

class PreviewWorker(QObject):
    finished = Signal()
    resultReady = Signal(Image.Image)

    def __init__(self, image, contrastLimLower, contrastLimUpper):
        super().__init__()
        self.image = image
        self.low = contrastLimLower
        self.high = contrastLimUpper

    def run(self):
        from main import passes  # lokaler Import, um Zyklus zu vermeiden
        result = passes.contrastMask(self.image, self.low, self.high)
        self.resultReady.emit(result)
        self.finished.emit()

class GUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pixel sorter")
        self.resize(600, 600)

        self.drop_label = ImageDropLabel(self)

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(15,15,15,15)

        # Slider
        self.slider = superqt.QLabeledRangeSlider(Qt.Orientation.Horizontal)
        self.slider.setBarIsRigid(True)
        self.slider.setEdgeLabelMode(0)
        self.slider.setRange(0,255)
        self.slider.setValue((30, 130))
        self.layout.addWidget(self.slider)

        self.previewMask = QPushButton("Preview luminance mask")
        self.previewMask.clicked.connect(self.runLumMaskPreview)
        self.layout.addWidget(self.previewMask)

        # Select box for sorting mode
        self.sortMode = QComboBox()
        self.sortMode.addItems(["lum", "hue", "r", "g", "b"])
        self.layout.addWidget(self.sortMode)

        # Toggles
        self.lToggles = QVBoxLayout()
        self.lToggles.setSpacing(0)
        
        self.lRotate = QHBoxLayout()
        self.sRotate = superqt.QToggleSwitch()
        self.tRotate = QLabel("Rotate Image")
        self.sRotate.setChecked(True)
        self.lRotate.addWidget(self.sRotate)
        self.lRotate.addWidget(self.tRotate)
        self.lToggles.addLayout(self.lRotate)
        
        self.lMirror = QHBoxLayout()
        self.sMirror = superqt.QToggleSwitch()
        self.tMirror = QLabel("Mirror Image")
        self.lMirror.addWidget(self.sMirror)
        self.lMirror.addWidget(self.tMirror)
        self.lToggles.addLayout(self.lMirror)
        
        self.lvSplitting = QHBoxLayout()
        self.svSplitting = superqt.QToggleSwitch()
        self.svSplitting.setChecked(True)
        self.tvSplitting = QLabel("Use vertical splitting")
        self.lvSplitting.addWidget(self.svSplitting)
        self.lvSplitting.addWidget(self.tvSplitting)
        self.lToggles.addLayout(self.lvSplitting)
        
        self.lShow = QHBoxLayout()
        self.sShow = superqt.QToggleSwitch()
        self.tShow = QLabel("Show processed image")
        self.lShow.addWidget(self.sShow)
        self.lShow.addWidget(self.tShow)
        self.lToggles.addLayout(self.lShow)
        self.sShow.setChecked(True)
        
        self.lExport = QHBoxLayout()
        self.sExport = superqt.QToggleSwitch()
        self.tExport = QLabel("Export processed image")
        self.lExport.addWidget(self.sExport)
        self.lExport.addWidget(self.tExport)
        self.lToggles.addLayout(self.lExport)

        self.verticalSpacer0 = QSpacerItem(5, 15, QSizePolicy.Minimum)
        self.lToggles.addItem(self.verticalSpacer0)


        self.lShowMask = QHBoxLayout()
        self.sShowMask = superqt.QToggleSwitch()
        self.sShowMask.setChecked(True)
        self.tShowMask = QLabel("Show masks")
        self.lShowMask.addWidget(self.sShowMask)
        self.lShowMask.addWidget(self.tShowMask)
        self.lToggles.addLayout(self.lShowMask)
        
        self.layout.addLayout(self.lToggles)
        
        self.verticalSpacer1 = QSpacerItem(5, 15, QSizePolicy.Minimum)
        self.lToggles.addItem(self.verticalSpacer1)

        self.qL = QHBoxLayout()
        self.exportPath = QFileDialog()
        self.selectExportPath = QPushButton("Select export directory")
        self.selectExportPath.clicked.connect(self.selectExportDir)
        self.qL.addWidget(self.selectExportPath)

        self.outputText = superqt.QElidingLabel()
        self.outputText.setText("ehfiopuehiouhaioeughaoieghuoiaeughsi")
        self.qL.addWidget(self.outputText)

        self.layout.addLayout(self.qL)


        self.exportPathStr = f"C:/Users/{os.getlogin()}/Downloads"
        self.lCurDir = QLabel(f"Current Directory: {self.exportPathStr}")
        self.lCurDir.setFixedHeight(14)
        self.layout.addWidget(self.lCurDir)

        self.sExport.toggled.connect(self.onExportToggled)
        self.onExportToggled(self.sExport.isChecked())



        # Drag-and-drop area
        self.layout.addWidget(self.drop_label)
        
        # Run button
        self.bRun = QPushButton("Run with selected settings")
        self.bRun.clicked.connect(self.runProcessing)
        self.bRun.setFixedHeight(70)
        self.layout.addWidget(self.bRun)

        self.setLayout(self.layout)

    def selectExportDir(self):
        self.exportPathStr = self.exportPath.getExistingDirectory(self, 'Select Folder', f"C:/Users/{os.getlogin()}/Downloads")
        self.lCurDir.setText(f"Current Directory: {self.exportPathStr}")

    def onExportToggled(self, checked):
        self.selectExportPath.setEnabled(checked)

    def runLumMaskPreview(self):
        if self.drop_label.image_pixmap is None:
            self.drop_label.setText("Kein Bild ausgewählt!")
            return

        self.bRun.setEnabled(False)
        self.previewMask.setEnabled(False)
        self.drop_label.image_pixmap.save("temp_preview_input.png")
        image = Image.open("temp_preview_input.png").convert("RGB")
        low, high = self.slider.value()

        self.preview_thread = QThread()
        self.preview_worker = PreviewWorker(image, low, high)
        self.preview_worker.moveToThread(self.preview_thread)

        self.preview_thread.started.connect(self.preview_worker.run)
        self.preview_worker.finished.connect(self.preview_thread.quit)
        self.preview_worker.finished.connect(self.preview_worker.deleteLater)
        self.preview_thread.finished.connect(self.preview_thread.deleteLater)

        self.preview_worker.resultReady.connect(self.showPreviewImage)
        self.preview_worker.finished.connect(self.onPreviewGenerationFinished)

        self.preview_thread.start()

    
    def onPreviewGenerationFinished(self):
        self.previewMask.setEnabled(True)
        self.bRun.setEnabled(True)

    def runProcessing(self):
        if self.drop_label.image_pixmap is None:
            self.drop_label.setText("Kein Bild ausgewählt!")
            return

        # GUI sperren
        self.bRun.setEnabled(False)
        self.previewMask.setEnabled(False)

        # Bild speichern
        self.drop_label.image_pixmap.save("temp_input.png")
        image = Image.open("temp_input.png").convert("RGB")

        settings = {
            "contrastLimLower": self.slider.value()[0],
            "contrastLimUpper": self.slider.value()[1],
            "sortMode": self.sortMode.currentText(),
            "inverse": self.sMirror.isChecked(),
            "useVerticalSplitting": self.svSplitting.isChecked(),
            "rotateImage": self.sRotate.isChecked(),
            "save_output": self.sExport.isChecked(),
            "show_output": self.sShow.isChecked(),
            "show_render_masks": self.sShowMask.isChecked(),
            "export_path": self.exportPathStr
        }

        self.thread = QThread()
        self.worker = Worker(image, settings)
        self.worker.moveToThread(self.thread)

        # Signals verbinden
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.finished.connect(self.onProcessingFinished)

        # Thread starten
        self.thread.start()

    def onProcessingFinished(self):
        self.bRun.setEnabled(True)
        self.previewMask.setEnabled(True)

    def showPreviewImage(self, pil_image):
        from PIL.ImageQt import ImageQt
        from PySide6.QtWidgets import QDialog, QVBoxLayout

        qimage = ImageQt(pil_image)
        pixmap = QPixmap.fromImage(qimage)

        dialog = QDialog(self)
        dialog.setWindowTitle("Luminance Mask Preview")
        layout = QVBoxLayout()
        label = QLabel()
        label.setPixmap(pixmap.scaled(500, 500, Qt.AspectRatioMode.KeepAspectRatio))
        layout.addWidget(label)
        dialog.setLayout(layout)
        dialog.exec()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GUI()
    window.show()
    sys.exit(app.exec())
