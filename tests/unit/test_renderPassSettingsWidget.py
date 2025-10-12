import os
import sys
import unittest

# Ensure project scripts are importable
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from PySide6.QtCore import Qt

from scripts.guiElements.renderPassSettingsWidget import RenderPassSettingsWidget


class TestRenderPassSettingsWidget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not QApplication.instance():
            cls.app = QApplication([])
        else:
            cls.app = QApplication.instance()

    def test_multiple_sliders_plus_minus_independent(self):
        # Prepare a settings config with two sliders
        config = [
            {"label": "Slider A", "type": "slider", "min": 0, "max": 10, "default": 2, "name": "a"},
            {"label": "Slider B", "type": "slider", "min": 0, "max": 10, "default": 5, "name": "b"}
        ]

        widget = RenderPassSettingsWidget(config)

        # Retrieve controls
        controls = widget.controls
        self.assertIn('a', controls)
        self.assertIn('b', controls)

        ctrl_a = controls['a']
        ctrl_b = controls['b']
        self.assertIn('slider', ctrl_a)
        self.assertIn('slider', ctrl_b)

        slider_a = ctrl_a['slider']
        slider_b = ctrl_b['slider']

        # Find +/- buttons within the widget hierarchy by searching for QPushButton children
        container_a = ctrl_a['_widget']
        container_b = ctrl_b['_widget']
        from PySide6.QtWidgets import QPushButton
        buttons_a = container_a.findChildren(QPushButton)
        buttons_b = container_b.findChildren(QPushButton)

        self.assertGreaterEqual(len(buttons_a), 2)
        self.assertGreaterEqual(len(buttons_b), 2)

        # We assume order: [minus, plus]
        minus_a, plus_a = buttons_a[0], buttons_a[1]
        minus_b, plus_b = buttons_b[0], buttons_b[1]

        # Record initial values
        val_a0 = slider_a.value()
        val_b0 = slider_b.value()

        # Click plus on A twice
        QTest.mouseClick(plus_a, Qt.LeftButton)
        QTest.mouseClick(plus_a, Qt.LeftButton)

        # Click minus on B once
        QTest.mouseClick(minus_b, Qt.LeftButton)

        # Determine expected changes based on each slider's singleStep()
        step_a = slider_a.singleStep()
        step_b = slider_b.singleStep()
        expected_a = min(val_a0 + 2 * step_a, slider_a.maximum())
        expected_b = max(val_b0 - step_b, slider_b.minimum())

        # Assert values changed independently (allow for floating-point rounding)
        self.assertAlmostEqual(slider_a.value(), expected_a, places=6)
        self.assertAlmostEqual(slider_b.value(), expected_b, places=6)


if __name__ == '__main__':
    unittest.main()
