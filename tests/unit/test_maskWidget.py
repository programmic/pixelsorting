import unittest
from PySide6.QtWidgets import QApplication
from guiElements.maskWidget import MaskWidget

class TestMaskWidget(unittest.TestCase):
    """Test cases for MaskWidget class"""

    @classmethod
    def setUpClass(cls):
        """Set up QApplication for Qt tests"""
        if not QApplication.instance():
            cls.app = QApplication([])

    def setUp(self):
        """Set up test fixtures before each test method"""
        self.available_slots = ["slot1", "slot2", "slot3"]
        self.mock_callback = lambda *args: None
        self.widget = MaskWidget(self.available_slots, self.mock_callback)

    def test_toggle_switch(self):
        """Test enabling and disabling the mask"""
        self.widget.enabled.setChecked(True)
        self.assertTrue(self.widget.enabled.isChecked(), "Mask should be enabled")

        self.widget.enabled.setChecked(False)
        self.assertFalse(self.widget.enabled.isChecked(), "Mask should be disabled")

    def test_slot_selection(self):
        """Test slot selection callback"""
        self.widget._on_select_clicked()
        # Check if the callback was called with the correct argument
        # (This would require a more complex setup with a mock callback)

    def test_get_values(self):
        """Test get_values method"""
        self.widget.enabled.setChecked(True)
        self.widget.set_slot("slot1")
        values = self.widget.get_values()
        self.assertEqual(values, {"enabled": True, "slot": "slot1"}, "get_values() should return correct data")

    def test_load_settings(self):
        """Test loading settings into the widget"""
        settings = {"enabled": True, "slot": "slot1"}
        self.widget.load_settings(settings)
        self.assertTrue(self.widget.enabled.isChecked(), "Mask should be enabled after loading settings")
        self.assertEqual(self.widget.selected_slot, "slot1", "Slot should be set correctly after loading settings")

if __name__ == '__main__':
    unittest.main()
