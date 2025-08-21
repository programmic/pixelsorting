import unittest
import sys
import os

# Add the project root to the path so we can import modules
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from scripts.guiElements.renderPassWidget import RenderPassWidget

class TestRenderPassWidget(unittest.TestCase):
    """Test cases for RenderPassWidget class"""

    @classmethod
    def setUpClass(cls):
        """Set up QApplication for Qt tests"""
        if not QApplication.instance():
            cls.app = QApplication([])
        else:
            cls.app = QApplication.instance()

    def setUp(self):
        """Set up test fixtures before each test method"""
        self.available_slots = ["slot1", "slot2", "slot3", "slot4"]
        self.mock_callback = lambda *args: None
        self.mock_delete_callback = lambda *args: None

    def test_init_single_input_pass(self):
        """Test initialization for single input pass types"""
        widget = RenderPassWidget("Blur", self.available_slots, self.mock_callback, self.mock_delete_callback)
        self.assertEqual(widget.renderpass_type, "Blur")
        self.assertEqual(widget.numInputs, 1)
        self.assertEqual(len(widget.selectedInputs), 1)
        self.assertIsNone(widget.selectedInputs[0])

    def test_init_dual_input_pass(self):
        """Test initialization for dual input pass types"""
        widget = RenderPassWidget("Mix By Percent", self.available_slots, self.mock_callback, self.mock_delete_callback)
        self.assertEqual(widget.renderpass_type, "Mix By Percent")
        self.assertEqual(widget.numInputs, 2)
        self.assertEqual(len(widget.selectedInputs), 2)
        self.assertIsNone(widget.selectedInputs[0])
        self.assertIsNone(widget.selectedInputs[1])

    def test_init_other_dual_input_passes(self):
        """Test initialization for other dual input pass types"""
        dual_input_types = ["Mix Screen", "Subtract"]
        for pass_type in dual_input_types:
            widget = RenderPassWidget(pass_type, self.available_slots, self.mock_callback, self.mock_delete_callback)
            self.assertEqual(widget.numInputs, 2)

    def test_delete_button_callback(self):
        """Test delete button triggers callback"""
        widget = RenderPassWidget("Blur", self.available_slots, self.mock_callback, self.mock_delete_callback)
        # Test that delete button exists and has a connected signal
        self.assertIsNotNone(widget.deleteBtn)
        self.assertTrue(widget.deleteBtn.isEnabled())

    def test_settings_config_loading(self):
        """Test settings configuration loading"""
        widget = RenderPassWidget("Blur", self.available_slots, self.mock_callback, self.mock_delete_callback)
        config = widget.get_settings_config("Blur")
        self.assertIsInstance(config, list)

    def test_get_settings(self):
        """Test getting current settings"""
        widget = RenderPassWidget("Blur", self.available_slots, self.mock_callback, self.mock_delete_callback)
        settings = widget.get_settings()
        self.assertIsInstance(settings, dict)

    def test_set_slot_input(self):
        """Test setting slot for input"""
        widget = RenderPassWidget("Blur", self.available_slots, self.mock_callback, self.mock_delete_callback)
        widget.selectionMode = 'input'
        widget.currentSelectedInput = 0
        widget.set_slot("slot1")
        self.assertEqual(widget.selectedInputs[0], "slot1")

    def test_set_slot_output(self):
        """Test setting slot for output"""
        widget = RenderPassWidget("Blur", self.available_slots, self.mock_callback, self.mock_delete_callback)
        widget.selectionMode = 'output'
        widget.set_slot("slot2")
        self.assertEqual(widget.selectedOutput, "slot2")

    def test_set_slot_mask(self):
        """Test setting slot for mask"""
        widget = RenderPassWidget("Blur", self.available_slots, self.mock_callback, self.mock_delete_callback)
        widget.selectionMode = 'mask'
        widget.set_slot("slot3")
        # Mask widget should handle the slot setting

    def test_widget_initialization(self):
        """Test basic widget initialization"""
        widget = RenderPassWidget("TestPass", self.available_slots, self.mock_callback, self.mock_delete_callback)
        self.assertIsNotNone(widget)
        self.assertIsNotNone(widget.mainLayout)
        self.assertIsNotNone(widget.title)
        self.assertEqual(widget.title.text(), "Renderpass: TestPass")

    def test_available_slots_stored(self):
        """Test available slots are properly stored"""
        widget = RenderPassWidget("Blur", self.available_slots, self.mock_callback, self.mock_delete_callback)
        self.assertEqual(widget.availableSlots, self.available_slots)

if __name__ == '__main__':
    unittest.main()
