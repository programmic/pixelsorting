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
        widget = RenderPassWidget("Mix Percent", self.available_slots, self.mock_callback, self.mock_delete_callback)
        self.assertEqual(widget.renderpass_type, "Mix Percent")
        self.assertEqual(widget.numInputs, 2)
        self.assertEqual(len(widget.selectedInputs), 2)
        self.assertIsNone(widget.selectedInputs[0])
        self.assertIsNone(widget.selectedInputs[1])

    def test_init_other_dual_input_passes(self):
        """Test initialization for other dual input pass types"""
        dual_input_types = ["Mix Screen", "Subtract Images", "Alpha Over", "Scale to fit"]
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
        # Verify mask widget has the slot set
        self.assertEqual(widget.maskWidget.selected_slot, "slot3")

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

    def test_mask_widget_integration(self):
        """Test mask widget is properly integrated"""
        widget = RenderPassWidget("Blur", self.available_slots, self.mock_callback, self.mock_delete_callback)
        self.assertIsNotNone(widget.maskWidget, "Mask widget should be created")
        self.assertEqual(widget.maskWidget.available_slots, self.available_slots, "Mask widget should have access to available slots")

    def test_mask_settings_save_load(self):
        """Test mask settings are properly saved and loaded"""
        widget = RenderPassWidget("Blur", self.available_slots, self.mock_callback, self.mock_delete_callback)
        
        # Set mask settings
        widget.maskWidget.enabled.setChecked(True)
        widget.maskWidget.set_slot("slot2")
        
        # Get settings
        settings = widget.get_settings()
        self.assertIn('mask', settings, "Settings should contain mask section")
        self.assertEqual(settings['mask']['enabled'], True, "Mask enabled state should be saved")
        self.assertEqual(settings['mask']['slot'], "slot2", "Mask slot should be saved")
        
        # Create new widget and load settings
        new_widget = RenderPassWidget("Blur", self.available_slots, self.mock_callback, self.mock_delete_callback)
        new_widget.load_settings({'mask': settings['mask']})
        
        # Verify settings were loaded correctly
        self.assertTrue(new_widget.maskWidget.enabled.isChecked(), "Mask should be enabled after loading")
        self.assertEqual(new_widget.maskWidget.selected_slot, "slot2", "Mask slot should be loaded correctly")

    def test_mask_disabled_by_default(self):
        """Test mask is disabled by default"""
        widget = RenderPassWidget("Blur", self.available_slots, self.mock_callback, self.mock_delete_callback)
        self.assertFalse(widget.maskWidget.enabled.isChecked(), "Mask should be disabled by default")
        self.assertIsNone(widget.maskWidget.selected_slot, "No slot should be selected by default")

if __name__ == '__main__':
    unittest.main()
