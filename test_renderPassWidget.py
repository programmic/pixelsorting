import unittest
from PySide6.QtWidgets import QApplication
from scripts.guiElements.renderPassWidget import RenderPassWidget

import sys

app = QApplication.instance()
if not app:
    app = QApplication([])

class TestRenderPassWidget(unittest.TestCase):
    def setUp(self):
        self.available_slots = ["slot1", "slot2", "slot3"]
        self.renderpass_type = "Mix By Percent"
        self.widget = RenderPassWidget(
            renderpass_type=self.renderpass_type,
            available_slots=self.available_slots,
            on_select_slot=lambda mode, widget: None,
            on_delete=lambda widget: None
        )

    def test_initial_state(self):
        self.assertEqual(self.widget.renderpass_type, self.renderpass_type)
        self.assertEqual(self.widget.available_slots, self.available_slots)
        self.assertEqual(len(self.widget.input_labels), self.widget.num_inputs)
        self.assertIsNone(self.widget.selected_output)
        self.assertEqual(self.widget.selected_inputs, [None] * self.widget.num_inputs)

    def test_get_settings_config(self):
        config = self.widget.get_settings_config(self.renderpass_type)
        self.assertIsInstance(config, list)
        self.assertTrue(all(isinstance(item, dict) for item in config))

    def test_set_and_get_slot(self):
        self.widget.selection_mode = 'input'
        self.widget.current_selected_input = 0
        self.widget.set_slot("slot1")
        self.assertEqual(self.widget.selected_inputs[0], "slot1")
        self.assertEqual(self.widget.input_labels[0].text(), "Input 1: slot1")

        self.widget.selection_mode = 'output'
        self.widget.set_slot("slot2")
        self.assertEqual(self.widget.selected_output, "slot2")
        self.assertEqual(self.widget.output_label.text(), "Output: slot2")

    def test_cleanup_and_delete(self):
        # Add a child widget to test cleanup
        child = self.widget.input_labels[0]
        self.assertIsNotNone(child)
        self.widget._cleanup()
        # After cleanup, child should have no parent
        self.assertIsNone(child.parent())

        # Test delete calls cleanup and removes widget
        self.widget._delete_self()
        self.assertIsNone(self.widget.parent())

if __name__ == "__main__":
    unittest.main()
