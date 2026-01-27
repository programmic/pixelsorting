"""Unit tests for createSettingsTool.py helpers."""
import unittest
import sys
import os
import json
from pathlib import Path

# Ensure scripts package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts import createSettingsTool
from scripts import enums

class TestCreateSettingsTool(unittest.TestCase):
    def test_is_function_ignored_in_source(self):
        # read passes.py
        repo_root = Path(__file__).resolve().parents[2]
        passes_path = repo_root / 'scripts' / 'passes.py'
        src = passes_path.read_text(encoding='utf-8')
        # ensure known function with #globalignore exists
        self.assertIn('def ensure_rgba', src)
        self.assertTrue(createSettingsTool._is_function_ignored_in_source('ensure_rgba', src))

    def test_get_unimplemented_functions_excludes_ignored(self):
        missing = createSettingsTool.get_unimplemented_functions()
        # ensure ensure_rgba (which is marked with #globalignore) is not in missing list
        self.assertNotIn('ensure_rgba', missing)

    def test_update_json_config_enum_normalize(self):
        # Call update_json_config with Enums in settings and check that renderPasses.json gets primitive values
        repo_root = Path(__file__).resolve().parents[2]
        json_path = repo_root / 'renderPasses.json'
        # load the json to restore later
        orig = json.loads(json_path.read_text(encoding='utf-8') or '{}')
        try:
            settings = [
                {'label': enums.SortMode.LUM, 'alias': enums.SortMode.LUM, 'type': enums.ControlType.SLIDER, 'default': enums.SortMode.LUM}
            ]
            createSettingsTool.update_json_config('TEST_ENUM_NORMALIZE', settings, 1)
            cfg = json.loads(json_path.read_text(encoding='utf-8') or '{}')
            self.assertIn('TEST_ENUM_NORMALIZE', cfg)
            entry = cfg['TEST_ENUM_NORMALIZE']
            self.assertIsInstance(entry.get('settings'), list)
            s = entry['settings'][0]
            # label/alias/type/default should be primitives (strings)
            self.assertIsInstance(s.get('label'), (str, type(None)))
            self.assertIsInstance(s.get('alias'), (str, type(None)))
            self.assertIsInstance(s.get('type'), (str, type(None)))
            # default in our test becomes 'lum' (string)
            self.assertEqual(s.get('default'), 'lum')
        finally:
            # restore original json to avoid test side-effects
            json_path.write_text(json.dumps(orig, indent=2), encoding='utf-8')

    def test_dynamic_enum_generation(self):
        repo_root = Path(__file__).resolve().parents[2]
        enums_path = repo_root / 'enums.json'
        # Save original enums.json content
        orig = json.loads(enums_path.read_text(encoding='utf-8') or '{}') if enums_path.exists() else {}
        try:
            # Test saving and loading enums
            dynamic_enums = {"DynamicEnum": ["value1", "value2", "value3"]}
            enums.save_enums_to_json(str(enums_path), dynamic_enums)
            loaded_enums = enums.load_enums_from_json(str(enums_path))
            self.assertEqual(dynamic_enums, loaded_enums)

            # Test dynamic enum generation
            DynamicEnum = enums.generate_enum("DynamicEnum", loaded_enums["DynamicEnum"])
            self.assertTrue(hasattr(DynamicEnum, "VALUE1"))
            self.assertEqual(DynamicEnum.VALUE1.value, "value1")
        finally:
            # Restore original enums.json content
            enums_path.write_text(json.dumps(orig, indent=2), encoding='utf-8')

    def test_control_type_selection(self):
        settings = [
            {'label': "TestLabel", 'alias': "test_label", 'type': "dropdown", 'default': "value1", 'options': ["value1", "value2"]},
            {'label': "TestLabel2", 'alias': "test_label2", 'type': "radio", 'default': "value2", 'options': ["value1", "value2"]}
        ]
        repo_root = Path(__file__).resolve().parents[2]
        json_path = repo_root / 'renderPasses.json'
        orig = json.loads(json_path.read_text(encoding='utf-8') or '{}')
        try:
            createSettingsTool.update_json_config('TEST_CONTROL_TYPE', settings, 1)
            cfg = json.loads(json_path.read_text(encoding='utf-8') or '{}')
            self.assertIn('TEST_CONTROL_TYPE', cfg)
            entry = cfg['TEST_CONTROL_TYPE']
            self.assertEqual(entry['settings'][0]['type'], "dropdown")
            self.assertEqual(entry['settings'][1]['type'], "radio")
        finally:
            json_path.write_text(json.dumps(orig, indent=2), encoding='utf-8')


if __name__ == '__main__':
    unittest.main()
