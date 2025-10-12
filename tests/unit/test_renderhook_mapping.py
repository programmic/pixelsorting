import unittest
from scripts import renderHook


class TestRenderHookMapping(unittest.TestCase):
    def test_alias_to_param_preferred(self):
        meta = {
            'alias_to_param': {'Strength': 'strength', 'Kernel': 'kernel'},
            'settings': [
                {'label': 'Strength', 'alias': 'strength', 'name': 'strength'},
                {'label': 'Kernel', 'alias': 'kernel', 'name': 'kernel'},
            ]
        }
        mapping = renderHook.run_render_pass.__globals__['build_setting_name_map'](meta)
        self.assertEqual(mapping.get('Strength'), 'strength')
        self.assertEqual(mapping.get('Kernel'), 'kernel')

    def test_fallback_from_settings(self):
        meta = {
            'settings': [
                {'label': 'Strength', 'alias': 'strength', 'name': 'strength'},
                {'label': 'Kernel', 'alias': 'kernel', 'name': 'kernel'},
            ]
        }
        mapping = renderHook.run_render_pass.__globals__['build_setting_name_map'](meta)
        self.assertEqual(mapping.get('Strength'), 'strength')
        self.assertEqual(mapping.get('kernel'), 'kernel')

    def test_missing_meta_returns_empty(self):
        mapping = renderHook.run_render_pass.__globals__['build_setting_name_map'](None)
        self.assertEqual(mapping, {})


if __name__ == '__main__':
    unittest.main()
