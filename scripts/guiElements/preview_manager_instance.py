# preview_manager_instance.py
# This file provides a clean interface to access the preview manager
# without causing circular imports

def get_preview_manager():
    """Get the preview manager instance."""
    from .previewManager import PreviewManager
    return PreviewManager.instance()

# Create the global instance when accessed
preview_manager = get_preview_manager()
