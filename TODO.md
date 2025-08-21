# Fix ModernSlotPreviewWidget RuntimeError

## Issue Analysis
- RuntimeError: Internal C++ object (ModernSlotPreviewWidget) already deleted
- Occurs in `_close_after_animation` method when calling `super().hide()`
- Widget has `Qt.WA_DeleteOnClose` attribute causing premature deletion
- Multiple components managing preview widgets independently

## Fix Plan

### 1. ModernSlotPreviewWidget Fixes
- [ ] Remove `Qt.WA_DeleteOnClose` to prevent premature deletion
- [ ] Add null checks before accessing C++ objects
- [ ] Disconnect animation signals properly
- [ ] Add proper cleanup in destructor

### 2. Usage Pattern Fixes
- [ ] ModernSlotTableWidget: Implement singleton pattern for preview widget
- [ ] ImportedImagesWidget: Fix lifecycle management
- [ ] Ensure proper cleanup when parent widgets are destroyed

### 3. Animation System Fixes
- [ ] Disconnect animation finished signal before closing
- [ ] Add safety checks in animation callbacks
- [ ] Ensure animation completion before deletion

### 4. Testing
- [ ] Test rapid mouse movements
- [ ] Test widget destruction scenarios
- [ ] Test multiple preview triggers
