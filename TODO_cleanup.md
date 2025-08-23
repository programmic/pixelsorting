# PixelSorting V2 Project Cleanup Plan

## Phase 1: Remove Unused Files
- [x] Check if `scripts/alpha_over_fix.py` is redundant and can be removed
- [x] Remove `scripts/alpha_over_fix.py` file
- [x] Update `tests/unit/test_alpha_over.py` to remove references to alpha_over_fix
- [ ] Identify any other unused or redundant files

## Phase 2: Merge Related Files
- [x] Analyze `modernSlotPreviewWidget.py` vs `modernSlotPreviewWidgetFixed.py` for merging
- [x] Create merged file `modernSlotPreviewWidgetMerged.py`
- [x] Update references in `masterGUI.py`, `modernSlotTableWidget.py`, and `previewManager.py`
- [ ] Review `renderPassWidget.py` and `renderPassSettingsWidget.py` for potential consolidation

## Phase 3: Refactor Code
- [ ] Refactor `passes.py` for better organization and readability
- [ ] Ensure function name consistency between `passes.py` and `renderHook.py`

## Phase 4: Update Documentation
- [ ] Update README.md with current project structure and setup instructions
- [ ] Remove references to non-existent files like `setupProject.py` and `main.py`

## Phase 5: Testing
- [ ] Run existing tests to ensure functionality remains intact
- [ ] Add new tests if necessary

## Current Status: Phase 2 partially completed - ModernSlotPreviewWidget files merged and references updated
