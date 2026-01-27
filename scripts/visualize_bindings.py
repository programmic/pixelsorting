"""
Small helper script to render the current GUI pipeline bindings to DOT/PNG.

Usage (when the GUI is running and you have a Python REPL with access to the GUI instance):

    from scripts import renderHook
    renderHook.visualize_pipeline(gui_instance, out_dot='saved/pipeline.dot', out_png='saved/pipeline.png')

If you prefer a non-interactive way: open the GUI, then in a Python console in the same process call the function above.

Note: Rendering PNG requires Graphviz `dot` to be installed and on PATH.
"""

if __name__ == '__main__':
    print("This script is a helper. Import `renderHook.visualize_pipeline` from inside the running GUI process.")
