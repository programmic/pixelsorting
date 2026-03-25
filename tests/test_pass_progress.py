import inspect
import os
import re


def _import_passes_module():
    try:
        from scripts import passes as passes_mod
        return passes_mod
    except Exception:
        try:
            from ..scripts import passes as passes_mod
            return passes_mod
        except Exception:
            raise


def _load_nodes_source():
    # look for nodegraph_ui nodes module in expected location
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts', 'nodegraph_ui', 'nodes.py')
    if not os.path.exists(path):
        # fallback to repository relative
        path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'nodegraph_ui', 'nodes.py')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return ''


def _function_supports_progress(fn):
    try:
        sig = inspect.signature(fn)
        for pname, param in sig.parameters.items():
            lname = pname.lower()
            if 'progress' in lname or 'callback' in lname:
                return True
        return False
    except Exception:
        return False


def test_list_passes_and_progress_support(capsys=None):
    """Print a table of pass functions, whether the function has a progress/callback
    parameter, and whether any node in `nodegraph_ui/nodes.py` calls it with a
    `progress=` argument.
    """
    passes = _import_passes_module()
    nodes_src = _load_nodes_source()

    funcs = []
    for name, obj in inspect.getmembers(passes, inspect.isfunction):
        # only include functions defined in the passes module and skip private
        if getattr(obj, '__module__', '').endswith('passes') and not name.startswith('_'):
            funcs.append((name, obj))

    # print header
    print("pass_name | function supports progress callback | node supports callback")
    print("-" * 80)

    for name, fn in sorted(funcs, key=lambda x: x[0]):
        func_has = _function_supports_progress(fn)

        # look for occurrences of `passes.<name>` in nodes source and see if
        # a nearby `progress=` appears in the same call
        node_support = False
        if nodes_src:
            for m in re.finditer(r"\bpasses\.%s\b" % re.escape(name), nodes_src):
                start = max(0, m.start() - 120)
                end = min(len(nodes_src), m.end() + 240)
                snippet = nodes_src[start:end]
                if 'progress=' in snippet or 'progress =' in snippet or 'progress:' in snippet:
                    node_support = True
                    break

        print(f"{name} | {func_has} | {node_support}")

    # ensure test doesn't fail; this is an informational test
    assert True


if __name__ == '__main__':
    # allow running the script directly outside pytest
    # ensure repository root is on sys.path so `from scripts import ...` works
    import sys
    repo_root = os.path.dirname(os.path.dirname(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    test_list_passes_and_progress_support()
