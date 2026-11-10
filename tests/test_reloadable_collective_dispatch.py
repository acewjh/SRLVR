"""Communication dispatch must not synchronize outstanding peer operations."""
import ast
from contextlib import contextmanager
from pathlib import Path

import pytest


def wrapper(events):
    # Execute the production context manager without importing/patching a live
    # distributed runtime in this CPU regression test.
    path = Path(__file__).resolve().parents[1] / 'frameworks/slime-ascend/slime/utils/reloadable_process_group.py'
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_wrap_low_level_call')

    def clear_memory():
        raise AssertionError('device synchronize would wait for an undispatched peer collective')

    namespace = {'contextmanager': contextmanager, 'available_memory': lambda: {'free_GB': .1},
                 'clear_memory': clear_memory, 'print_memory': lambda tag: events.append(tag) or {'free_GB': .1}}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['_wrap_low_level_call']


def test_low_memory_does_not_block_collective_dispatch():
    events = []
    with wrapper(events)():
        events.append('collective dispatched')
    assert events == ['collective dispatched']


def test_collective_failure_preserves_exception_and_memory_diagnostic():
    events = []
    error = RuntimeError('transport failure')
    with pytest.raises(RuntimeError) as caught:
        with wrapper(events)():
            raise error
    assert caught.value is error
    assert events == ['after torch distributed error']
    assert 'free_GB' in error.__notes__[0]
