"""policy/ and digital_twin/ import strictly downward, and never the hub.

    policy.configs < policy.data < digital_twin < processing < preprocessing < training < eval < policy/*.py
                                                                                     (stages, __main__)

digital_twin/ (the MuJoCo twin core, the real machine, the CLOAK client and runner) is the shared base hub/ and policy/
both import. From policy it imports only the leaves policy.configs and policy.data, and the CLOAK letterbox
(policy.training.cloak.masking: cv2, numpy and policy.data only; the client applies the server's own preprocessing).
policy.configs and policy.data never import digital_twin, which would make a cycle.

Every ``.py`` under policy/ and digital_twin/ is parsed (no imports run); module-level and function-level ``import``
/ ``from ... import`` statements count, relative ones resolved. Tests (``tests/`` directories, ``test_*.py``,
``conftest.py``) may cross layers (an integration test reads several), but not import the hub.
Imports inside a layer are free. ``python -m`` subprocess commands are not imports and are not checked.
"""
from __future__ import annotations

import ast
from pathlib import Path
import sys

POLICY = Path(__file__).resolve().parents[1]
ROOT = POLICY.parent
TWIN = ROOT / 'digital_twin'
LAYERS = ('configs', 'data', 'processing', 'preprocessing', 'training', 'eval')
TOP = len(LAYERS)                       # policy/__main__.py, policy/stages.py
FOREIGN = ('hub',)
SKIP = ('__pycache__', 'vendor')
# The leaf below every stage: these third-party packages, the standard library, and policy.configs.
DATA_THIRD_PARTY = {'numpy', 'PIL'}
# What digital_twin may import from policy (each module or anything below it).
TWIN_MAY_IMPORT = ('policy.configs', 'policy.data', 'policy.training.cloak.masking')
# The policy modules below digital_twin: they must not import it.
BELOW_TWIN = ('policy.configs', 'policy.data', 'policy.training.cloak.masking')


def is_test(path):
    return 'tests' in path.parts or path.name.startswith('test_') or path.name == 'conftest.py'


def sources(root=POLICY):
    return [p for p in sorted(root.rglob('*.py')) if not set(p.relative_to(ROOT).parts) & set(SKIP)]


def within(module, prefixes):
    return any(module == p or module.startswith(p + '.') for p in prefixes)


def module_of(path):
    """(dotted module name, its package's parts) of a file under the repository root."""
    parts = list(path.relative_to(ROOT).with_suffix('').parts)
    return '.'.join(p for p in parts if p != '__init__'), parts[:-1]


def imports(path, source=None):
    """(line, imported module) for every import statement in ``path`` (or ``source`` as if it were that file),
    relative imports made absolute."""
    _, package = module_of(path)
    tree = ast.parse(path.read_text() if source is None else source, str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[:len(package) - node.level + 1]
                module = '.'.join([*base, *([node.module] if node.module else [])])
            else:
                module = node.module
            yield node.lineno, module
            if module == 'policy':          # from policy import configs, stages
                for alias in node.names:
                    yield node.lineno, f'policy.{alias.name}'


def layer(module):
    """Index in LAYERS, TOP for policy/*.py, None for ``policy`` itself and anything outside policy."""
    parts = module.split('.')
    if parts[0] != 'policy' or len(parts) == 1:
        return None
    return LAYERS.index(parts[1]) if parts[1] in LAYERS else TOP


def where(path, line, module):
    return f'{path.relative_to(ROOT)}:{line} imports {module}'


def test_every_file_is_in_a_layer():
    """A new subpackage must be placed in LAYERS (anything else directly under policy/ counts as the top). A layer
    is a subpackage or a single module (policy/preprocessing.py)."""
    packages = {p.name for p in POLICY.iterdir() if p.is_dir() and any(p.rglob('*.py'))} - {'tests', '__pycache__'}
    modules = {p.stem for p in POLICY.glob('*.py') if p.stem in LAYERS}
    assert not packages - set(LAYERS), f'unplaced policy subpackages: {packages - set(LAYERS)}'
    assert packages | modules == set(LAYERS), f'missing layers: {set(LAYERS) - packages - modules}'


def test_policy_and_twin_never_import_the_hub():
    bad = [where(path, line, module) for path in sources() + sources(TWIN) for line, module in imports(path)
           if module.split('.')[0] in FOREIGN]
    assert not bad, 'policy/ and digital_twin/ must not import hub/ (it imports them):\n' + '\n'.join(bad)


def test_twin_imports_only_the_policy_leaves():
    bad = [where(path, line, module) for path in sources(TWIN) if not is_test(path) for line, module in imports(path)
           if module.split('.')[0] == 'policy' and not within(module, TWIN_MAY_IMPORT)]
    assert not bad, f'digital_twin/ may import from policy only {", ".join(TWIN_MAY_IMPORT)}:\n' + '\n'.join(bad)


def test_policy_leaves_never_import_the_twin():
    bad = [where(path, line, module) for path in sources() if not is_test(path) and within(module_of(path)[0], BELOW_TWIN)
           for line, module in imports(path) if module.split('.')[0] == 'digital_twin']
    assert not bad, f'{", ".join(BELOW_TWIN)} sit below digital_twin (it imports them):\n' + '\n'.join(bad)


def test_imports_only_go_down():
    bad = []
    for path in sources():
        if is_test(path):
            continue
        own = layer(module_of(path)[0])
        for line, module in imports(path):
            target = layer(module)
            if target is not None and own is not None and target > own:
                bad.append(where(path, line, module) + f'  ({LAYERS[own]} -> '
                           f'{LAYERS[target] if target < TOP else "policy top level"})')
    assert not bad, ' < '.join(LAYERS) + ' < policy/*.py; upward imports:\n' + '\n'.join(bad)


def test_data_is_a_leaf():
    """policy/data: standard library, numpy, PIL and policy.configs only (no cv2, JAX, TensorFlow, PyYAML at import
    of the package itself)."""
    bad = []
    for path in sources():
        if is_test(path) or layer(module_of(path)[0]) != LAYERS.index('data'):
            continue
        for line, module in imports(path):
            top = module.split('.')[0]
            if top in sys.stdlib_module_names or top in DATA_THIRD_PARTY or top == '__future__':
                continue
            if module.startswith(('policy.data', 'policy.configs')):
                continue
            bad.append(where(path, line, module))
    assert not bad, 'policy/data may import only the standard library, numpy, PIL and policy.configs:\n' + '\n'.join(bad)


def test_the_checker_sees_relative_and_function_level_imports():
    """Guards against a vacuous pass: relative imports resolve, function-level ones count."""
    found = {module for _, module in imports(POLICY / 'training' / 'cloak' / 'experiment.py')}
    assert {'policy.training.cloak.evaluator', 'policy.data.cloak_dataset'} <= found
    probe = POLICY / 'data' / 'probe.py'          # need not exist
    source = 'from . import data_root\ndef f():\n    from ..eval import report\n    import hub.app\n'
    assert [m for _, m in imports(probe, source)] == ['policy.data', 'policy.eval', 'hub.app']
    assert {m for _, m in imports(POLICY / 'stages.py', 'from policy import configs, stages')} == \
        {'policy', 'policy.configs', 'policy.stages'}
    assert layer('policy.eval.report') > layer('policy.data.cloak_dataset')
    assert layer('policy.stages') == TOP and layer('policy') is None and layer('numpy') is None
    assert layer('digital_twin.sim.kinematics') is None
    assert 'digital_twin.paths' in {m for _, m in imports(TWIN / 'sim' / 'kinematics.py')}
    assert 'policy.training.cloak.masking' in {m for _, m in imports(TWIN / 'policy' / 'cloak.py')}
    assert within('policy.data.camera', TWIN_MAY_IMPORT) and not within('policy.database', TWIN_MAY_IMPORT)
