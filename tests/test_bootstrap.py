"""Runtime metadata regression tests: no dependency installs or model imports."""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from aquafina_detector import bootstrap


@pytest.fixture
def runtime(monkeypatch):
    pins = {"torch": "2.5.1+cu121", "torchvision": "0.20.1+cu121"}
    requirements = Path(bootstrap.__file__).resolve().parents[2] / "requirements/colab.txt"
    for line in requirements.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            name, version = line.strip().split('==')
            pins[name] = version
    monkeypatch.setattr(bootstrap.importlib.metadata, 'distributions', lambda: [
        SimpleNamespace(metadata={'Name': name}, version=version) for name, version in pins.items()])
    monkeypatch.setattr(bootstrap.importlib.util, 'find_spec', lambda name: None)
    monkeypatch.setitem(sys.modules, 'yolox', SimpleNamespace(__file__='/fixture/YOLOX/yolox/__init__.py'))
    monkeypatch.setattr(bootstrap, 'audit_source', lambda root: {'import_audit': 'passed'})
    return pins


@pytest.mark.parametrize('version', ['0.1.1-2209072238', '0.1.1.post2209072238'])
def test_thop_equivalent_versions_pass(runtime, version):
    runtime['thop'] = version
    report = bootstrap.audit_runtime()
    assert report['packages']['thop'] == version
    assert report['packages']['torch'] == '2.5.1+cu121'


@pytest.mark.parametrize('version', ['0.1.1.post2209072239', '0.1.2', '0.1.1', 'invalid', None])
def test_thop_different_invalid_or_missing_version_fails(runtime, version):
    if version is None:
        runtime.pop('thop')
    else:
        runtime['thop'] = version
    with pytest.raises(RuntimeError, match='Pinned dependency mismatch.*thop'):
        bootstrap.audit_runtime()


def test_different_torch_release_with_cuda_suffix_fails(runtime):
    runtime['torch'] = '2.5.2+cu121'
    with pytest.raises(RuntimeError, match='Pinned dependency mismatch.*torch'):
        bootstrap.audit_runtime()
