"""Product versions are looked up once per process."""
import types

import pytest

import pfs.utils.versions as versions


@pytest.fixture
def fakeEups(monkeypatch, tmp_path):
    """eups with every product setup at 1.0 in a directory that is not a git checkout; counts its uses."""
    calls = []

    class Eups:
        def findSetupProduct(self, productName):
            calls.append(productName)
            return types.SimpleNamespace(version=f'{productName}-1.0', dir=str(tmp_path))

    versions.getVersion.cache_clear()
    monkeypatch.setattr(versions, 'eups', types.SimpleNamespace(Eups=Eups))
    yield calls
    versions.getVersion.cache_clear()


def test_versionIsLookedUpOncePerProduct(fakeEups):
    assert versions.getVersion('pfs_utils') == 'pfs_utils-1.0'
    nLookups = len(fakeEups)
    assert versions.collectVersions(['pfs_utils']) == dict(pfs_utils='pfs_utils-1.0')
    assert len(fakeEups) == nLookups
    # Negative control: another product is looked up.
    assert versions.getVersion('datamodel') == 'datamodel-1.0'
    assert len(fakeEups) > nLookups


def test_failedLookupIsNotCached(fakeEups, monkeypatch, tmp_path):
    monkeypatch.setattr(versions, 'eups', None)
    monkeypatch.setenv('PFS_UTILS_DIR', str(tmp_path))
    with pytest.raises(RuntimeError, match='Could not determine version'):
        versions.getVersion('pfs_utils')

    monkeypatch.setattr(versions, 'eups', types.SimpleNamespace(Eups=lambda: types.SimpleNamespace(
        findSetupProduct=lambda productName: types.SimpleNamespace(version='7.4.42', dir=str(tmp_path)))))
    assert versions.getVersion('pfs_utils') == '7.4.42'
