"""Shared fixtures for the v2 test suite.

Real CHASE files are used when available.  Point ``SCV_TEST_DATA`` (and the
optional ``SCV_TEST_FE`` / ``SCV_TEST_UNCOMPRESSED``) at the data directory;
tests that need a missing file are skipped rather than failed.
"""

import hashlib
import os

import pytest

# No machine-specific defaults: point the environment variables at your data.
# When a variable is unset the path is empty, os.path.exists() is False, and the
# tests that need that file are skipped rather than failed.
DATA = os.environ.get("SCV_TEST_DATA", "")
FE_DIR = os.environ.get("SCV_TEST_FE", "")
UNCOMPRESSED = os.environ.get("SCV_TEST_UNCOMPRESSED", "")

HA_RICE = os.path.join(DATA, "RSM20260424T080829_0016_HA.fits") if DATA else ""
VDOP = os.path.join(DATA, "RSM20260424T080829_0016_HA_Vdop.fits") if DATA else ""
FE_RICE = FE_DIR if FE_DIR else ""

# v1 parameter file archived for RSM20260424T080829_0016_HA_Vdop.fits
V1_PARAMS = dict(A=1.797406, B=-0.165739, C=-0.720372, D=0.006673,
                 G=-0.839428, e=0.000005, N=5.0, MC1=0.037062, MC2=-0.049982)

requires_ha = pytest.mark.skipif(not os.path.exists(HA_RICE),
                                 reason="RICE Halpha test file not available")
requires_vdop = pytest.mark.skipif(not os.path.exists(VDOP),
                                   reason="Doppler test file not available")
requires_fe = pytest.mark.skipif(not os.path.exists(FE_RICE),
                                 reason="RICE Fe test file not available")
requires_plain = pytest.mark.skipif(not os.path.exists(UNCOMPRESSED),
                                    reason="uncompressed test file not available")


def dir_digest(path):
    """Order-independent digest of every file in a directory."""
    h = hashlib.sha256()
    for name in sorted(os.listdir(path)):
        p = os.path.join(path, name)
        if not os.path.isfile(p):
            continue
        h.update(name.encode())
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="session")
def legacy_run(tmp_path_factory):
    """Run the processor once in legacy mode on the Doppler product."""
    if not os.path.exists(VDOP):
        pytest.skip("Doppler test file not available")
    from solar_chromosphere_velocity import ChromosphereProcessor, Config
    out = tmp_path_factory.mktemp("legacy")
    cfg = Config(input_dir=DATA, output_dir=str(out), legacy=True,
                 log_level="ERROR", write_observation_log=True)
    return ChromosphereProcessor(cfg).process_file(VDOP)


@pytest.fixture(scope="session")
def full_run(tmp_path_factory):
    """Run the processor once with the complete physics."""
    if not os.path.exists(VDOP):
        pytest.skip("Doppler test file not available")
    from solar_chromosphere_velocity import ChromosphereProcessor, Config
    out = tmp_path_factory.mktemp("full")
    cfg = Config(input_dir=DATA, output_dir=str(out), legacy=False,
                 log_level="ERROR")
    return ChromosphereProcessor(cfg).process_file(VDOP)
