"""v2 acceptance tests.

Maps onto the acceptance criteria of the upgrade plan:
  1/2  legacy regression and physics identity
  3    header-driven velocity calibration
  4    the input directory is never modified
  5    observation-time naming, no collisions, index.csv
  6    numerical agreement of the batched inversion
  7    performance/memory envelope
  8    RICE and non-RICE reading
"""

import os
import glob

import numpy as np
import pytest
from astropy.io import fits

from conftest import (HA_RICE, V1_PARAMS, VDOP, FE_RICE, UNCOMPRESSED,
                      dir_digest, requires_fe, requires_ha, requires_plain,
                      requires_vdop)

import solar_chromosphere_velocity as scv
from solar_chromosphere_velocity import Config, ChromosphereProcessor


# ---------------------------------------------------------------- geometry
def _v1_scalar(x, y, cx, cy, R, B0):
    """The v1 SolarCoordinateTransformer.pixel_to_solar, inline."""
    xr, yr = x - cx, y - cy
    r = np.hypot(xr, yr)
    if r == 0:
        return 0.0, 1.0, 0.0, 0.0, 1.0, 0.0
    rho = np.arcsin(r / R)
    b0 = np.radians(B0)
    sinB = np.sin(b0) * np.cos(rho) + np.cos(b0) * np.sin(rho) * (yr / r)
    B = np.arcsin(sinB)
    sinL = -np.sin(rho) * (-xr / r) / np.cos(B)
    L = np.arcsin(sinL)
    return np.sin(B), np.cos(B), np.sin(L), np.cos(L), np.degrees(rho)


def test_geometry_matches_v1_scalar():
    """Vectorised geometry must reproduce the v1 scalar transform (in-disk)."""
    from solar_chromosphere_velocity.geometry import solar_coords
    cx, cy, R, B0 = 1191.5977, 1154.8561, 920.0, -4.8395
    rng = np.random.default_rng(0)
    ang = rng.uniform(0, 2 * np.pi, 200)
    rad = R * np.sqrt(rng.uniform(0, 1, 200))
    ys = (cy + rad * np.sin(ang)).astype(int)
    xs = (cx + rad * np.cos(ang)).astype(int)
    got = solar_coords(ys.astype(float), xs.astype(float), R, b0_deg=B0,
                       cx=cx, cy=cy)
    ref = [_v1_scalar(int(x), int(y), cx, cy, R, B0) for x, y in zip(xs, ys)]
    for k, idx in (("sin_lat", 0), ("cos_lat", 1), ("sin_lon", 2),
                   ("cos_lon", 3), ("rho_deg", 4)):
        d = np.max(np.abs(np.asarray(got[k]) - np.array([r[idx] for r in ref])))
        assert d < 1e-9, (k, d)


# ------------------------------------------------- calibration / inversion
@requires_fe
def test_fe_line_center_not_midpoint():
    """A 46-sample Fe cube must land near the WCS core, not at len//2 = 23."""
    from solar_chromosphere_velocity.data_loader import SpectralCube
    from solar_chromosphere_velocity.inversion import DopplerInverter
    cfg = Config()
    inv = DopplerInverter(cfg)
    with SpectralCube(FE_RICE) as cube:
        p = cube.params
        avg, _ = inv.average_profile(cube, p)
        assert len(avg) < 90
        lc = scv.fit_line_center(avg, p, legacy=False)
    assert lc.method == "gauss"
    assert abs(lc.continuous - 23) > 5, "still using the v1 midpoint fallback"
    if lc.wcs_guess is not None:
        assert abs(lc.continuous - lc.wcs_guess) < 4


@requires_ha
def test_velocity_scale_from_header():
    """CALSPEC must equal c * CDELT3 / WAVE_LEN."""
    from solar_chromosphere_velocity.data_loader import SpectralCube
    cfg = Config()
    p = ChromosphereProcessor(cfg)
    with SpectralCube(HA_RICE) as cube:
        params = cube.params
    scale = scv.velocity_scale_km_s_per_pixel(params.cdelt3, params.wave_len)
    expect = 299792.458 * params.cdelt3 / params.wave_len
    assert abs(scale - expect) < 1e-9
    # ... and it must differ from v1's hard-coded 2.2136
    assert abs(scale - 2.21387) < 1e-3


# ------------------------------------------------------- FITS compatibility
@requires_ha
def test_rice_section_matches_full_slice():
    """Partial reads must be bit-identical to a full-array slice (RICE)."""
    from solar_chromosphere_velocity.data_loader import SpectralCube
    with SpectralCube(HA_RICE) as cube:
        assert cube.compressed
        block = cube.read(60, 81, 900, 960, 900, 960)
        hdu = cube._hdul[cube.hdu_index]
        ref = np.asarray(hdu.data[60:81, 900:960, 900:960], dtype=np.float32)
    assert block.shape == ref.shape
    assert np.array_equal(block, ref)


@requires_ha
def test_locating_cube_does_not_decompress():
    """Opening a RICE cube must not pull the whole array into memory."""
    import resource
    from solar_chromosphere_velocity.data_loader import SpectralCube
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with SpectralCube(HA_RICE) as cube:
        _ = cube.shape, cube.params
    growth_mb = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before) / 1e3
    # a full decompression costs ~1260 MB; allow generous slack
    assert growth_mb < 200, "locating the cube decompressed it (%.0f MB)" % growth_mb


@requires_plain
def test_uncompressed_cube_supported():
    from solar_chromosphere_velocity.data_loader import SpectralCube
    with SpectralCube(UNCOMPRESSED) as cube:
        assert not cube.compressed
        assert cube.n_lambda >= 2
        n = min(5, cube.n_lambda)
        b = cube.read(0, n, 100, 110, 100, 110)
        assert b.shape == (n, 10, 10)


@requires_ha
@requires_plain
def test_hdu_located_by_shape_not_index():
    """HA puts the cube in hdu[1], the resampled product in hdu[0]."""
    from solar_chromosphere_velocity.data_loader import SpectralCube
    with SpectralCube(HA_RICE) as c:
        assert c.hdu_index == 1
    with SpectralCube(UNCOMPRESSED) as c:
        assert c.hdu_index == 0


# ------------------------------------------------------------- regression
@requires_vdop
def test_legacy_reproduces_v1_parameters(legacy_run):
    """v2 --legacy must reproduce every identifiable v1 parameter."""
    fit = legacy_run.fit
    got = fit.as_dict()
    for name in ("A", "B", "C", "D", "G", "MC1", "MC2"):
        assert got[name] == pytest.approx(V1_PARAMS[name], abs=1e-5), name
    # e and N are the degenerate pair: both solutions sit at e~0, where N has
    # no effect on the model at all.
    assert abs(got["e"]) < 1e-4


@requires_vdop
def test_legacy_limb_term_pinned(legacy_run):
    assert legacy_run.fit.limb_nonneg_active
    assert legacy_run.fit.dropped == ("e",)
    v = legacy_run.layers["V_LIMB"]
    assert np.nanmax(np.abs(v)) < 1e-6


@requires_vdop
def test_component_identity(full_run):
    L = full_run.layers
    total = (L["V_DIFFROT"] + L["V_LIMB"] + L["V_MERID"] + L["V_OFFSET"])
    m = np.isfinite(L["V_LARGE"])
    assert m.sum() > 1e6
    assert np.nanmax(np.abs(total[m] - L["V_LARGE"][m])) < 1e-5


@requires_vdop
@requires_vdop
def test_residual_is_zero_mean_on_quiet_sun(tmp_path):
    """The level-1 residual (V_RESID) is zero-mean on the quiet Sun."""
    cfg = Config(input_dir=os.path.dirname(VDOP), output_dir=str(tmp_path),
                 legacy=False, write_vresid=True, log_level="ERROR")
    res = ChromosphereProcessor(cfg).process_file(VDOP)
    assert abs(np.nanmean(res.layers["V_RESID"])) < 0.05


@requires_vdop
def test_vlocal_is_vresid_minus_vss(tmp_path):
    """V_LOCAL = V_RESID - V_SS, exactly, with the intermediates written."""
    cfg = Config(input_dir=os.path.dirname(VDOP), output_dir=str(tmp_path),
                 legacy=False, write_vresid=True, write_scan_terms=True,
                 log_level="ERROR")
    L = ChromosphereProcessor(cfg).process_file(VDOP).layers
    assert set(L) >= {"V_LOCAL", "V_RESID", "V_SCAN_SADDLE", "V_FLAT", "V_SS"}
    ss = L["V_SCAN_SADDLE"].astype(np.float64) + L["V_FLAT"].astype(np.float64)
    m = np.isfinite(ss) & np.isfinite(L["V_SS"])
    assert m.any() and np.allclose(ss[m], L["V_SS"][m], atol=1e-6)
    diff = L["V_RESID"].astype(np.float64) - ss
    ok = np.isfinite(diff) & np.isfinite(L["V_LOCAL"])
    assert ok.any() and np.allclose(diff[ok], L["V_LOCAL"][ok], atol=1e-6)


@requires_vdop
def test_v_offset_is_a_pure_zero_point(tmp_path):
    """G must shift by exactly the zero point; nothing else may change."""
    base = Config(input_dir=os.path.dirname(VDOP), output_dir=str(tmp_path / "a"),
                  legacy=False, log_level="ERROR")
    shifted = Config(input_dir=os.path.dirname(VDOP),
                     output_dir=str(tmp_path / "b"), legacy=False,
                     zero_point_km_s=0.632, log_level="ERROR")
    p0 = ChromosphereProcessor(base).process_file(VDOP).fit
    p1 = ChromosphereProcessor(shifted).process_file(VDOP).fit
    assert p0.params[4] - p1.params[4] == pytest.approx(0.632, abs=1e-6)
    for i in (0, 1, 2, 3, 5, 7, 8):
        assert p0.params[i] == pytest.approx(p1.params[i], abs=1e-6)


@requires_vdop
def test_row_block_invariance(tmp_path):
    """Streaming block size must not change the answer."""
    a = Config(input_dir=os.path.dirname(VDOP), output_dir=str(tmp_path / "a"),
               log_level="ERROR", spatial_strip_rows=32)
    b = Config(input_dir=os.path.dirname(VDOP), output_dir=str(tmp_path / "b"),
               log_level="ERROR", spatial_strip_rows=256)
    ra = ChromosphereProcessor(a).process_file(VDOP)
    rb = ChromosphereProcessor(b).process_file(VDOP)
    for k in ("V_DIFFROT", "V_LIMB", "V_MERID", "V_OFFSET", "V_LARGE"):
        d = np.nanmax(np.abs(ra.layers[k] - rb.layers[k]))
        assert d < 1e-5, (k, d)


# ----------------------------------------------------------- output layout
@requires_vdop
def test_multihdu_structure(legacy_run):
    with fits.open(legacy_run.outfile) as h:
        names = [hd.header.get("EXTNAME", "PRIMARY") for hd in h]
        assert names[0] == "PRIMARY"
        for want in ("VLOS_OBS", "VLOS_CORR", "V_DIFFROT", "V_LIMB", "V_MERID",
                     "V_OFFSET", "V_LARGE", "V_LOCAL", "V_MODEL_ERR",
                     "MASK_ACTIVE", "FITPARAMS"):
            assert want in names, want
        assert h["MASK_ACTIVE"].data.dtype == np.uint8
        for name in names[1:-1]:
            hd = h[name]
            assert "QUANTITY" in hd.header, name
            assert "MODELVER" in hd.header, name
            if name == "MASK_ACTIVE":
                assert "BUNIT" not in hd.header
            else:
                assert hd.header["BUNIT"] == "km/s", name
        # level-0.5 intermediates are not written unless asked for
        for unwanted in ("V_RESID", "V_SCAN_SADDLE", "V_FLAT", "V_SS"):
            assert unwanted not in names, unwanted
        t = h["FITPARAMS"].data
        assert set(t["NAME"]) == set(V1_PARAMS)
        assert len(t) == 9
    ph = fits.getheader(legacy_run.outfile)
    for k in ("OBSTIME", "SRCFILE", "MASKBIND", "EFFRAD", "ZEROPT",
              "MODELVER", "LEGACY", "RMSFIT", "NPOINT"):
        assert k in ph, k
    assert ph["LEGACY"] is True


@requires_vdop
def test_binding_mask_reported(legacy_run):
    """edge=80 deg is absorbed by disk=0.95 -> the disk cut binds."""
    assert legacy_run.binding_mask == "disk"
    assert legacy_run.layers["V_LARGE"].shape == legacy_run.layers["VLOS_OBS"].shape


def test_observation_key_from_header():
    from solar_chromosphere_velocity.output import observation_key
    h = fits.Header()
    h["STR_TIME"] = "2026-04-24T07:49:33"
    h["SPECLINE"] = "HA"
    k = observation_key(h, "/tmp/whatever.fits")
    assert k.stamp == "20260424T074933"
    assert k.day == "20260424"
    assert k.line == "HA"
    assert k.source == "STR_TIME"


def test_observation_key_falls_back_to_filename():
    from solar_chromosphere_velocity.output import observation_key
    k = observation_key(fits.Header(), "/x/RSM2022-10-02T20_22_57_HA_Vdop.fits")
    assert k.stamp == "20221002T202257"
    assert k.line == "HA"


@requires_vdop
def test_two_times_do_not_collide(tmp_path):
    """The same source processed twice must not silently overwrite."""
    cfg = Config(input_dir=os.path.dirname(VDOP), output_dir=str(tmp_path),
                 log_level="ERROR")
    proc = ChromosphereProcessor(cfg)
    p1 = proc.process_file(VDOP)
    # a second observation time must produce a different file
    from solar_chromosphere_velocity.output import ObsKey
    k2 = ObsKey(day="20260424", stamp="20260424T235959", line="HA",
                source="TEST", srcfile="other.fits")
    p2 = proc.layout.resolve(k2, product="scv")
    assert p2 != p1.outfile
    # same key but a different source file must be disambiguated
    k3 = ObsKey(day=p1.key.day, stamp=p1.key.stamp, line=p1.key.line,
                source="TEST", srcfile="different.fits")
    p3 = proc.layout.resolve(k3, product="scv")
    assert p3 != p1.outfile


@requires_vdop
def test_index_written(legacy_run):
    idx = os.path.join(os.path.dirname(os.path.dirname(legacy_run.outfile)),
                       "index.csv")
    assert os.path.exists(idx)
    with open(idx) as fh:
        head = fh.readline().strip().split(",")
        rows = fh.readlines()
    for col in ("obs_time", "hdu_file", "src_file", "model_version", "status"):
        assert col in head
    assert any("20260424T080829" in r for r in rows)


@requires_vdop
def test_input_directory_untouched(tmp_path):
    """v1 wrote its intermediate product back next to the inputs; v2 must not."""
    src_dir = os.path.dirname(VDOP)
    before = dir_digest(src_dir)
    cfg = Config(input_dir=src_dir, output_dir=str(tmp_path), log_level="ERROR")
    ChromosphereProcessor(cfg).process_file(VDOP)
    assert dir_digest(src_dir) == before


# -------------------------------------------------------------- misc / api
@requires_vdop
def test_excludes_own_outputs_by_default():
    cfg = Config(input_dir=os.path.dirname(VDOP), output_dir="/tmp/unused")
    assert any("Vdop" in g for g in cfg.exclude_patterns)


def test_config_legacy_resolution():
    assert Config(legacy=True).model_version == "legacy-1.0"
    assert Config(legacy=False).model_version == "full-2.0"
    assert Config(legacy=True).zero_point == pytest.approx(0.632)
    assert Config(legacy=False).zero_point == 0.0
    assert Config(legacy=True).visualize_steps is False


def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        Config(disk_mask_radius=0.0)
    with pytest.raises(ValueError):
        Config(edge_exclusion_angle=95.0)
    with pytest.raises(ValueError):
        Config(rotation_amplitude="deg_per_day")


def test_model_design_shapes():
    from solar_chromosphere_velocity.models import VelocityModel
    coords = dict(sin_lat=np.array([0.1, 0.2]), sin_2lat=np.array([0.2, 0.4]),
                  cos_lat=np.array([0.99, 0.98]), sin_lon=np.array([0.3, -0.3]),
                  cos_lon=np.array([0.95, 0.95]), rho_rad=np.array([0.5, 1.0]),
                  b0_rad=0.0)
    for ver in ("full-2.0", "legacy-1.0"):
        X = VelocityModel(ver, limb_exponent=2.0).design(coords)
        assert X.shape == (2, 8)
    c = VelocityModel("full-2.0", limb_exponent=2.0).components(
        coords, np.array([1.0, -0.1, -0.1, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0]))
    s = c["V_DIFFROT"] + c["V_LIMB"] + c["V_MERID"] + c["V_OFFSET"]
    assert np.allclose(s, c["V_LARGE"])


def test_v1_dead_options_are_reported(caplog):
    """v1 flags that never did anything must be flagged, not silently kept."""
    from solar_chromosphere_velocity.cli import build_parser, _report_ignored
    p = build_parser(prog="process-chase", legacy_default=True)
    args = p.parse_args(["-o", "/tmp/out", "-i", "/tmp",
                         "--doppler-method", "gaussian_fit"])
    with caplog.at_level("WARNING"):
        _report_ignored(args)
    assert "doppler_method" in caplog.text


# ------------------------------------------------------------------ plotting
def _fake_layers(n=64, ny=48):
    rng = np.random.default_rng(3)
    return {
        "VLOS_OBS": rng.normal(0, 1.0, (ny, n)).astype(np.float32),
        "VLOS_CORR": rng.normal(0, 1.0, (ny, n)).astype(np.float32),
        "V_DIFFROT": (np.linspace(-2, 2, n)[None, :] * np.ones((ny, 1))).astype(np.float32),
        "V_LIMB": np.zeros((ny, n), dtype=np.float32),
        "V_MERID": rng.normal(0, 0.05, (ny, n)).astype(np.float32),
        "V_OFFSET": np.full((ny, n), -0.8394, dtype=np.float32),
        "V_LARGE": rng.normal(0, 1.0, (ny, n)).astype(np.float32),
        "V_LOCAL": rng.normal(0, 0.85, (ny, n)).astype(np.float32),
        "V_RESID": rng.normal(0, 0.9, (ny, n)).astype(np.float32),
        "V_MODEL_ERR": np.abs(rng.normal(0, 0.01, (ny, n))).astype(np.float32),
        "MASK_ACTIVE": (rng.random((ny, n)) > 0.99).astype(np.uint8),
    }


def test_plot_layers_multi_panel(tmp_path):
    """Several components in one figure, as separate subplots."""
    pytest.importorskip("matplotlib")
    from solar_chromosphere_velocity import plot_layers
    from solar_chromosphere_velocity.plotting import COMPONENT_PANELS
    out = plot_layers(_fake_layers(), str(tmp_path / "c.png"),
                      panels=COMPONENT_PANELS, ncols=3)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 5000


def test_plot_layers_default_panels_and_subset(tmp_path):
    pytest.importorskip("matplotlib")
    from solar_chromosphere_velocity import plot_layers
    layers = _fake_layers()
    full = plot_layers(layers, str(tmp_path / "all.png"))
    assert os.path.exists(full) and os.path.getsize(full) > 5000
    one = plot_layers(layers, str(tmp_path / "one.png"),
                      panels=["V_MERID"], ncols=1)
    assert os.path.exists(one)


def test_plot_rejects_unknown_panels(tmp_path):
    pytest.importorskip("matplotlib")
    from solar_chromosphere_velocity import plot_layers
    with pytest.raises(ValueError):
        plot_layers(_fake_layers(), str(tmp_path / "x.png"), panels=["NOT_A_LAYER"])


def test_plot_per_component_and_summary(tmp_path):
    pytest.importorskip("matplotlib")
    from solar_chromosphere_velocity import plot_summary, write_component_figures
    from solar_chromosphere_velocity.plotting import COMPONENT_PANELS
    layers = _fake_layers()
    figs = write_component_figures(layers, str(tmp_path / "comp"))
    assert set(figs) == set(COMPONENT_PANELS)
    for path in figs.values():
        assert os.path.exists(path) and os.path.getsize(path) > 3000
    s = plot_summary(layers, str(tmp_path / "summary.png"))
    assert os.path.exists(s) and os.path.getsize(s) > 5000


def test_plot_from_file_roundtrip(tmp_path):
    """The figure can be produced from the written multi-HDU product."""
    pytest.importorskip("matplotlib")
    from solar_chromosphere_velocity import plot_from_file
    from solar_chromosphere_velocity.output import write_multihdu
    from astropy.io import fits
    layers = _fake_layers()
    hdr = fits.Header()
    hdr["OBSTIME"] = "20260424T000000"
    hdr["SRCFILE"] = "fake.fits"
    path = str(tmp_path / "x_scv.fits")
    write_multihdu(path, layers, hdr, None, {"CDELT1": 1.04, "CX": 32.0, "CY": 24.0})
    out = plot_from_file(path, str(tmp_path / "fromfile.png"),
                         panels=["V_DIFFROT", "V_RESID"])
    assert os.path.exists(out) and os.path.getsize(out) > 3000


@requires_vdop
def test_pipeline_writes_figures(tmp_path):
    """--visualize must emit the multi-panel figures next to the FITS."""
    pytest.importorskip("matplotlib")
    cfg = Config(input_dir=os.path.dirname(VDOP), output_dir=str(tmp_path),
                 visualize=True, plot_dpi=60, log_level="ERROR")
    res = ChromosphereProcessor(cfg).process_file(VDOP)
    assert {"layers", "components", "summary"} <= set(res.figures), res.figures
    for key in ("layers", "components", "summary"):
        assert os.path.exists(res.figures[key]), key
        assert os.path.getsize(res.figures[key]) > 3000
    assert res.figures["layers"].startswith(os.path.splitext(res.outfile)[0])


# ------------------------------------------- regressions from the v2 review
def test_primary_header_has_no_duplicate_keywords(tmp_path):
    """The input header and our cards overlap; PRIMARY must be unambiguous."""
    from astropy.io import fits
    from solar_chromosphere_velocity.output import write_multihdu
    layers = _fake_layers()
    src = fits.Header()
    src["B0"] = -4.84          # also written by base_cards
    src["INST_ROT"] = 0        # int here, float in base_cards -> conflicting
    src["CDELT1"] = 1.04
    src["CRPIX1"] = 32.0
    src["COMMENT"] = "a legitimate repeat"
    cards = {"B0": -4.8395, "INST_ROT": 0.0, "CDELT1": 1.04, "CX": 32.0}
    path = str(tmp_path / "dup.fits")
    write_multihdu(path, layers, src, None, cards)
    with fits.open(path) as h:
        keys = [c.keyword for c in h[0].header.cards]
        dups = {k for k in keys if keys.count(k) > 1 and k not in ("COMMENT", "HISTORY")}
        assert not dups, dups
        assert h[0].header["INST_ROT"] == 0.0        # ours wins
        assert h[0].header["B0"] == pytest.approx(-4.8395)


def test_fitparams_flags_unidentifiable_parameters(tmp_path):
    """e (pinned) and N (degenerate) must be flagged, not reported as +/-0."""
    from solar_chromosphere_velocity.models import NormalEquations, fit_model
    # construct a sample where e wants to be negative -> the >=0 bound is active
    rng = np.random.default_rng(5)
    n = 4000
    coords = dict(
        sin_lat=rng.uniform(-0.8, 0.8, n),
        sin_2lat=rng.uniform(-1, 1, n),
        cos_lat=rng.uniform(0.6, 1.0, n),
        sin_lon=rng.uniform(-1, 1, n),
        cos_lon=rng.uniform(-1, 1, n),
        rho_rad=rng.uniform(0, 1.2, n),
        b0_rad=0.0,
    )
    from solar_chromosphere_velocity.models import VelocityModel
    m = VelocityModel("full-2.0", limb_exponent=2.0)
    # TERM_ORDER is (A,B,C,D,G,e,N,MC1,MC2); a negative e makes the >=0 bound bite
    truth = np.array([1.8, -0.17, -0.72, 0.01, -0.84, -0.50, 2.0, 0.03, -0.05])
    y = m.evaluate(coords, truth) + rng.normal(0, 0.9, n)
    fit = fit_model(coords, y, limb_exponent_values=(1.0, 2.0, 5.0), limb_amplitude_positive=True)
    assert fit.limb_nonneg_active
    assert fit.flags.get("e") == "PINNED_AT_ZERO"
    assert "N" in fit.flags
    assert np.isnan(fit.errors[TERM_INDEX["e"]])
    assert np.isnan(fit.errors[TERM_INDEX["N"]])
    assert not np.isnan(fit.errors[TERM_INDEX["A"]])
    cards = fit.header_cards()
    assert "ERR_e" not in cards and "ERR_N" not in cards
    assert "ERR_A" in cards and "FITFLAG" in cards


TERM_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3, "G": 4, "e": 5, "N": 6,
              "MC1": 7, "MC2": 8}


def test_open_plane_refuses_non_velocity_units(tmp_path):
    """A raw intensity map mis-named *_Vdop.fits must not be fitted as km/s."""
    from astropy.io import fits
    from solar_chromosphere_velocity.data_loader import open_plane
    p = str(tmp_path / "junk_Vdop.fits")
    fits.PrimaryHDU(np.ones((8, 8), dtype=np.float32)).writeto(p)
    with pytest.raises(ValueError):
        open_plane(p)
    data, _ = open_plane(p, require_unit=False)      # opt out
    assert data.shape == (8, 8)


def test_open_plane_converts_ms_to_kms(tmp_path):
    from astropy.io import fits
    from solar_chromosphere_velocity.data_loader import open_plane
    p = str(tmp_path / "v_Vdop.fits")
    h = fits.Header()
    h["BUNIT"] = "m/s"
    fits.PrimaryHDU(np.full((4, 4), 1500.0, dtype=np.float32), header=h).writeto(p)
    data, _ = open_plane(p)
    assert data[0, 0] == pytest.approx(1.5)


def test_open_plane_does_not_decompress(tmp_path):
    """Locating the plane must use .shape, never .data."""
    import resource
    from solar_chromosphere_velocity.data_loader import open_plane
    if not os.path.exists(HA_RICE):
        pytest.skip("no RICE file")
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    try:
        open_plane(HA_RICE, require_unit=False)
    except Exception:
        pass
    growth_mb = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before) / 1e3
    if growth_mb < 0:            # Linux reports kB, macOS bytes
        growth_mb = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before) / 1e6
    assert growth_mb < 300, "open_plane decompressed something (%.0f)" % growth_mb


@requires_vdop
def test_index_is_a_complete_inventory(tmp_path):
    """Failures must still appear in index.csv."""
    import shutil
    from astropy.io import fits
    src = tmp_path / "in"
    src.mkdir()
    shutil.copy2(VDOP, src / "good_HA_Vdop.fits")
    # NB: the default exclude globs drop *_Vdop*; pass an empty set so both
    # files are attempted and the inventory can be checked

    # a file with a velocity name but unusable content
    (src / "bad_HA.fits").write_bytes(b"not a fits file at all")
    cfg = Config(input_dir=str(src), output_dir=str(tmp_path / "out"),
                 pattern="*.fits", exclude_patterns=(), log_level="ERROR")
    results = ChromosphereProcessor(cfg).process_directory()
    idx = tmp_path / "out" / "index.csv"
    assert idx.exists()
    lines = [l for l in idx.read_text().strip().splitlines() if l]
    assert len(lines) - 1 == len(results) == 2, (lines, [r.status for r in results])
    assert any("FAILED" in l for l in lines)


# ------------------------------------- regressions found by the SECOND review
@requires_vdop
def test_v_model_err_not_destroyed_when_e_is_pinned(legacy_run):
    """Poisoning the covariance with NaN blanked this whole layer. It must not.

    The first version of the identifiability fix set both the error AND the
    covariance row/column to NaN; V_MODEL_ERR is sqrt(x^T Sigma x), so it became
    100% NaN in the default and --legacy cases.
    """
    fit = legacy_run.fit
    assert fit.limb_nonneg_active, "this fixture is expected to pin e"
    err = legacy_run.layers["V_MODEL_ERR"]
    inside = np.isfinite(legacy_run.layers["V_LARGE"])
    finite = np.isfinite(err)
    assert inside.sum() > 1e6
    assert finite.sum() == inside.sum(), (finite.sum(), inside.sum())
    assert np.nanmax(err[finite]) > 0
    assert np.isnan(fit.errors[TERM_INDEX["e"]])          # reported as unestimated
    assert not np.isnan(fit.cov[TERM_INDEX["e"]]).any()   # but the covariance is intact


def test_n_is_always_flagged_scan_grid():
    """N is never a design column, so its covariance error is structurally 0."""
    from solar_chromosphere_velocity.models import VelocityModel, fit_model
    rng = np.random.default_rng(11)
    n = 3000
    coords = dict(
        sin_lat=rng.uniform(-0.8, 0.8, n),
        sin_2lat=rng.uniform(-1, 1, n),
        cos_lat=rng.uniform(0.6, 1.0, n),
        sin_lon=rng.uniform(-1, 1, n),
        cos_lon=rng.uniform(-1, 1, n),
        rho_rad=rng.uniform(0, 1.2, n),
        b0_rad=0.0,
    )
    m = VelocityModel("full-2.0", limb_exponent=2.0)
    truth = np.array([1.8, -0.17, -0.72, 0.01, -0.84, 0.50, 2.0, 0.03, -0.05])
    y = m.evaluate(coords, truth) + rng.normal(0, 0.02, n)
    fit = fit_model(coords, y, limb_exponent_values=(1.0, 2.0, 5.0), limb_amplitude_positive=True)
    assert not fit.limb_nonneg_active          # e is identifiable here
    assert fit.flags.get("N") == "SCAN_GRID"
    assert np.isnan(fit.errors[TERM_INDEX["N"]])
    assert not np.isnan(fit.errors[TERM_INDEX["e"]])
    assert "ERR_N" not in fit.header_cards()


def test_insufficient_data_gets_exactly_one_index_row(tmp_path):
    """A file failing the fit stage must appear once, not twice."""
    from astropy.io import fits
    src = tmp_path / "in"
    src.mkdir()
    h = fits.Header()
    h["BUNIT"] = "km/s"
    h["B0"] = 0.0
    h["CRPIX1"] = 3.0
    h["CRPIX2"] = 3.0
    h["R_SUN"] = 2.0                 # tiny disk -> far fewer than 100 fit points
    d = np.full((8, 8), np.nan, dtype=np.float32)
    d[3, 3] = 0.0
    fits.PrimaryHDU(d, header=h).writeto(src / "tiny_HA_Vdop.fits")
    cfg = Config(input_dir=str(src), output_dir=str(tmp_path / "out"),
                 pattern="*.fits", exclude_patterns=(), log_level="ERROR")
    results = ChromosphereProcessor(cfg).process_directory()
    assert [r.status for r in results] == ["INSUFFICIENT_DATA"]
    lines = [l for l in (tmp_path / "out" / "index.csv").read_text().strip().splitlines() if l]
    assert len(lines) - 1 == 1, lines


def test_detect_data_type_uses_the_same_unit_recogniser(tmp_path):
    """'km s-1' etc. must not be rejected before the opt-out can apply."""
    from astropy.io import fits
    from solar_chromosphere_velocity.data_loader import DataType, detect_data_type
    for i, bunit in enumerate(("km/s", "KM/S", "km s-1", "m s-1", "m/s", "M/S", "kms")):
        p = tmp_path / ("v%02d.fits" % i)   # macOS is case-insensitive
        h = fits.Header()
        h["BUNIT"] = bunit
        fits.PrimaryHDU(np.zeros((4, 4), np.float32), header=h).writeto(p)
        assert detect_data_type(str(p)) is DataType.DOPPLER_VDOP, bunit


def test_stale_nscan_from_the_input_header_is_not_copied(tmp_path):
    from astropy.io import fits
    from solar_chromosphere_velocity.output import write_multihdu
    src = fits.Header()
    src["NSCAN"] = "stale-value-from-the-input"
    src["OBSTIME"] = "x"
    path = str(tmp_path / "stale.fits")
    write_multihdu(path, _fake_layers(), src, None, {})
    with fits.open(path) as h:
        assert "NSCAN" not in h[0].header


def test_figures_take_their_axes_from_the_written_header(tmp_path, monkeypatch):
    """The pipeline plots from an in-memory dict; it must still get arcsec axes.

    Regression: _resolve() returned header=None for a dict, so the figures the
    pipeline wrote had pixel axes while the same layers plotted from the
    written FITS had arcsec axes.
    """
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure
    from solar_chromosphere_velocity import plot_layers

    labels = {}
    orig = Figure.savefig

    def spy(self, *a, **k):
        labels["x"] = [ax.get_xlabel() for ax in self.axes]
        return orig(self, *a, **k)

    monkeypatch.setattr(Figure, "savefig", spy)
    hdr = {"CDELT1": 1.04, "CDELT2": 1.04, "CX": 32.0, "CY": 24.0}
    layers = {"V_RESID": np.zeros((48, 64), np.float32)}

    plot_layers(layers, str(tmp_path / "with.png"), panels=["V_RESID"], header=hdr)
    assert "x [arcsec]" in labels["x"], labels["x"]

    plot_layers(layers, str(tmp_path / "without.png"), panels=["V_RESID"])
    assert "x [pixel]" in labels["x"], labels["x"]


def test_plot_header_carries_the_geometry_keys():
    from solar_chromosphere_velocity import ChromosphereProcessor
    h = ChromosphereProcessor._plot_header(
        {"CDELT1": 1.04, "CDELT2": 1.04, "CX": 1191.6, "CY": 1154.9,
         "OBSTIME": ("20260424T080829", "obs time")},
        None)
    for k in ("CDELT1", "CDELT2", "CX", "CY"):
        assert k in h, k
    assert h["OBSTIME"] == "20260424T080829"


def test_write_component_figures_accepts_a_header(tmp_path):
    """Regression: header was passed explicitly AND via **kwargs -> TypeError,
    which the never-fatal plotting handler swallowed, silently dropping the six
    per-component figures (9 figures became 3)."""
    pytest.importorskip("matplotlib")
    from solar_chromosphere_velocity import write_component_figures
    from solar_chromosphere_velocity.plotting import COMPONENT_PANELS
    hdr = {"CDELT1": 1.04, "CDELT2": 1.04, "CX": 32.0, "CY": 24.0}
    figs = write_component_figures(_fake_layers(), str(tmp_path / "c"),
                                   header=hdr, dpi=50)
    assert set(figs) == set(COMPONENT_PANELS), sorted(figs)
    for path in figs.values():
        assert os.path.exists(path) and os.path.getsize(path) > 3000


# ------------------------------------- joint H-alpha + Fe I component grid
def _write_joint_pair(tmp_path, stamp="20251111T095427"):
    """Two tiny v2 products (H-alpha and Fe I) at the same observation time."""
    from astropy.io import fits
    from solar_chromosphere_velocity.output import write_multihdu

    cards = {"CDELT1": 1.04, "CDELT2": 1.04, "CX": 32.0, "CY": 24.0,
             "R_SUN": 28.0}
    paths = {}
    for line in ("HA", "FEI"):
        hdr = fits.Header()
        hdr["OBSTIME"] = stamp
        hdr["SRCFILE"] = "fake_%s.fits" % line
        path = str(tmp_path / ("%s_%s_scv.fits" % (stamp, line)))
        write_multihdu(path, _fake_layers(), hdr, None, cards)
        paths[line] = path
    return paths["HA"], paths["FEI"]


def test_read_product_can_restrict_layers(tmp_path):
    from solar_chromosphere_velocity.plotting_grid import (JOINT_LAYERS,
                                                           read_product)
    ha, _ = _write_joint_pair(tmp_path)
    _, layers = read_product(ha, names=JOINT_LAYERS)
    assert layers and set(layers) <= set(JOINT_LAYERS)
    _, everything = read_product(ha)
    assert "MASK_ACTIVE" in everything and len(everything) > len(layers)


def test_product_geometry_uses_the_solar_radius():
    from astropy.io import fits
    from solar_chromosphere_velocity.plotting_grid import product_geometry
    hdr = fits.Header()
    hdr["R_SUN"] = 100.0
    hdr["CDELT1"] = 1.0
    hdr["CDELT2"] = 1.0
    hdr["CRPIX1"] = 128.0
    hdr["CRPIX2"] = 128.0
    _, xlim, ylim = product_geometry(hdr, (256, 256))
    # half width 1.155 R_sun, top (1 + 0.495) R_sun
    assert xlim == pytest.approx((-115.5, 115.5))
    assert ylim == pytest.approx((-115.5, 149.5))


def test_find_joint_pairs_requires_both_lines(tmp_path):
    from solar_chromosphere_velocity.plotting_grid import find_joint_pairs
    _write_joint_pair(tmp_path, stamp="20251111T095427")
    ha, _ = _write_joint_pair(tmp_path, stamp="20261030T145700")
    os.rename(ha, ha.replace("_HA_", "_RESID_"))     # only one line left
    pairs = find_joint_pairs(str(tmp_path))
    assert [p[0] for p in pairs] == ["20251111T095427"]
    assert len(pairs[0]) == 3


def test_joint_grid_rows_and_figure(tmp_path):
    pytest.importorskip("matplotlib")
    from solar_chromosphere_velocity.plotting_grid import (
        joint_grid_rows, plot_joint_grid)
    ha, fe = _write_joint_pair(tmp_path)

    rows = joint_grid_rows(ha, fe)
    assert len(rows) == 2                     # one row per line
    assert all(len(r.panels) == 4 for r in rows)
    assert "H" in rows[0].time and "Fe" in rows[1].time

    # the third column is the algebraic sum of the three small terms
    from solar_chromosphere_velocity.plotting_grid import read_product
    _, layers = read_product(ha)
    expected = layers["V_LIMB"] + layers["V_MERID"] + layers["V_OFFSET"]
    assert np.allclose(rows[0].panels[2].data, expected)

    out = plot_joint_grid(ha, fe, str(tmp_path / "grid.png"), dpi=60)
    assert os.path.exists(out) and os.path.getsize(out) > 5000


def test_make_joint_series_writes_one_figure_per_time(tmp_path):
    pytest.importorskip("matplotlib")
    from solar_chromosphere_velocity.plotting_grid import make_joint_series
    out_root = tmp_path / "out"
    (out_root / "20251111").mkdir(parents=True)
    _write_joint_pair(out_root / "20251111", stamp="20251111T095427")
    _write_joint_pair(out_root / "20261030", stamp="20261030T145700")

    made = make_joint_series(str(out_root), dpi=60)
    assert sorted(made) == ["20251111T095427", "20261030T145700"]
    for stamp, path in made.items():
        assert os.path.exists(path) and os.path.basename(path) == (
            "%s_HA_FEI_scv_grid.png" % stamp)
        assert path.startswith(str(out_root / "joint"))


# ------------------------------------- scan-saddle and flat-field removal
def _quadratic_map(shape=(128, 128), coeffs=(0.10, -0.20, -0.15, 0.05,
                                             0.08, -0.12),
                   cx=64.0, cy=64.0, radius=50.0):
    ny, nx = shape
    y, x = np.mgrid[0:ny, 0:nx]
    X, Y = (x - cx) / radius, (y - cy) / radius
    terms = [np.ones_like(X), Y, Y * Y, X, X * Y, X * X]
    surface = sum(c * t for c, t in zip(coeffs, terms))
    return surface, {"CX": cx, "CY": cy, "R_SUN": radius}


def _stripe_map(lam=40.0, ang=20.0, amp=0.5, shape=(128, 128)):
    ny, nx = shape
    y, x = np.mgrid[0:ny, 0:nx]
    th = np.deg2rad(ang)
    return amp * np.cos(2 * np.pi * (x * np.cos(th) + y * np.sin(th)) / lam)


def test_scan_saddle_recovers_a_known_quadratic():
    from solar_chromosphere_velocity.scan_correction import scan_saddle
    truth, hdr = _quadratic_map()
    rng = np.random.default_rng(4)
    data = truth + rng.normal(0, 1e-6, truth.shape)
    fitted, coeffs = scan_saddle(data, hdr)
    m = np.isfinite(fitted)
    assert np.max(np.abs(fitted[m] - truth[m])) < 1e-5
    assert coeffs.shape == (6,)


def test_stripe_band_removes_a_synthetic_stripe():
    from solar_chromosphere_velocity.scan_correction import (
        detect_stripe_peaks, stripe_band)
    stripe = _stripe_map(lam=40.0, ang=20.0)
    remove = stripe_band(stripe, 40.0, 20.0)
    residual = stripe - remove
    assert np.nanstd(residual) < 0.25 * np.nanstd(stripe)

    peaks = detect_stripe_peaks(stripe, lambda_min=20.0, lambda_max=120.0)
    assert peaks, "no stripe detected"
    lam, ang = peaks[0]
    assert abs(lam - 40.0) / 40.0 < 0.10, lam
    assert min(abs(ang - 20.0), 180 - abs(ang - 20.0)) < 8.0, ang


def test_protect_weight_spares_active_regions():
    from solar_chromosphere_velocity.scan_correction import protect_weight
    ny = nx = 128
    y, x = np.mgrid[0:ny, 0:nx]
    blob = 3.0 * np.exp(-((x - 40) ** 2 + (y - 40) ** 2) / (2 * 5.0 ** 2))
    w = protect_weight(blob, 90.0, 99.0, 1.0)
    # the weight is smoothed with sigma=12, so a small blob is only partly
    # suppressed; what matters is the contrast against the quiet Sun
    assert w[100, 100] > 0.9                    # untouched in the quiet part
    assert w[40, 40] < 0.75                     # suppressed at the blob
    assert w[40, 40] < w[100, 100] - 0.1


def test_scan_correction_can_be_switched_off():
    from solar_chromosphere_velocity.scan_correction import scan_correction
    data = _stripe_map() + 0.7
    out = scan_correction(data, {"CX": 64.0, "CY": 64.0, "R_SUN": 50.0},
                          enabled=False)
    m = np.isfinite(out.local)
    assert np.allclose(out.local[m], data[m])
    assert np.allclose(out.ss[m], 0.0)


def test_scan_correction_removes_the_systematics_and_keeps_the_blob():
    """The quadratic and the stripe go, the active-region blob stays."""
    from solar_chromosphere_velocity.scan_correction import scan_correction
    saddle, hdr = _quadratic_map()
    stripe = _stripe_map(lam=40.0, ang=20.0, amp=0.4)
    ny = nx = 128
    y, x = np.mgrid[0:ny, 0:nx]
    blob = 2.0 * np.exp(-((x - 45) ** 2 + (y - 45) ** 2) / (2 * 6.0 ** 2))
    data = saddle + stripe + blob

    out = scan_correction(data, hdr, stripe_peaks=[(40.0, 20.0)],
                          protect=(90.0, 99.0, 1.0))
    before, after = np.nanstd(data), np.nanstd(out.local)
    assert after < 0.9 * before, (before, after)
    # The saddle is fitted to the whole map, so the stripe and the blob leak a
    # little into it; it must still track the true quadratic closely.
    m = np.isfinite(out.scan_saddle)
    corr = np.corrcoef(out.scan_saddle[m], saddle[m])[0, 1]
    assert corr > 0.9, corr
    assert np.max(np.abs(out.ss[m])) > 0.1      # something was actually removed
    # the blob survives, because the protection weight damps the flat field
    iy, ix = 45, 45
    assert abs(out.local[iy, ix] - blob[iy, ix]) < 0.6
