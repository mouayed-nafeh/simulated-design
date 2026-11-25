"""Tests of the nonlinear response history analysis (NRHA) of the BNSM
models, under short synthetic ground motions."""
import sys
from pathlib import Path
import numpy as np
import pytest

# Add the project root to sys.path
sys.path.append(str(Path(__file__).parents[1]))

from simdesign import rcmrf  # noqa

DT = 0.01


def ground_motion(pga: float, seed: int, duration: float = 3.0
                  ) -> np.ndarray:
    """Band-limited, enveloped white noise scaled to the given PGA (g)."""
    t = np.arange(0.0, duration, DT)
    rng = np.random.default_rng(seed)
    acc = np.convolve(rng.standard_normal(t.size), np.ones(5) / 5, 'same')
    acc *= np.minimum(1.0, t / 0.5) * np.exp(-np.maximum(0.0, t - 1.5))
    return acc / np.abs(acc).max() * pga


@pytest.fixture(scope='module')
def bnsm(designs):
    """Bare CP01 model of the low-code design."""
    return rcmrf.BNSM(design=designs['eu_cdl'], model='CP01',
                      include_infills=False)


@pytest.mark.parametrize('model', ['CP01', 'CP03', 'DP02'])
@pytest.mark.parametrize('infills', [False, True])
def test_nrha_outputs(designs, model, infills):
    """The analysis runs to the end and gives finite results of the
    expected shapes, with the ground motion PGA at the base."""
    acc_x, acc_y = ground_motion(0.2, 1), ground_motion(0.2, 2)
    bnsm = rcmrf.BNSM(design=designs['eu_cdl'], model=model,
                      include_infills=infills)
    res = bnsm.do_nrha(acc_x=acc_x, acc_y=acc_y, dt=DT)
    n = len(bnsm.floors)
    assert res['ok'] == 0
    assert not res['collapse']
    assert res['time'] == pytest.approx(res['duration'])
    assert res['peak_drift'].shape == (n, 2)
    assert res['residual_drift'].shape == (n, 2)
    assert res['peak_disp'].shape == (n, 2)
    assert res['peak_accel'].shape == (n + 1, 2)
    assert res['storey_heights'].shape == (n,)
    for key in ('peak_drift', 'peak_disp', 'peak_accel', 'peak_base_shear'):
        assert np.all(np.isfinite(res[key]))
        assert np.all(res[key] > 0)
    assert res['peak_accel'][0] == pytest.approx(
        [np.abs(acc_x).max(), np.abs(acc_y).max()], rel=1e-6)
    assert 0 < res['max_peak_drift'] < 0.05
    assert res['max_peak_drift'] == res['peak_drift'].max()
    assert res['max_peak_drift_dir'] in ('X', 'Y')
    assert 1 <= res['max_peak_drift_loc'] <= n
    assert 'histories' not in res


def test_rayleigh_damping_ratio(bnsm):
    """The Rayleigh coefficients give the damping ratio at the periods of
    both damping modes."""
    res = bnsm.do_nrha(acc_x=ground_motion(0.01, 1, 1.0), dt=DT,
                       damping_ratio=0.03, damping_modes=(1, 2))
    alpha_m, beta_k = res['rayleigh']
    for period in res['periods']:
        omega = 2 * np.pi / period
        assert alpha_m / (2 * omega) + beta_k * omega / 2 == pytest.approx(
            0.03)
    assert res['periods'][0] > res['periods'][1]


def test_elastic_response_is_linear(bnsm):
    """Under very small ground motions, the response scales linearly."""
    acc = ground_motion(1.0, 3)
    small = bnsm.do_nrha(acc_x=acc, dt=DT, scale_factor=0.001)
    double = bnsm.do_nrha(acc_x=acc, dt=DT, scale_factor=0.002)
    assert double['peak_drift'][:, 0] == pytest.approx(
        2 * small['peak_drift'][:, 0], rel=0.02)
    assert double['peak_base_shear'][0] == pytest.approx(
        2 * small['peak_base_shear'][0], rel=0.02)


def test_acceleration_units(bnsm):
    """Records in m/s2 give the same response as the same records in g."""
    acc = ground_motion(0.1, 4)
    in_g = bnsm.do_nrha(acc_x=acc, dt=DT)
    in_si = bnsm.do_nrha(acc_x=acc * 9.81, dt=DT, acc_units='m/s2')
    assert in_si['peak_drift'] == pytest.approx(in_g['peak_drift'])
    assert in_si['peak_accel'] == pytest.approx(in_g['peak_accel'])


def test_unidirectional_excitation(bnsm):
    """Excitation in Y only drives the response mainly in Y."""
    res = bnsm.do_nrha(acc_y=ground_motion(0.2, 5), dt=DT)
    assert res['peak_accel'][0, 0] == 0.0
    assert np.all(res['peak_drift'][:, 1] > 5 * res['peak_drift'][:, 0])
    assert res['max_peak_drift_dir'] == 'Y'


def test_collapse_drift_stops_the_analysis(bnsm):
    """The analysis stops once the collapse drift is reached."""
    res = bnsm.do_nrha(acc_x=ground_motion(0.3, 6), dt=DT,
                       collapse_drift=0.002)
    assert res['ok'] == 0
    assert res['collapse']
    assert res['time'] < res['duration']
    assert res['max_peak_drift'] >= 0.002


def test_rigid_joint_links_are_not_damped(designs):
    """CP03 (rigid joint links) and CP01 (no links) give similar responses.
    Stiffness-proportional damping of the rigid links would suppress the
    response of CP03."""
    acc = ground_motion(0.2, 7)
    drifts = [
        rcmrf.BNSM(design=designs['eu_cdl'], model=model,
                   include_infills=False).do_nrha(
            acc_x=acc, dt=DT)['max_peak_drift']
        for model in ('CP01', 'CP03')
    ]
    assert drifts[1] == pytest.approx(drifts[0], rel=0.2)


def test_histories_and_saved_outputs(bnsm, tmp_path):
    """Histories are consistent with the peak values and saved to files."""
    res = bnsm.do_nrha(acc_x=ground_motion(0.1, 8), dt=DT,
                       free_vibration=1.0, out_dir=tmp_path,
                       save_histories=True)
    hist = res['histories']
    n_steps, n = hist['time'].size, len(bnsm.floors)
    assert res['time'] == pytest.approx(res['duration'])
    assert res['duration'] == pytest.approx(2.99 + 1.0)
    assert hist['floor_disp'].shape == (n_steps, n, 2)
    assert hist['floor_accel'].shape == (n_steps, n, 2)
    assert hist['drift'].shape == (n_steps, n, 2)
    assert hist['ground_accel'].shape == (n_steps, 2)
    assert hist['base_shear'].shape == (n_steps, 2)
    assert np.abs(hist['drift']).max(axis=0) == pytest.approx(
        res['peak_drift'])
    assert hist['drift'][-1] == pytest.approx(res['residual_drift'])
    assert np.abs(hist['ground_accel']).max(axis=0)[0] == pytest.approx(
        res['peak_accel'][0, 0])
    for name in ('peak_drift', 'peak_accel', 'residual_drift', 'time',
                 'floor_disp_x', 'floor_accel_y', 'log_nrha'):
        assert (tmp_path / f'{name}.out').exists()
    assert np.loadtxt(tmp_path / 'peak_drift.out') == pytest.approx(
        res['peak_drift'], abs=1e-8)


@pytest.mark.parametrize('kwargs', [
    {},  # no ground motion
    {'acc_x': [0.1, 0.2], 'dt': 0.0},
    {'acc_x': [0.1]},
    {'acc_x': [0.1, np.nan]},
    {'acc_x': [0.1, 0.2], 'acc_units': 'cm/s2'},
    {'acc_x': [0.1, 0.2], 'damping_ratio': 1.0},
    {'acc_x': [0.1, 0.2], 'damping_modes': (1, 1)},
    {'acc_x': [0.1, 0.2], 'damping_modes': (0, 2)},
    {'acc_x': [0.1, 0.2], 'damping_stiffness': 'tangent'},
    {'acc_x': [0.1, 0.2], 'dt_analysis': -0.01},
    {'acc_x': [0.1, 0.2], 'free_vibration': -1.0},
    {'acc_x': [0.1, 0.2], 'collapse_drift': 0.0},
])
def test_invalid_inputs_raise(bnsm, kwargs):
    """Invalid inputs raise ValueError before the model is built."""
    kwargs.setdefault('dt', DT)
    with pytest.raises(ValueError):
        bnsm.do_nrha(**kwargs)
