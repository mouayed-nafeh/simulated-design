"""Tests of the storey-level response (storey drifts and shears) recorded in
the pushover analyses of the BNSM models."""
import sys
from pathlib import Path
import numpy as np
import pytest

# Add the project root to sys.path
sys.path.append(str(Path(__file__).parents[1]))

from simdesign import rcmrf  # noqa
from simdesign.rcmrf.bnsm.baselib.constants import (  # noqa
    NSPA_EQUILIBRIUM_TOL)


@pytest.mark.parametrize('model, infills', [
    ('CP01', False), ('CP01', True), ('CP03', False), ('DP02', False)])
@pytest.mark.parametrize('ctrl_dof', [1, 2])
def test_storey_curves(designs, model, infills, ctrl_dof):
    """Storey drifts and shears are consistent with the pushover curve."""
    bnsm = rcmrf.BNSM(design=designs['eu_cdl'], model=model,
                      include_infills=infills, max_drift=0.03)
    ctrl_disp, base_shear, ok = bnsm.do_nspa(ctrl_dof)
    curves = bnsm.storey_curves['xy'[ctrl_dof - 1]]
    n_steps, n = len(base_shear), len(bnsm.floors)
    assert ok == 0
    for key in ('floor_disp', 'storey_drift', 'storey_shear'):
        assert curves[key].shape == (n_steps, n)
        assert np.all(curves[key][0] == 0.0)  # gravity state
    assert curves['storey_heights'].shape == (n,)
    assert np.all(curves['storey_heights'] > 0)
    assert curves['floor_masses'].shape == (n,)
    assert np.all(curves['floor_masses'] > 0)
    assert curves['ok'] == ok
    assert curves['base_shear'] == pytest.approx(base_shear)
    assert curves['ctrl_disp'] == pytest.approx(ctrl_disp)

    # Ground storey shear is the base shear (sum of support reactions)
    shear = curves['storey_shear']
    assert np.all(np.abs(shear[:, 0] - curves['base_shear'])
                  <= NSPA_EQUILIBRIUM_TOL * np.maximum(shear[:, 0], 1.0))
    # Storey shears decrease with height and are positive when loaded
    assert np.all(np.diff(shear, axis=1) <= 1e-9)
    assert np.all(shear[1:] > 0)
    # Drifts times heights add up to the roof displacement
    roof = (curves['storey_drift'] * curves['storey_heights']).sum(axis=1)
    assert roof == pytest.approx(curves['floor_disp'][:, -1], abs=1e-12)
    # The frame is pushed in the loading direction
    peak = np.argmax(curves['base_shear'])
    assert np.all(curves['storey_drift'][peak] > 0)


def test_storey_curves_both_directions_and_files(designs, tmp_path):
    """Both directions are kept, and the curves are saved to files."""
    bnsm = rcmrf.BNSM(design=designs['eu_cdl'], model='CP01',
                      include_infills=False, max_drift=0.02)
    assert bnsm.storey_curves == {}
    bnsm.do_nspa(1, out_dir=tmp_path)
    bnsm.do_nspa(2, out_dir=tmp_path)
    assert set(bnsm.storey_curves) == {'x', 'y'}
    for d in 'xy':
        curves = bnsm.storey_curves[d]
        drift = np.loadtxt(tmp_path / f'storey_drifts_{d}.out')
        shear = np.loadtxt(tmp_path / f'storey_shears_{d}.out')
        assert drift == pytest.approx(curves['storey_drift'], abs=1e-8)
        assert shear == pytest.approx(curves['storey_shear'], abs=1e-8)
