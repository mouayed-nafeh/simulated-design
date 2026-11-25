"""Example of a nonlinear response history analysis (NRHA) of a simulated
building under a bi-directional ground motion.

Set RECORD_X and RECORD_Y to single-column text files of the horizontal
ground acceleration components (in g) with time step DT. If they are not
found, a synthetic ground motion is used for demonstration.
"""
from pathlib import Path
import sys
import numpy as np

# Add the src directory to sys.path
sys.path.append(str(Path(__file__).parents[1]))

from simdesign import rcmrf  # noqa
from simdesign.utils.misc import make_dir  # noqa

# Ground motion inputs
RECORD_X = Path("record_x.txt")
RECORD_Y = Path("record_y.txt")
DT = 0.01  # time step of the records (s)
SCALE_FACTOR = 1.0

# Output directory
outdir = Path(__file__).parents[1] / "tmp/Outputs-NRHA"
make_dir(outdir)


def synthetic_record(pga: float, seed: int, duration: float = 15.0
                     ) -> np.ndarray:
    """Band-limited, enveloped white noise scaled to the given PGA (g)."""
    t = np.arange(0.0, duration, DT)
    rng = np.random.default_rng(seed)
    acc = np.convolve(rng.standard_normal(t.size), np.ones(5) / 5, 'same')
    acc *= np.minimum(1.0, t / 2.0) * np.exp(-np.maximum(0.0, t - 6.0) / 3)
    return acc / np.abs(acc).max() * pga


if RECORD_X.exists() and RECORD_Y.exists():
    acc_x, acc_y = np.loadtxt(RECORD_X), np.loadtxt(RECORD_Y)
else:
    print("Records not found, using a synthetic ground motion.")
    acc_x, acc_y = synthetic_record(0.3, 1), synthetic_record(0.25, 2)

# Generate and design a building
bcim = rcmrf.BCIM()
bcim.generate(design_class="eu_cdl", sample_size=1, num_storeys=4, beta=0.1)
bdim = rcmrf.BDIM(bcim.taxonomy[0])
bdim.set_seed_for_quality_adjustments(bcim.inputs.seed)
bdim.run_iterative_design_algorithm()
if not bdim.ok:
    raise RuntimeError("The design did not converge.")

# Nonlinear model, with infills
bnsm = rcmrf.BNSM(design=bdim, model="CP01", include_infills=True)

# Run the NRHA, with 5 s of free vibration for the residual drifts
res = bnsm.do_nrha(
    acc_x=acc_x, acc_y=acc_y, dt=DT, scale_factor=SCALE_FACTOR,
    damping_ratio=0.05, damping_modes=(1, 3), free_vibration=5.0,
    collapse_drift=0.10, out_dir=outdir, save_histories=True,
)

print(f"Converged: {res['ok'] == 0}, collapse: {res['collapse']}, "
      f"time: {res['time']:.2f} / {res['duration']:.2f} s")
print(f"Damping periods: {np.round(res['periods'], 3)} s")
print(f"Max peak drift: {res['max_peak_drift']:.4f} "
      f"({res['max_peak_drift_dir']}, storey {res['max_peak_drift_loc']})")
print("Peak drifts (X, Y):\n", np.round(res['peak_drift'], 4))
print("Peak floor accelerations (g, X, Y), ground first:\n",
      np.round(res['peak_accel'], 3))
print("Residual drifts (X, Y):\n", np.round(res['residual_drift'], 5))
