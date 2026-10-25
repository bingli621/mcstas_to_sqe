# mcstas_to_sqe

[![CI](https://github.com/bingli621/mcstas_to_sqe/actions/workflows/ci.yml/badge.svg)](https://github.com/bingli621/mcstas_to_sqe/actions/workflows/ci.yml)

Neutron instruments described as components, built and simulated in McStas (McStasScript), with
tools to read the McStas NeXus output (McStasToX, scippneutron) and convert it to momentum transfer.
It comes with a single-crystal diffractometer with a three-axis goniometer, and ready-made samples.

The diffractometer with La2Ni7 at omega 30 deg, seen from above, its goniometer arms as axes, drawn by
McStasScript (`inst.show_instrument(backend="matplotlib_2d", projection="zx")`;
`projection="zy"` for the view from the side):

![Top view of the diffractometer](output/top_view_zx.png)

An omega scan of La2Ni7 over 132 points, in momentum transfer, normalized (`scripts/plot_scan_Q.py`):

![Normalized Q map of an omega scan](output/scan_Q_normalized_132pts.gif)

## Use

```bash
pixi run python scripts/build_instrument.py   # the .instr file and its diagram
pixi run python scripts/run_mcstas.py         # simulate (edit MODE: a run, a scan, or a split run)
pixi run python scripts/plot_scan_2d.py       # output/scan.gif
pixi run python scripts/plot_scan_Q.py        # output/scan_Q.gif
pixi run python scripts/print_peaks.py        # peak list of the newest scan
pixi run pytest --cov=mcstas_to_sqe           # tests and coverage
pixi run pre-commit install                   # once: ruff format on every commit
```

```python
import numpy as np

from mcstas_to_sqe.instruments import Params, SingleCrystalDiffractometer
from mcstas_to_sqe.io import load, save
from mcstas_to_sqe.mcstas.simulation import Simulation
from mcstas_to_sqe.reduction import find_peaks, read_scan
from mcstas_to_sqe.samples import la2ni7

# an instrument: made in code, or the default with a file laid over it
d = SingleCrystalDiffractometer(
    la2ni7(mosaic=60, at=[0, 0, 0], relative="gonio_phi"), Params(ei=14)
)
d = load("my_params.yaml", sample=d.sample, params=Params(ei=14))
d.slit.xwidth = 0.02
save(d, "copy.yaml")

# simulate a scan and read it back
sim = Simulation(d)
h5 = sim.scan("omega", np.linspace(-48, 73, 122), chi=-5.893, phi=0)
points = read_scan(h5.parent)  # a (parameters, events) per scan point
peaks = find_peaks(h5.parent)

# a split run: source to sample once, then the crystal for many angles in parallel
sim.run_to_sample(ncount=1e8)
sim.scan_from_sample("omega", range(-48, 83), chi=-5.893, phi=0)
```

## Configuration

An instrument is a set of components, plain attributes (`d.slit.xwidth`); positions and
reciprocal vectors are computed when used. `print(d)` summarizes it.

- **Params** (`ei` in meV; `omega`, `chi`, `phi` in deg) are a frozen dataclass given when the
  instrument is made; change them with `d.update_params(ei=25)`, or per run with
  `Simulation(d).run(omega=30)`.
- **Instrument parameters** are what McStas gets (the wavelength and the goniometer angles),
  derived from the params. Components may refer to them by name, as in McStas
  (`rotated: [0, omega, 0]`).
- **Parameter files** name their instrument and hold only what differs from its default; a value
  left out or `null` keeps the default. The full default is
  [`instruments/diffractometer/params.yaml`](src/mcstas_to_sqe/instruments/diffractometer/params.yaml).

```yaml
# my_params.yaml
instrument: SingleCrystalDiffractometer
source:
  radius: 0.02
detector:
  radius: 0.8
```

## Layout

```
src/mcstas_to_sqe/
  core/          entities shared by any instrument: Instrument, components, params, geometry
  instruments/   the instruments, one package each (diffractometer/)
  samples/       ready-made samples (aluminum, la2ni7) and their CIF files
  io/            instruments to and from dicts and YAML files
  mcstas/        build the McStasScript instrument, run simulations, read NeXus output, run folders,
                 show the top and side views
  reduction/     to Q, sample frame, solid-angle normalization, find_peaks
scripts/         build, run, plot and print peaks
tests/
mcstas_input/    generated McStas files, per instrument (git-ignored)
mcstas_output/   simulation results, a folder per run (git-ignored)
output/          figures and peak lists
```

Dependencies point inwards: `core` imports nothing from the other layers; `instruments` and
`samples` build on it; `io`, `mcstas` and `reduction` are adapters that work with any
instrument.

Outputs go to the work folder: the current folder for the library, the repository for the
scripts. Set `MCSTAS_TO_SQE_WORKDIR` or pass `Simulation(d, work_dir=...)` to move them.
`mcstas.run_folders` lists, finds and cleans runs (`list_runs`, `latest_run`, `clean_runs`).

`mcstastox` is installed from its `entry_choice` branch, which can read individual scan entries.
