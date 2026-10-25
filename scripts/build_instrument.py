"""Build the McStas instrument of the single-crystal diffractometer (La2Ni7 sample), show it,
and write its .instr file to mcstas_input/<instrument name>/ of the work folder (work_dir.py).
run_mcstas.py simulates it.

In VS Code: run this file, or from a notebook/Interactive Window:

    from build_instrument import create_instrument, create_mcstas, plot_instrument
    scd = create_instrument()
    inst = create_mcstas(scd, plot=False)
    inst.show_diagram()
    plot_instrument(inst)
"""

from work_dir import WORK_DIR

from mcstas_to_sqe.instruments import Params, SingleCrystalDiffractometer
from mcstas_to_sqe.mcstas.builder import build_mcstas, default_input_dir
from mcstas_to_sqe.mcstas.display import draw
from mcstas_to_sqe.samples import la2ni7


def create_instrument(params=None):
    """Return the single-crystal diffractometer with the La2Ni7 sample, printing the sample,
    the instrument, its parameters and its source."""
    # the sample, in the innermost goniometer arm
    sample = la2ni7(mosaic=60, at=[0, 0, 0], relative="gonio_phi")
    print(sample)

    scd = SingleCrystalDiffractometer(sample, params=params or Params())
    print(scd)
    print(scd.params)
    print(scd.source)

    return scd


def create_mcstas(instrument, input_path=None, plot=True):
    """Return the McStasScript instrument of `instrument`, its .instr file written to
    `input_path` (None for mcstas_input/<instrument name>/). Print its parameters, components and instrument
    file, and with `plot` draw its diagram."""
    inst = build_mcstas(
        instrument, input_path=input_path or default_input_dir(instrument, WORK_DIR)
    )
    inst.write_full_instrument()  # show_instrument_file reads the file on disk: write it first
    inst.show_parameters()
    inst.show_components()
    inst.show_instrument_file(line_numbers=True)  # reads the file written above

    if plot:
        inst.show_diagram()

    return inst


def plot_instrument(instrument, **kwargs):
    """Show `instrument` from above and from the side with McStasScript."""
    draw(instrument, **kwargs)


if __name__ == "__main__":
    scd = create_instrument(params=Params(omega=30))
    inst = create_mcstas(scd, plot=False)
    plot_instrument(inst)
