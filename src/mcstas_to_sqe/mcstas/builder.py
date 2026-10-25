"""Build the McStasScript instrument of any :class:`.Instrument`.

The instrument parameters are added first, then the components one by one in beam order, since
McStas positions are relative to earlier components. Each component is:

- added as its ``component_name``, named after its attribute in the instrument, at its
  ``at``/``rotated``/``relative``;
- given its McStas parameters (``Component.mcstas_parameters``: its fields plus what it
  computes). String parameters are quoted here.

A field McStas does not know is left out, since the component only computes from it. A computed
parameter McStas does not know raises.

Coordinate system: McStas, z along the beam, y up, x to the left of the beam.
"""

from dataclasses import asdict, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcstasscript.interface import instr

from mcstas_to_sqe import default_work_dir

if TYPE_CHECKING:
    from mcstas_to_sqe.core import Component, Instrument


def default_input_dir(instrument: "Instrument", work_dir: str | Path | None = None) -> Path:
    """Return the folder for the McStas-generated files of an instrument.

    This is McStasScript's ``input_path``, where mcrun runs, and holds the .instr, .c and
    compiled binary. Each instrument gets its own folder so they do not overwrite each other.

    Parameters
    ----------
    instrument:
        The instrument.
    work_dir:
        Work folder. If ``None``, :func:`.default_work_dir` is used.

    Returns
    -------
    :
        ``work_dir/mcstas_input/<instrument name>``.
    """
    return Path(work_dir or default_work_dir()) / "mcstas_input" / instrument.name


def _formatted_parameters(
    mcstas_comp: Any, component: "Component", params: dict[str, Any]
) -> dict[str, Any]:
    """Return the McStas parameters of a component that its McStasScript component takes.

    Each entry of ``params`` (from ``component.mcstas_parameters``) must be one of:

    - a McStas parameter: kept, quoted if it is a string;
    - a field McStas does not take: left out, as the component only computes from it (e.g. the
      detector's ``theta_bins``, put into its ``options``).

    Anything else was computed under a name McStas does not take, likely a typo.

    Returns
    -------
    :
        The values to set, by McStas parameter name.

    Raises
    ------
    ValueError
        Naming every computed parameter McStas does not take.
    """
    mcstas_names = set(mcstas_comp.parameter_names)  # what the McStas component takes
    field_names = {f.name for f in fields(component)}
    if unknown := sorted(params.keys() - mcstas_names - field_names):
        raise ValueError(
            f"{component.name} ({component.component_name}): no McStas parameter {unknown}"
        )

    valid = {}
    for name, value in params.items():
        if name not in mcstas_names:
            continue  # a field the component computes from: not for McStas
        if mcstas_comp.parameter_types[name] == "string":
            value = f'"{value}"'  # McStas takes a string as a quoted C string
        valid[name] = value
    return valid


def build_mcstas(
    instrument: "Instrument", input_path: str | Path | None = None
) -> instr.McStas_instr:
    """Build the McStasScript instrument of an instrument.

    Parameters
    ----------
    instrument:
        The instrument to build.
    input_path:
        Folder for the McStas files, created if missing. If ``None``,
        :func:`default_input_dir` is used.

    Returns
    -------
    :
        The McStasScript instrument, with the instrument parameters and all components.
    """
    input_path = Path(input_path or default_input_dir(instrument))
    input_path.mkdir(parents=True, exist_ok=True)

    mcstas_instr = instr.McStas_instr(instrument.name, input_path=str(input_path))
    # a run sets its own values with `set_parameters`
    for param in instrument.instr_params:
        mcstas_instr.add_parameter(**asdict(param))

    for name, component in instrument.components.items():
        mcstas_comp = mcstas_instr.add_component(
            name,
            component.component_name,
            AT=list(component.at),
            ROTATED=list(component.rotated),
            RELATIVE=component.relative or "ABSOLUTE",  # of both AT and ROTATED
        )
        params = component.mcstas_parameters(instrument)  # its fields, and what it computes
        mcstas_comp.set_parameters(_formatted_parameters(mcstas_comp, component, params))
    return mcstas_instr
