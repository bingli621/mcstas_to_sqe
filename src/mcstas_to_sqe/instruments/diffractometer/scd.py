"""The single-crystal diffractometer.

Beam path: source -> slit -> incident monitor -> goniometer (three nested arms) with the
sample -> detector.

:class:`SingleCrystalDiffractometer` creates its default components around a given sample
(lengths in m, angles in deg). Its control parameters are :class:`Params`: ``ei`` and the
goniometer angles. The McStas instrument parameters (``instr_params``: the wavelength of ``ei``,
and the angles) are computed from them when used.

The instrument knows no dicts, files or McStas: io/ converts it to and from its nested dict and
a YAML file, and mcstas/builder.py builds the McStas instrument from it.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcstas_to_sqe.core.base import Instrument
from mcstas_to_sqe.core.components import (
    Arm,
    Monitor,
    Monitor_nD,
    Progress_bar,
    Single_crystal,
    Slit,
    Source_simple,
)
from mcstas_to_sqe.core.params import InstrParam, param


@dataclass(frozen=True)
class Params:
    """Control parameters of the diffractometer.

    Frozen: change them with ``SingleCrystalDiffractometer.update_params``. Each field's unit and
    comment are in its metadata.

    Parameters
    ----------
    ei:
        Incident energy in meV.
    omega:
        Goniometer rotation about +y, in deg.
    chi:
        Goniometer rotation about -z, in deg.
    phi:
        Goniometer rotation about +x, in deg.
    """

    ei: float = param(14.0, "meV", "incident energy")
    omega: float = param(0.0, "deg", "goniometer rotation about +Y")
    chi: float = param(0.0, "deg", "goniometer rotation about -Z")
    phi: float = param(0.0, "deg", "goniometer rotation about +X")


DEFAULT_PARAMS = Params()  # frozen, so one shared default is safe


class SingleCrystalDiffractometer(Instrument):
    """The single-crystal diffractometer, with its default components.

    Lengths are in m and angles in deg. A run may give its own goniometer angles.

    Parameters
    ----------
    sample:
        The sample, e.g. ``la2ni7(at=[0, 0, 0], relative="gonio_phi")`` from samples/.
        If ``None``, there is no sample.
    params:
        Control parameters, e.g. ``Params(ei=25, omega=30)``.
    """

    params_class = Params
    name = "scd"  # the McStas instrument name

    # the components, for type hints; `__init__` sets them, and its order is the beam's
    origin: Progress_bar
    source: Source_simple
    slit: Slit
    incident_monitor: Monitor
    sample_pos: Arm
    gonio_omega: Arm
    gonio_chi: Arm
    gonio_phi: Arm
    sample: Single_crystal | None
    detector: Monitor_nD

    def __init__(
        self, sample: Single_crystal | None = None, params: Params = DEFAULT_PARAMS
    ) -> None:
        super().__init__(params)
        self.origin = Progress_bar(at=[0, 0, 0])
        self.source = Source_simple(
            radius=0.03,
            focus_xw=0.01,  # m
            focus_yh=0.01,  # m
            focus_on="sample_pos",  # the component the source focuses on
            flux=1.0e10,
            dlambda=0,  # monochromatic
            at=[0, 0, 0],
            relative="origin",
        )
        self.slit = Slit(xwidth=0.03, yheight=0.03, at=[0, 0, 1.0], relative="source")
        # integrated incident intensity, 10 cm before the sample; the neutrons carry on
        self.incident_monitor = Monitor(
            xwidth=0.02, yheight=0.02, restore_neutron=1, at=[0, 0, 0.9], relative="slit"
        )
        # the beam hits the goniometer centre; it can be dumped here
        self.sample_pos = Arm(at=[0, 0, 0.1], relative="incident_monitor")
        # goniometer arms, outermost first; each is rotated by an instrument parameter
        self.gonio_omega = Arm(at=[0, 0, 0], rotated=[0, "omega", 0], relative="sample_pos")  # +Y
        self.gonio_chi = Arm(at=[0, 0, 0], rotated=[0, 0, "-chi"], relative="gonio_omega")  # -Z
        self.gonio_phi = Arm(at=[0, 0, 0], rotated=["phi", 0, 0], relative="gonio_chi")  # +X
        # the sample, in the innermost goniometer arm; None: none
        self.sample = sample
        # cylindrical ("banana") detector around the vertical axis through the sample
        self.detector = Monitor_nD(
            radius=1.0,
            theta_limits=[-125.0, -5.0],  # signed: negative = right of the beam
            theta_bins=1200,
            ymin=-0.25,  # 0.5 m high, centred on the beam
            ymax=0.25,
            y_bins=50,
            options="banana, list all, pixel id, t",  # an event list mcstastox can read
            filename="detector",
            nexus_bins=1,
            restore_neutron=1,
            at=[0, 0, 0],
            relative="sample_pos",
        )

    @property
    def instr_params(self) -> list[InstrParam]:
        """McStas instrument parameters: the wavelength of ``ei``, and the goniometer angles."""
        p = self._params
        wavelength = Source_simple.wavelength_from_ei(p.ei)
        return [
            InstrParam("wavelength", wavelength, "angstrom", "neutron wavelength"),
            InstrParam("omega", p.omega, "deg", "goniometer rotation about +Y"),
            InstrParam("chi", p.chi, "deg", "goniometer rotation about -Z"),
            InstrParam("phi", p.phi, "deg", "goniometer rotation about +X"),
        ]

    def draw(
        self,
        projections: tuple[str, ...] = ("zx", "zy"),
        path: str | Path | None = None,
        **kwargs: Any,
    ) -> list[Path]:
        """Show, or save to ``path``, the top and side views at the current angles.

        Drawn by McStasScript; see :func:`.display.draw`.
        """
        import tempfile

        from mcstas_to_sqe.mcstas.builder import build_mcstas
        from mcstas_to_sqe.mcstas.display import draw

        with tempfile.TemporaryDirectory() as folder:  # McStasScript writes its files here
            return draw(build_mcstas(self, input_path=folder), projections, path, **kwargs)
