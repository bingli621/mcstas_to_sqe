"""Draw any instrument with McStasScript."""

from typing import Any

from mcstasscript.interface.instr import McStas_instr


def draw(inst: McStas_instr, projections: tuple[str, ...] = ("zx", "zy"), **kwargs: Any) -> None:
    """Show a McStasScript instrument at its parameters' values, with its ``show_instrument``.

    Parameters
    ----------
    inst:
        The McStasScript instrument.
    projections:
        The views, one figure each, as McStasScript names them: ``"zx"`` from above (the beam
        across, +x up), ``"zy"`` from the side, ``"xy"`` along the beam.
    **kwargs:
        Passed on to ``show_instrument``, e.g. ``width`` and ``height`` (in pixels) or
        ``limits`` (``[xmin, xmax, ymin, ymax]``, in m).
    """
    for projection in projections:
        inst.show_instrument(backend="matplotlib_2d", projection=projection, **kwargs)
