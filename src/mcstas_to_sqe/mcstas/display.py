"""Draw any instrument with McStasScript."""

import warnings
from pathlib import Path
from typing import Any

from mcstasscript.interface.instr import McStas_instr


def draw(
    inst: McStas_instr,
    projections: tuple[str, ...] = ("zx", "zy"),
    path: str | Path | None = None,
    **kwargs: Any,
) -> list[Path]:
    """Show a McStasScript instrument at its parameters' values, with its ``show_instrument``.

    Parameters
    ----------
    inst:
        The McStasScript instrument.
    projections:
        The views, one figure each, as McStasScript names them: ``"zx"`` from above (the beam
        across, +x up), ``"zy"`` from the side, ``"xy"`` along the beam.
    path:
        If given, the figures are saved instead of shown, one file per view:
        ``<stem>_<projection><suffix>``, e.g. ``top_view_zx.png`` for ``top_view.png``, in the
        format of the suffix.
    **kwargs:
        Passed on to ``show_instrument``, e.g. ``width`` and ``height`` (in pixels) or
        ``limits`` (``[xmin, xmax, ymin, ymax]``, in m).

    Returns
    -------
    :
        The files saved, one per view; empty if ``path`` is ``None``.
    """
    if path is None:
        for projection in projections:
            inst.show_instrument(backend="matplotlib_2d", projection=projection, **kwargs)
        return []

    import matplotlib.pyplot as plt  # only to save: drawing alone does not need it

    path = Path(path)
    saved = []
    backend = plt.get_backend()
    plt.switch_backend("Agg")  # show_instrument's plt.show() then leaves the figure open
    try:
        for projection in projections:
            with warnings.catch_warnings():  # Agg cannot show: expected, it only saves
                warnings.filterwarnings("ignore", "FigureCanvasAgg is non-interactive")
                inst.show_instrument(backend="matplotlib_2d", projection=projection, **kwargs)
            file = path.with_name(f"{path.stem}_{projection}{path.suffix}")
            plt.gcf().savefig(file, bbox_inches="tight")
            plt.close("all")
            saved.append(file)
    finally:
        plt.switch_backend(backend)
    return saved
