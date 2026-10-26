"""Base class of every instrument.

An instrument is two things:

- its components, one attribute each, in beam order. The attribute name is the McStas instance
  name and the name a component's ``relative`` refers to.
- its control parameters, ``params``. This is a frozen dataclass, the class's ``params_class``,
  whose fields carry a unit and a comment in their metadata (see :func:`.params.param`).

``INSTRUMENTS`` maps every subclass's class name to the class. A parameters file's
``instrument:`` entry names one of them.

The io/ and mcstas/ layers rely only on what this base class provides: ``name``, ``params``,
``update_params``, ``components``, ``instr_params``, ``positions``, equality and the summary
(``repr``).
"""

from dataclasses import asdict, fields, replace
from typing import Any

from .components import Component
from .geometry import place_absolute
from .params import InstrParam
from .utils import format_value

INSTRUMENTS: dict[str, type] = {}  # every subclass of Instrument by its class name


class Instrument:
    """Base class of an instrument made of components.

    Where a component takes a number, or a list of numbers such as ``at`` and ``rotated``, it may
    instead be given the name of an instrument parameter, or an expression of them such as
    ``"-chi"``, as in McStas.

    Parameters
    ----------
    params:
        Control parameters, an instance of :attr:`params_class`.
        If ``None``, the defaults of :attr:`params_class` are used.

    Notes
    -----
    A subclass, e.g. ``SingleCrystalDiffractometer``, must:

    - set the class attributes :attr:`params_class` and :attr:`name`;
    - create its components in ``__init__`` in beam order, after calling
      ``super().__init__(params)``. ``__init__`` must accept ``params`` as a keyword, since io/
      creates instruments with that argument only;
    - implement :attr:`instr_params`.
    """

    params_class: type  # the frozen dataclass of the control parameters: a subclass sets it
    name: str  # the McStas instrument name: a subclass sets it (an instance may change its own)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register the subclass in ``INSTRUMENTS`` under its class name."""
        super().__init_subclass__(**kwargs)
        INSTRUMENTS[cls.__name__] = cls

    def __setattr__(self, attr: str, value: Any) -> None:
        """Set an attribute, naming a component after the attribute it is set on.

        Raises
        ------
        ValueError
            If ``value`` is a component and ``attr`` is already taken by something that is not
            a component: a method, a property or another attribute of the instrument (e.g.
            ``positions``, ``name``, ``_params``). The component would replace it.
        """
        if isinstance(value, Component):
            current = vars(self).get(attr)
            taken = hasattr(type(self), attr) or not isinstance(current, Component | None)
            if taken:
                raise ValueError(
                    f"{attr}: reserved name, already used by {type(self).__name__}; "
                    "give the component another name"
                )
            value.name = attr
        super().__setattr__(attr, value)

    def __init__(self, params: Any | None = None) -> None:
        self._params = self.params_class() if params is None else params

    @property
    def params(self) -> Any:
        """Control parameters, an instance of :attr:`params_class`.

        They are frozen; use :meth:`update_params` to change them.
        """
        return self._params

    def update_params(self, **values: Any) -> None:
        """Change control parameters.

        Parameters
        ----------
        **values:
            New values by parameter name, e.g. ``update_params(ei=25)``.

        Raises
        ------
        TypeError
            If a name is not a field of :attr:`params_class`.
        """
        self._params = replace(self._params, **values)

    @property
    def instr_params(self) -> list[InstrParam]:
        """McStas instrument parameters, computed from :attr:`params`.

        A subclass must implement this.
        """
        raise NotImplementedError

    @property
    def components(self) -> dict[str, Component]:
        """Components by name, in beam order.

        These are the attributes that hold a component, in the order ``__init__`` first sets
        them. An attribute set to ``None`` is left out.
        """
        return {n: v for n, v in vars(self).items() if isinstance(v, Component)}

    def positions(self) -> dict[str, list[float]]:
        """Compute the position of every component in the lab frame.

        Returns
        -------
        :
            ``[x, y, z]`` in m by component name, at the current angles. Values are rounded to
            1 nm so that rotation noise reads ``0.0``, never ``-0.0``.
        """
        instr_param_dict = {p.name: p.value for p in self.instr_params}
        at_rotated_dict = place_absolute(
            self.components, instr_param_dict
        )  # name -> (position, rotation)
        return {
            name: (position.round(9) + 0.0).tolist()
            for name, (position, _rotation) in at_rotated_dict.items()
        }

    def __repr__(self) -> str:
        """Summarize the instrument as tables.

        The tables show the params, the McStas instrument parameters, and the components in beam
        order with their placement and lab-frame position. A component's own ``repr`` shows its
        values.
        """
        from tabulate import tabulate  # only when printed: importing does not need it

        values = asdict(self._params)
        params = [
            (f.name, format_value(values[f.name]), f.metadata["unit"], f.metadata["comment"])
            for f in fields(self._params)
        ]
        instr_params = [
            (p.name, format_value(p.value), p.unit, p.comment) for p in self.instr_params
        ]
        positions = self.positions()
        components = []
        for name, c in self.components.items():
            where = (
                format_value(c.at),
                format_value(c.rotated),
                c.relative or "-",
                format_value(positions[name]),
            )
            components.append((name, c.component_name, *where))
        return "\n".join(
            [
                f"{type(self).__name__} {self.name!r}",
                "",
                "Params:",
                tabulate(params, ("param", "value", "unit", "comment")),
                "",
                "Instrument parameters (McStas):",
                tabulate(instr_params, ("name", "value", "unit", "comment")),
                "",
                "Components, in the order of the beam (m, deg):",
                tabulate(
                    components, ("name", "component", "at", "rotated", "relative", "position")
                ),
            ]
        )

    def __eq__(self, other: object) -> bool:  # also makes it unhashable: its components change
        if type(other) is not type(self):
            return False
        mine = (self.name, self._params, self.components)
        return mine == (other.name, other._params, other.components)
