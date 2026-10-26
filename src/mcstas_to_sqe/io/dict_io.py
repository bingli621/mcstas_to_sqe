"""Convert an instrument to and from its nested dict: what a parameters file holds, in any format.

The dict names the instrument class (``instrument``, a key of ``INSTRUMENTS``). Its ``params``
are those of the class's ``params_class``, and every other key is a component section.

- :func:`to_dict`: the complete dict of an instrument.
- :func:`from_dict`: the instrument of a complete dict, of the class it names.
- :func:`from_overrides`: the default instrument of the class a dict names, with the dict's
  values laid over it. This is what a parameters file holds.
- :func:`merge`: how one dict is laid over another.

Every dict, complete or of overrides, must name an instrument class: there is no default.

Layout, for the diffractometer as in its params.yaml (instruments/diffractometer/)::

    instrument: SingleCrystalDiffractometer   # the class: a key of INSTRUMENTS
    name: scd
    params:                   # each param: its value, unit and comment (its field's metadata)
      ei: {value: 14.0, unit: meV, comment: incident energy}
      ...
    slit:                     # one section per component, in beam order
      component: Slit         # the McStas component (the class of the same name)
      xwidth: 0.01            # its own parameters
      yheight: 0.01
      at: [0, 0, 1.0]         # where it sits: McStas AT, ROTATED, RELATIVE
      rotated: [0, 0, 0]
      relative: source
      position: [0.0, 0.0, 1.0]   # in the lab frame: written for reading, never read back
    ...

Rules:

- A param's unit, if given, must match its unit in the code. Its comment is only for reading.
- A bare number stands for the value: ``ei: 25`` is ``ei: {value: 25}``.
- A component value that an instrument parameter provides is the parameter's name, as in McStas
  (``rotated: [0, omega, 0]``).
- The components are built in the order of their sections, which is the order McStas traces
  them. A component's ``relative`` must name a component of an earlier section, as in McStas.
  ``position`` plays no part in the order.

Errors raise ``ValueError`` saying where: ``instrument: ...``, ``params.ei: ...``,
``slit: ...``. They cover an unknown or missing instrument class; an unknown param, unit,
component class or component parameter; and a ``relative`` that names no earlier component.
"""

import copy
from dataclasses import asdict, fields
from typing import Any

from mcstas_to_sqe import instruments  # noqa: F401  (adds every instrument to INSTRUMENTS)
from mcstas_to_sqe.core import INSTRUMENTS, Component, Instrument
from mcstas_to_sqe.core.components import COMPONENTS


def to_dict(instrument: Instrument) -> dict:
    """Convert an instrument to its complete nested dict.

    Parameters
    ----------
    instrument:
        The instrument.

    Returns
    -------
    :
        The class name (``instrument``), ``name``, ``params`` and one section per component,
        as :func:`from_dict` takes it.
    """
    return {
        "instrument": type(instrument).__name__,
        "name": instrument.name,
        "params": _params_to_dict(instrument.params),
        **_components_to_dict(instrument),
    }


def from_dict(data: dict) -> Instrument:
    """Create an instrument from its complete nested dict.

    The instrument is created with ``params`` alone, and its components are then replaced by
    those of the sections, in the order of the sections: that is the beam order. A component of
    the class that has no section is left out (set to ``None``, as the diffractometer's sample
    when there is none). The lab-frame ``position`` a section may hold is not read.

    Parameters
    ----------
    data:
        The complete dict, as :func:`to_dict` returns it. ``data`` is not changed.

    Returns
    -------
    :
        An instrument of the class ``data["instrument"]`` names.

    Raises
    ------
    ValueError
        If the instrument class is missing or unknown; if a param, a unit, a component class
        or a component parameter is unknown; if a component lacks a required parameter; or if
        a component's ``relative`` names no component of an earlier section.
    KeyError
        If ``data`` has no ``params`` or no ``name``.
    """
    instrument_class = _instrument_class(data)
    data = copy.deepcopy(data)  # the components get its lists: not shared with the caller
    params = _params_from_dict(data["params"], instrument_class.params_class)
    components = _components_from_dict(data)
    _check_relatives(components)
    instrument = instrument_class(params=params)
    instrument.name = data["name"]
    _set_components(instrument, components)
    return instrument


def from_overrides(overrides: dict, **kwargs: Any) -> Instrument:
    """Create the default instrument of a class, with overrides laid over it.

    A value in ``overrides`` wins. A key it leaves out or sets to ``None`` keeps the default
    (see :func:`merge`). Use :func:`from_dict` for a complete dict instead.

    Parameters
    ----------
    overrides:
        Names the instrument class and holds only what differs, e.g.
        ``{"instrument": "SingleCrystalDiffractometer", "params": {"omega": 30}}``.
    **kwargs:
        Passed to the class to create the default instrument; for the diffractometer,
        ``sample`` and ``params``.

    Returns
    -------
    :
        The instrument.

    Raises
    ------
    ValueError
        As :func:`from_dict`.
    """
    instrument_class = _instrument_class(overrides)
    defaults = to_dict(instrument_class(**kwargs))  # names the class: from_dict follows it
    return from_dict(merge(defaults, overrides))


def merge(defaults: dict, overrides: dict) -> dict:
    """Lay one nested dict over another.

    For each key of ``overrides``:

    - a dict over a dict is merged the same way, key by key;
    - ``None`` (null in a file) keeps the default. It is stored only for a key ``defaults``
      lacks;
    - anything else, a list included, replaces the default whole.

    A key that ``overrides`` leaves out keeps its default.

    The keys of ``defaults`` keep their order. A key that only ``overrides`` has is put right
    after the key before it in ``overrides`` (first, if it is the first key there). So a
    component section a parameters file adds lands where the file puts it in the beam: after
    the section written above it.

    Parameters
    ----------
    defaults:
        The dict to lay over. Not changed.
    overrides:
        The dict laid over it. Not changed.

    Returns
    -------
    :
        A new merged dict.

    Examples
    --------
    >>> merge({"a": 1, "b": {"c": 2, "d": 3}}, {"b": {"c": 5}, "a": None})
    {'a': 1, 'b': {'c': 5, 'd': 3}}
    >>> merge({"a": 1, "c": 3}, {"a": 1, "b": 2})
    {'a': 1, 'b': 2, 'c': 3}
    """
    merged = copy.deepcopy(defaults)
    previous = None  # the key before `key` in `overrides`, already in `merged`
    for key, override in overrides.items():
        if key not in merged:
            merged = _insert_after(merged, previous, key, copy.deepcopy(override))
        elif isinstance(override, dict) and isinstance(merged[key], dict):
            merged[key] = merge(merged[key], override)
        elif override is not None:
            merged[key] = copy.deepcopy(override)
        previous = key
    return merged


def _insert_after(data: dict, after: Any, key: Any, value: Any) -> dict:
    """Return a copy of ``data`` with ``key`` inserted right after ``after`` (first if None)."""
    items = list(data.items())
    index = 0 if after is None else list(data).index(after) + 1
    items.insert(index, (key, value))
    return dict(items)


def _instrument_class(data: dict) -> type[Instrument]:
    """Return the class that ``data["instrument"]`` names, a key of ``INSTRUMENTS``.

    Raises
    ------
    ValueError
        If ``data`` names no class, or an unknown one.
    """
    named = data.get("instrument")
    if named not in INSTRUMENTS:
        raise ValueError(f"instrument: {named!r} is no instrument (known: {sorted(INSTRUMENTS)})")
    return INSTRUMENTS[named]


# the params section


def _params_to_dict(params: Any) -> dict:
    """Convert a params dataclass to ``{name: {"value", "unit", "comment"}}``."""
    return {
        f.name: {
            "value": getattr(params, f.name),
            "unit": f.metadata["unit"],
            "comment": f.metadata["comment"],
        }
        for f in fields(params)
    }


def _params_from_dict(section: dict, params_class: type) -> Any:
    """Create a ``params_class`` from a params section.

    An entry is as :func:`_params_to_dict` writes it, or a bare value. A comment is not read.

    Raises
    ------
    ValueError
        If ``section`` names a param that ``params_class`` lacks, or a unit that differs from
        the param's unit in the code.
    """
    unknown = set(section) - {f.name for f in fields(params_class)}
    if unknown:
        raise ValueError(f"params: no such param {sorted(unknown)}")
    values = {}
    for f in fields(params_class):
        if f.name not in section:
            continue
        entry = section[f.name]
        if not isinstance(entry, dict):
            entry = {"value": entry}
        unit = entry.get("unit")
        if unit is not None and unit != f.metadata["unit"]:
            raise ValueError(f"params.{f.name}: unit {unit!r}, but it is {f.metadata['unit']!r}")
        values[f.name] = entry["value"]
    return params_class(**values)


# the component sections


def _components_to_dict(instrument: Instrument) -> dict:
    """Convert the components to sections, keyed by name, in beam order.

    Each section holds ``component``, the component's parameters (unset optional ones left out),
    ``at``, ``rotated``, ``relative``, and last its lab-frame ``position`` in m at the
    instrument's params. :func:`_components_from_dict` ignores ``position``.

    The placement comes after the component's own parameters on purpose, although the dataclass
    fields start with it (from ``Component``). That way a section reads as params.yaml does:
    what the component is, then where it sits. :func:`from_dict` does not depend on the order.
    """
    positions = instrument.positions()
    sections = {}
    for name, component in instrument.components.items():
        values = asdict(component)
        placement = {key: values.pop(key) for key in ("at", "rotated", "relative")}  # moved last
        sections[name] = {
            "component": component.component_name,
            **{key: value for key, value in values.items() if value is not None},
            **placement,
            "position": positions[name],
        }
    return sections


def _components_from_dict(data: dict, component_classes: dict = COMPONENTS) -> dict:
    """Create the components of a dict, keyed by name, in the order of their sections.

    Every key but ``instrument``, ``name`` and ``params`` is a section. Each section's
    ``component`` names its class in ``component_classes`` (e.g. ``"Slit"``).

    Raises
    ------
    ValueError
        If a section names no known class, or gives a parameter the class does not take.
    """
    built = {}
    for name, section in data.items():
        if name in ("instrument", "name", "params"):  # the keys that are not components
            continue
        kwargs = {
            key: value for key, value in section.items() if key not in ("component", "position")
        }
        component_class = component_classes.get(section.get("component"))
        if component_class is None:
            raise ValueError(
                f"{name}: no such component {section.get('component')!r}"
                f" (known: {sorted(component_classes)})"
            )
        try:
            built[name] = component_class(**kwargs)
        except TypeError as error:  # a parameter the class does not take, or one it lacks
            raise ValueError(f"{name}: {error}") from error
    return built


def _check_relatives(components: dict[str, Component]) -> None:
    """Check that every ``relative`` names a component that comes earlier, as McStas requires.

    Parameters
    ----------
    components:
        The components by name, in beam order.

    Raises
    ------
    ValueError
        Naming the component whose ``relative`` names no component, or a later one.
    """
    earlier = set()
    for name, component in components.items():
        relative = component.relative
        if relative is not None and relative not in earlier:
            where = "comes after it" if relative in components else "is no component"
            raise ValueError(f"{name}: relative {relative!r} {where}")
        earlier.add(name)


def _set_components(instrument: Instrument, components: dict[str, Component]) -> None:
    """Make ``components``, in their order, the components of ``instrument``.

    An instrument's components are its attributes in the order they were first set, so its own
    components (and a placeholder left at ``None``, such as the diffractometer's sample) are
    removed first and the new ones set in order. One of its own that ``components`` lacks is set
    back to ``None``, after them.
    """
    own = {n: v for n, v in vars(instrument).items() if isinstance(v, Component) or v is None}
    for name in own:
        delattr(instrument, name)
    for name, component in components.items():
        setattr(instrument, name, component)
    for name in own.keys() - components.keys():
        setattr(instrument, name, None)
