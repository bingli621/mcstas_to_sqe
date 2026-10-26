import pytest

from mcstas_to_sqe.core import Slit
from mcstas_to_sqe.instruments import SingleCrystalDiffractometer
from mcstas_to_sqe.io.dict_io import _components_from_dict, from_dict, merge, to_dict
from mcstas_to_sqe.samples import aluminum

DATA = {
    "name": "tiny",
    "slit": {
        "component": "Slit",
        "xwidth": 0.5,
        "yheight": 0.5,
        "at": [0, 0, 1],
        "rotated": [0, 0, 0],
        "relative": None,
    },
}


def test_a_component_is_made_of_its_section_with_where_it_sits():
    slit = _components_from_dict(DATA, {"Slit": Slit})["slit"]
    assert (slit.xwidth, slit.at) == (0.5, [0, 0, 1])


def test_the_order_is_the_order_of_the_sections():
    data = {
        "b": {"component": "Slit", "xwidth": 1, "yheight": 1, "at": [0, 0, 0]},
        "name": "two",
        "a": {"component": "Slit", "xwidth": 2, "yheight": 2, "at": [0, 0, 1], "relative": "b"},
    }
    assert list(_components_from_dict(data, {"Slit": Slit})) == ["b", "a"]


def test_merge_lays_one_dict_over_another():
    base = {"a": {"x": 1, "y": 2}, "b": [1, 2]}
    assert merge(base, {"a": {"x": 5, "y": None}, "b": [3]}) == {"a": {"x": 5, "y": 2}, "b": [3]}
    assert base == {"a": {"x": 1, "y": 2}, "b": [1, 2]}  # not changed


def test_a_wrong_component_says_which_section():
    with pytest.raises(ValueError, match="slit: no such component 'Slt'"):
        _components_from_dict({"slit": {"component": "Slt", "at": [0, 0, 0]}})
    with pytest.raises(ValueError, match="slit: .*xwidht"):
        _components_from_dict(
            {"slit": {"component": "Slit", "xwidht": 1, "yheight": 1, "at": [0, 0, 0]}}
        )


EXTRA_SLIT = {
    "component": "Slit",
    "xwidth": 0.01,
    "yheight": 0.01,
    "at": [0, 0, 0.5],
    "relative": "source",
}


def _with_section_after(data, after, name, section):
    """`data` with the section `name` inserted right after the key `after`."""
    items = list(data.items())
    items.insert(list(data).index(after) + 1, (name, section))
    return dict(items)


def test_a_new_component_is_built_where_its_section_is():
    data = to_dict(SingleCrystalDiffractometer())
    data = _with_section_after(data, "source", "extra_slit", EXTRA_SLIT)
    assert list(from_dict(data).components)[:4] == ["origin", "source", "extra_slit", "slit"]


def test_the_components_are_built_in_the_order_of_the_sections_not_of_the_class():
    data = to_dict(SingleCrystalDiffractometer(aluminum(at=[0, 0, 0], relative="gonio_phi")))
    sample = data.pop("sample")
    data["sample"] = sample  # after the detector now: McStas would trace it there
    assert list(from_dict(data).components)[-2:] == ["detector", "sample"]


def test_the_position_written_in_a_section_plays_no_part_in_the_order():
    data = to_dict(SingleCrystalDiffractometer())
    data["slit"]["position"] = [0, 0, 99.0]  # far downstream: ignored
    assert list(from_dict(data).components) == list(SingleCrystalDiffractometer().components)


def test_a_component_of_the_class_without_a_section_is_left_out():
    data = to_dict(SingleCrystalDiffractometer())
    del data["detector"]  # no other component is placed relative to it
    d = from_dict(data)
    assert "detector" not in d.components and d.detector is None


@pytest.mark.parametrize(
    ("relative", "message"),
    [("sorce", "extra_slit: relative 'sorce' is no component"), ("detector", "comes after it")],
)
def test_a_relative_must_name_an_earlier_component(relative, message):
    data = to_dict(SingleCrystalDiffractometer())
    data = _with_section_after(data, "source", "extra_slit", EXTRA_SLIT | {"relative": relative})
    with pytest.raises(ValueError, match=message):
        from_dict(data)


def test_merge_puts_a_new_key_after_the_key_before_it_in_the_overrides():
    defaults = {"a": 1, "b": 2, "c": 3}
    assert list(merge(defaults, {"b": 5, "new": 0})) == ["a", "b", "new", "c"]
    assert list(merge(defaults, {"new": 0, "c": 5})) == ["new", "a", "b", "c"]  # first: first
    assert list(merge(defaults, {"c": 5, "a": 5})) == ["a", "b", "c"]  # the defaults' order kept


def test_a_complete_dict_must_name_its_instrument():
    data = to_dict(SingleCrystalDiffractometer())
    assert from_dict(data) == SingleCrystalDiffractometer()
    del data["instrument"]
    with pytest.raises(ValueError, match="instrument: None"):
        from_dict(data)
