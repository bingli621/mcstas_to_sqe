import pytest

from mcstas_to_sqe.core import Slit
from mcstas_to_sqe.instruments import SingleCrystalDiffractometer
from mcstas_to_sqe.io.dict_io import _components_from_dict, from_dict, merge, to_dict

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


def test_a_complete_dict_must_name_its_instrument():
    data = to_dict(SingleCrystalDiffractometer())
    assert from_dict(data) == SingleCrystalDiffractometer()
    del data["instrument"]
    with pytest.raises(ValueError, match="instrument: None"):
        from_dict(data)


@pytest.mark.parametrize("name", ["positions", "update_params", "components", "_params"])
def test_a_section_named_like_an_attribute_of_the_instrument_is_refused(name):
    data = to_dict(SingleCrystalDiffractometer())
    data[name] = {"component": "Arm", "at": [0, 0, 0]}
    with pytest.raises(ValueError, match=f"{name}: reserved name"):
        from_dict(data)
