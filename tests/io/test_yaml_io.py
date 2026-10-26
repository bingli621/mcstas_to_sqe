from dataclasses import dataclass

import pytest

from mcstas_to_sqe.core import InstrParam, Instrument, Progress_bar, Slit
from mcstas_to_sqe.core.params import param
from mcstas_to_sqe.instruments import PARAMS_FILE, SingleCrystalDiffractometer
from mcstas_to_sqe.io import load, read_yaml, save, to_dict, write_yaml
from mcstas_to_sqe.samples import aluminum, la2ni7


def test_write_then_read_gives_the_same_dict(tmp_path):
    data = {"name": "x", "a": {"b": [1, 2.5, None]}}
    write_yaml(data, tmp_path / "x.yaml")
    assert read_yaml(tmp_path / "x.yaml") == data


def test_the_diffractometer_has_no_sample_by_default():
    sample = la2ni7(at=[0, 0, 0], relative="gonio_phi")
    assert SingleCrystalDiffractometer().sample is None
    assert "sample" not in SingleCrystalDiffractometer().components
    assert SingleCrystalDiffractometer(sample).sample is sample  # as it is


def test_params_yaml_is_the_default_diffractometer_saved():
    """The reference file is the built one with an aluminium sample: laid over the one with no
    sample it gives the one the code makes with it, and over that one it changes nothing."""
    al = aluminum(at=[0, 0, 0], relative="gonio_phi")
    assert load(PARAMS_FILE) == SingleCrystalDiffractometer(al)
    assert load(PARAMS_FILE, sample=al) == SingleCrystalDiffractometer(al)


def test_a_file_is_laid_over_the_code(tmp_path):
    write_yaml(
        {
            "instrument": "SingleCrystalDiffractometer",
            "slit": {"xwidth": 0.02, "yheight": None},
            "detector": {"radius": 0.8},
        },
        tmp_path / "over.yaml",
    )
    d = load(tmp_path / "over.yaml")
    assert (d.slit.xwidth, d.slit.yheight) == (0.02, 0.03)  # given wins; null: as built
    assert (d.detector.radius, d.detector.y_bins) == (0.8, 50)  # missing: as built
    assert d.positions()["sample_pos"] == [0, 0, 2.0]  # as built


def test_a_component_a_file_adds_is_built_where_the_file_puts_it(tmp_path):
    write_yaml(
        {
            "instrument": "SingleCrystalDiffractometer",
            "source": {"radius": 0.02},
            "extra_slit": {
                "component": "Slit",
                "xwidth": 0.01,
                "yheight": 0.01,
                "at": [0, 0, 0.5],
                "relative": "source",
            },
        },
        tmp_path / "extra.yaml",
    )
    d = load(tmp_path / "extra.yaml")
    assert list(d.components)[:4] == ["origin", "source", "extra_slit", "slit"]


def test_the_sample_of_params_yaml_is_built_before_the_detector():
    """params.yaml is laid over the diffractometer with no sample: its sample section is new,
    and is built where the file has it, not at the end."""
    assert list(load(PARAMS_FILE).components)[-2:] == ["sample", "detector"]


def test_the_diffractometer_roundtrips_through_a_yaml_file(diffractometer, tmp_path):
    inst = diffractometer({"detector": {"radius": 0.8}})
    write_yaml(to_dict(inst), tmp_path / "out.yaml")
    assert load(tmp_path / "out.yaml") == inst
    written = read_yaml(tmp_path / "out.yaml")
    assert {k: v["value"] for k, v in written["params"].items()} == {
        "ei": 14,
        "omega": 0.0,
        "chi": 0.0,
        "phi": 0.0,
    }
    assert "ei" not in written["source"] and "instr_params" not in written  # not saved twice


def test_numbers_in_the_yaml_are_numbers():
    """PyYAML reads 1.0e10 (no sign in the exponent) as a string: flux must be a number."""
    assert isinstance(load(PARAMS_FILE).source.flux, float)


def test_a_diffractometer_with_another_crystal_roundtrips_through_a_yaml_file(tmp_path):
    inst = SingleCrystalDiffractometer(la2ni7(mosaic=60, at=[0, 0, 0], relative="gonio_phi"))
    write_yaml(to_dict(inst), tmp_path / "la2ni7.yaml")  # the sample holds plain floats
    assert load(tmp_path / "la2ni7.yaml") == inst


def test_the_params_are_written_with_their_units_and_comments(tmp_path):
    write_yaml(to_dict(SingleCrystalDiffractometer()), tmp_path / "out.yaml")
    written = read_yaml(tmp_path / "out.yaml")["params"]
    assert written["ei"] == {"value": 14.0, "unit": "meV", "comment": "incident energy"}
    assert written["chi"]["unit"] == "deg"


def test_a_param_in_another_unit_is_refused(tmp_path):
    write_yaml(
        {
            "instrument": "SingleCrystalDiffractometer",
            "params": {"ei": {"value": 0.025, "unit": "eV"}},
        },
        tmp_path / "ev.yaml",
    )
    with pytest.raises(ValueError, match="params.ei"):
        load(tmp_path / "ev.yaml")


def test_a_param_may_be_a_bare_value_or_change_its_comment(tmp_path):
    write_yaml(
        {
            "instrument": "SingleCrystalDiffractometer",
            "params": {"ei": 25.0, "omega": {"value": 30.0, "comment": "run 3"}},
        },
        tmp_path / "p.yaml",
    )
    d = load(tmp_path / "p.yaml")
    assert (d.params.ei, d.params.omega) == (25.0, 30.0)


def test_a_misspelled_param_is_refused(tmp_path):
    write_yaml(
        {"instrument": "SingleCrystalDiffractometer", "params": {"omgea": 30.0}},
        tmp_path / "typo.yaml",
    )
    with pytest.raises(ValueError, match="omgea"):
        load(tmp_path / "typo.yaml")


def test_save_then_load_gives_the_diffractometer_back(tmp_path):
    inst = SingleCrystalDiffractometer(la2ni7(at=[0, 0, 0], relative="gonio_phi"))
    inst.update_params(omega=30)
    save(inst, tmp_path / "saved.yaml")
    assert load(tmp_path / "saved.yaml") == inst  # the sample comes from the file
    assert "at: [0, 0, 0]" in (tmp_path / "saved.yaml").read_text()  # laid out as params.yaml


@dataclass(frozen=True)
class _TinyParams:
    gap: float = param(0.01, "m", "slit opening")


_TINY_DEFAULT = _TinyParams()


class _Tiny(Instrument):
    """Another instrument: io/ works with it unchanged."""

    params_class = _TinyParams
    name = "tiny"

    def __init__(self, params: _TinyParams = _TINY_DEFAULT):
        super().__init__(params)
        self.origin = Progress_bar(at=[0, 0, 0])
        self.slit = Slit(xwidth="gap", yheight=0.02, at=[0, 0, 1.0], relative="origin")

    @property
    def instr_params(self):
        return [InstrParam("gap", self.params.gap, "m", "slit opening")]


def test_another_instrument_goes_through_the_same_io(tmp_path):
    tiny = _Tiny(_TinyParams(gap=0.05))
    save(tiny, tmp_path / "tiny.yaml")
    assert read_yaml(tmp_path / "tiny.yaml")["params"]["gap"]["unit"] == "m"
    assert load(tmp_path / "tiny.yaml") == tiny  # the file names its class
    write_yaml({"instrument": "_Tiny", "slit": {"yheight": 0.03}}, tmp_path / "over.yaml")
    assert load(tmp_path / "over.yaml").slit.yheight == 0.03  # a partial file names it too


def test_the_file_names_its_instrument(tmp_path):
    save(SingleCrystalDiffractometer(), tmp_path / "d.yaml")
    assert read_yaml(tmp_path / "d.yaml")["instrument"] == "SingleCrystalDiffractometer"
    assert read_yaml(PARAMS_FILE)["instrument"] == "SingleCrystalDiffractometer"
    write_yaml({"instrument": "Nothing"}, tmp_path / "x.yaml")
    with pytest.raises(ValueError, match="'Nothing' is no instrument"):
        load(tmp_path / "x.yaml")
    write_yaml({"slit": {"xwidth": 0.02}}, tmp_path / "unnamed.yaml")
    with pytest.raises(ValueError, match="None is no instrument"):
        load(tmp_path / "unnamed.yaml")  # no default: a file names its instrument
