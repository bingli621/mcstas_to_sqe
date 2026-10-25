"""Simulate an instrument into McStas NeXus (HDF5) output, read with :func:`.read_pixels`.

A :class:`Simulation` holds the instrument to simulate and the folder its runs write to. It
offers:

- ``run``: one run, source -> detector;
- ``scan``: one run per value of an instrument parameter, source -> detector, in parallel;
- ``run_to_sample``: source -> ``sample_pos`` once, the beam dumped into an MCPL beam file;
- ``run_from_sample``: one run from that beam file, ``sample_pos`` -> detector;
- ``scan_from_sample``: one run per value from that beam file, in parallel.

Everything runs through McStasScript's public API (``settings``, ``set_parameters``,
``run_to``, ``run_from``, ``backengine``). A scan runs ``backengine`` once per value and merges
the outputs into one file. The instrument is first compiled once with mcrun itself, since
McStasScript has no compile-only call.
"""

import copy
import os
import shlex
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from mcstas_to_sqe.mcstas.builder import build_mcstas, default_input_dir
from mcstas_to_sqe.mcstas.nexus_io import merge_entries
from mcstas_to_sqe.mcstas.run_folders import (
    clean_runs,
    default_output_dir,
    latest_beam,
    latest_run,
    list_runs,
    run_folder,
)

if TYPE_CHECKING:
    from mcstasscript.interface.instr import McStas_instr

    from mcstas_to_sqe.core import Instrument


def _compile(inst: "McStas_instr") -> None:
    """Write the instrument into its input folder and compile it there, without simulating.

    mcrun compiles (``-c``), then the instrument only prints its info (``--info``), so not even a
    beam file is read. The compile flags are those of ``inst``'s settings (NeXus, OpenACC,
    custom flags), so its runs reuse the executable.

    Raises
    ------
    RuntimeError
        With mcrun's output, if compiling fails.
    """
    inst.write_full_instrument()
    options = inst._run_settings  # McStasScript keeps them private
    mcrun = Path(options.get("executable_path", "")) / options.get("executable", "mcrun")
    command = [str(mcrun), "-c", "--info"]
    if options.get("NeXus"):
        command.append("--format=NeXus")
    if options.get("openacc"):
        command.append("--openacc")
    command += [*shlex.split(options.get("custom_flags", "")), f"{inst.name}.instr"]
    process = subprocess.run(
        command, cwd=inst.input_path, capture_output=True, text=True, check=False
    )
    if process.returncode != 0:
        raise RuntimeError(f"compiling {inst.name} failed:\n{process.stdout}{process.stderr}")


def _run_points_in_parallel(
    inst: "McStas_instr",
    output_path: Path,
    points: Sequence[Mapping[str, float]],
    seed: int | None = None,
    n_jobs: int | None = None,
) -> Path:
    """Run the instrument once per point and merge the outputs into one scan file.

    The instrument is written and compiled once, first, without simulating. Then every point
    runs a copy of ``inst`` in parallel, with its own values, output folder and random seed.

    Parameters
    ----------
    inst:
        The McStasScript instrument, with its settings.
    output_path:
        Folder to create for the scan.
    points:
        Instrument parameter values (by McStas name) of each point.
    seed:
        Seed of the first point; point ``n`` gets ``seed + n + 1``, so the runs do not repeat
        each other's random numbers. If ``None``, a random seed.
    n_jobs:
        Number of points run at a time. If ``None``, the number of cores.

    Returns
    -------
    :
        ``output_path/mccode.h5``, with one entry per point in the order of ``points``.

    Raises
    ------
    ValueError
        If there are no points, before anything runs.
    RuntimeError
        Naming the point that failed, once the points already running are done. Points not
        started yet are not run, and every point's folder is kept in ``output_path``.
    """
    if not points:
        raise ValueError("a scan needs at least one point")
    _compile(inst)  # once, before any point
    output_path.mkdir()  # holds a folder per point
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0] % 10**9)
    folders = [output_path / f"point_{n:04d}" for n in range(len(points))]

    runs = []
    for n, (values, folder) in enumerate(zip(points, folders)):
        run = copy.deepcopy(inst)  # a run's settings and values stay its own
        run.settings(
            output_path=str(folder),
            seed=seed + n + 1,  # mcrun does not accept seed 0
            force_compile=False,  # compiled once, above
        )
        run.set_parameters(**values)
        runs.append(run)

    def run_point(n: int) -> None:
        try:
            runs[n].backengine()  # a separate mcrun process
        except Exception as error:
            raise RuntimeError(f"point {n} {points[n]} failed, see {folders[n]}") from error

    with ThreadPoolExecutor(max_workers=n_jobs or os.cpu_count()) as pool:
        futures = [pool.submit(run_point, n) for n in range(len(runs))]
        for future in as_completed(futures):
            if future.exception() is not None:
                pool.shutdown(wait=False, cancel_futures=True)  # the exit waits for the running
                future.result()

    merged = merge_entries([folder / "mccode.h5" for folder in folders], output_path / "mccode.h5")
    for folder in folders:
        shutil.rmtree(folder)
    return merged


class Simulation:
    """Run McStas simulations of an instrument.

    Every run builds the instrument anew, into ``input_dir``, so a change to the instrument
    applies to the next run.

    Parameters
    ----------
    instrument:
        The instrument to simulate, e.g. a ``SingleCrystalDiffractometer``.
    output_dir:
        Folder for the run folders. If ``None``, ``work_dir/mcstas_output``.
    input_dir:
        Folder for the McStas files. If ``None``, ``work_dir/mcstas_input/<instrument name>``.
    work_dir:
        Work folder for the defaults above, fixed when the simulation is created. If ``None``,
        :func:`.default_work_dir`: ``MCSTAS_TO_SQE_WORKDIR`` or the current folder.

    Attributes
    ----------
    beam_file:
        Beam file the runs from the sample start from: that of the last :meth:`run_to_sample`,
        else the newest ``beam.mcpl.gz`` in ``output_dir``, so a beam dumped in an earlier
        session is found. ``None`` if there is none.

    Notes
    -----
    Each run writes into a new folder of ``output_dir``, named by its start time and mode:
    ``<time>_run``, ``<time>_scan_<variable>``, ``<time>_run_to_sample``,
    ``<time>_run_from_sample`` and ``<time>_scan_<variable>_from_sample``, with ``<time>`` e.g.
    ``20261022-143005``. A ``folder_name`` names the folder instead, and it is then replaced on
    every run. :meth:`list_runs` and :meth:`latest_run` find finished runs, and :meth:`clean_runs`
    removes older ones, never a ``run_to_sample`` one.

    The ``instr_params`` of a run are its instrument parameters by McStas name: ``omega``,
    ``chi``, ``phi`` (deg) and ``wavelength`` (Å). Those not given keep the instrument's values;
    the wavelength defaults to that of the instrument's ``ei``.
    """

    def __init__(
        self,
        instrument: "Instrument",
        output_dir: str | Path | None = None,
        input_dir: str | Path | None = None,
        work_dir: str | Path | None = None,
    ) -> None:
        self.instrument = instrument
        self.output_dir = Path(output_dir or default_output_dir(work_dir))
        self.input_dir = Path(input_dir or default_input_dir(instrument, work_dir))
        self.beam_file = latest_beam(self.output_dir)

    def list_runs(self) -> list[str]:
        """List the finished runs in ``output_dir``, newest first; see :func:`.list_runs`."""
        return list_runs(self.output_dir)

    def latest_run(self, pattern: str = "*") -> Path:
        """Return the newest finished run matching ``pattern``; see :func:`.latest_run`."""
        return latest_run(self.output_dir, pattern)

    def clean_runs(self, keep: int = 5) -> list[Path]:
        """Remove all but the ``keep`` newest runs; see :func:`.clean_runs`."""
        return clean_runs(self.output_dir, keep)

    def _prepare(
        self, folder_name: str | None, mode: str, **settings: object
    ) -> tuple[Path, "McStas_instr"]:
        """Pick the run folder and build the instrument with the settings every run shares.

        Parameters
        ----------
        folder_name:
            Run folder name. If ``None``, ``<time>_<mode>``.
        mode:
            What the run does, in the default folder name, e.g. ``"scan_omega"``.
        **settings:
            The run's own McStasScript settings, e.g. ``ncount`` and ``mpi``.

        Returns
        -------
        :
            The run folder, and the McStasScript instrument writing into it. A scan's points
            write into folders of their own instead (:func:`_run_points_in_parallel`).
        """
        output_path = run_folder(self.output_dir, folder_name, mode)
        inst = build_mcstas(self.instrument, input_path=self.input_dir)
        inst.settings(
            NeXus=True,
            increment_folder_name=False,
            suppress_output=True,
            output_path=str(output_path),
            force_compile=True,
            **settings,  # pyright: ignore[reportArgumentType]
        )
        return output_path, inst

    def _sample_pos_name(self) -> str:
        """Return the name of the instrument's ``sample_pos``, where a split run cuts the beam.

        Raises
        ------
        ValueError
            If the instrument has no ``sample_pos`` component.
        """
        sample_pos = getattr(self.instrument, "sample_pos", None)
        if sample_pos is None or sample_pos.name not in self.instrument.components:
            raise ValueError(
                f"{self.instrument.name} has no sample_pos component to split a run at"
            )
        return sample_pos.name

    def find_beam_file(self, beam_file: str | Path | None = None) -> Path:
        """Return the absolute path of the beam file a run from the sample starts from.

        The path is absolute because mcrun runs in ``input_dir``.

        Parameters
        ----------
        beam_file:
            The beam file. If ``None``, :attr:`beam_file`.

        Returns
        -------
        :
            The absolute path.

        Raises
        ------
        FileNotFoundError
            If both ``beam_file`` and :attr:`beam_file` are ``None``.
        """
        beam_file = beam_file or self.beam_file
        if beam_file is None:
            raise FileNotFoundError(f"no beam file in {self.output_dir}: run run_to_sample first")
        return Path(beam_file).resolve()

    def get_scan_points(
        self, variable: str, values: ArrayLike, instr_params: Mapping[str, float]
    ) -> list[dict[str, float]]:
        """Return the points of a scan.

        Parameters
        ----------
        variable:
            Instrument parameter to scan.
        values:
            Values of ``variable``.
        instr_params:
            The other instrument parameters, the same for every point.

        Returns
        -------
        :
            ``{**instr_params, variable: value}`` for each value.

        Raises
        ------
        KeyError
            If a name is not an instrument parameter, before any run.
        """
        names = {p.name for p in self.instrument.instr_params}
        if unknown := sorted({variable, *instr_params} - names):
            raise KeyError(f"no instrument parameter {unknown}, only {sorted(names)}")
        return [
            {**instr_params, variable: value} for value in np.asarray(values, dtype=float).tolist()
        ]

    def run(
        self,
        folder_name: str | None = None,
        ncount: float = 1e7,
        seed: int | None = None,
        mpi: int | str = "auto",
        custom_flags: str | None = None,
        **instr_params: float,
    ) -> Path:
        """Simulate source -> detector once.

        Parameters
        ----------
        folder_name:
            Run folder name. If ``None``, ``<time>_run``.
        ncount:
            Number of neutrons.
        seed:
            Random seed. If ``None``, McStas picks one.
        mpi:
            Number of MPI processes, or ``"auto"``.
        custom_flags:
            Extra flags for mcrun.
        **instr_params:
            Instrument parameters by McStas name.

        Returns
        -------
        :
            ``output_dir/<folder>/mccode.h5``.
        """
        output_path, inst = self._prepare(
            folder_name, "run", ncount=ncount, mpi=mpi, seed=seed, custom_flags=custom_flags
        )
        inst.set_parameters(**instr_params)  # the rest keep the instrument's values
        inst.backengine()  # compiles in its input_dir
        return output_path / "mccode.h5"

    def scan(
        self,
        variable: str,
        values: ArrayLike,
        folder_name: str | None = None,
        ncount: float = 1e7,
        n_jobs: int | None = None,
        seed: int | None = None,
        custom_flags: str | None = None,
        **instr_params: float,
    ) -> Path:
        """Simulate source -> detector for every value of an instrument parameter.

        The values run in parallel as single-process mcrun jobs.

        Parameters
        ----------
        variable:
            Instrument parameter to scan, e.g. ``"omega"``.
        values:
            Values of ``variable``, e.g. ``np.linspace(-48, 83, 11)``.
        folder_name:
            Run folder name. If ``None``, ``<time>_scan_<variable>``.
        ncount:
            Number of neutrons per value.
        n_jobs:
            Number of jobs at a time. If ``None``, the number of cores.
        seed:
            Random seed; value ``n`` gets ``seed + n + 1``, so the jobs do not repeat each
            other's random numbers. If ``None``, a random seed.
        custom_flags:
            Extra flags for mcrun.
        **instr_params:
            The other instrument parameters by McStas name.

        Returns
        -------
        :
            ``output_dir/<folder>/mccode.h5``, with one entry per value.

        Raises
        ------
        KeyError
            If a name is not an instrument parameter, before anything runs.
        """
        points = self.get_scan_points(variable, values, instr_params)  # checked before anything
        output_path, inst = self._prepare(
            folder_name,
            f"scan_{variable}",
            ncount=ncount,
            mpi=None,  # no MPI: the points run in parallel instead
            custom_flags=custom_flags,
        )
        return _run_points_in_parallel(inst, output_path, points, seed=seed, n_jobs=n_jobs)

    def run_to_sample(
        self,
        folder_name: str | None = None,
        ncount: float = 1e8,
        mpi: int | str = "auto",
        **instr_params: float,
    ) -> Path:
        """Simulate source -> ``sample_pos`` and dump the beam to an MCPL file.

        This is part 1 of a split run: run it once, with a large ``ncount``. The run folder also
        holds a ``mccode.h5`` with the incident monitor.

        The beam file only fits instruments with the same source, beam geometry and wavelength.
        Make a new one after changing the instrument's ``ei``, source, slit or monitor.

        Parameters
        ----------
        folder_name:
            Run folder name. If ``None``, ``<time>_run_to_sample``.
        ncount:
            Number of neutrons.
        mpi:
            Number of MPI processes, or ``"auto"``.
        **instr_params:
            Instrument parameters by McStas name.

        Returns
        -------
        :
            ``output_dir/<folder>/beam.mcpl.gz``, which also becomes :attr:`beam_file`.

        Raises
        ------
        ValueError
            If the instrument has no ``sample_pos``, before anything runs.
        """
        sample_pos = self._sample_pos_name()  # checked before anything
        output_path, inst = self._prepare(folder_name, "run_to_sample", ncount=ncount, mpi=mpi)
        inst.set_parameters(**instr_params)  # the rest keep the instrument's values
        inst.run_to(
            sample_pos,
            run_name="beam",
            comment=f"{ncount:g} neutrons",
            filename='"beam.mcpl"',
            # the ray count is stored in the file and the weights scaled by it (McStas >= 3.10
            # default; the raw weights of mode 2 are deprecated)
            weight_mode=1,
        )
        inst.backengine()
        self.beam_file = output_path / "beam.mcpl.gz"
        return self.beam_file

    def run_from_sample(
        self,
        folder_name: str | None = None,
        beam_file: str | Path | None = None,
        seed: int | None = None,
        mpi: int | str = "auto",
        **instr_params: float,
    ) -> Path:
        """Simulate ``sample_pos`` -> sample -> detector once, from a beam file.

        This is part 2 of a split run, and may run any time later, even in another session. The
        number of neutrons is the number in the beam file.

        Parameters
        ----------
        folder_name:
            Run folder name. If ``None``, ``<time>_run_from_sample``.
        beam_file:
            Beam file to start from. If ``None``, :attr:`beam_file`.
        seed:
            Random seed. If ``None``, McStas picks one.
        mpi:
            Number of MPI processes, or ``"auto"``. Each reads its own part of the beam file.
        **instr_params:
            Instrument parameters by McStas name. The wavelength must be that of the beam file.

        Returns
        -------
        :
            ``output_dir/<folder>/mccode.h5``.

        Raises
        ------
        ValueError
            If the instrument has no ``sample_pos``, before anything runs.
        FileNotFoundError
            If there is no beam file, before anything runs.
        """
        sample_pos = self._sample_pos_name()  # checked before anything
        beam_file = self.find_beam_file(beam_file)
        output_path, inst = self._prepare(folder_name, "run_from_sample", mpi=mpi, seed=seed)
        inst.run_from(sample_pos, filename=f'"{beam_file}"')
        inst.set_parameters(**instr_params)  # the rest keep the instrument's values
        inst.backengine()
        return output_path / "mccode.h5"

    def scan_from_sample(
        self,
        variable: str,
        values: ArrayLike,
        folder_name: str | None = None,
        beam_file: str | Path | None = None,
        n_jobs: int | None = None,
        seed: int | None = None,
        **instr_params: float,
    ) -> Path:
        """Simulate ``sample_pos`` -> detector from a beam file, for every value of a parameter.

        This is part 2 of a split run, for a scan. The values run in parallel as single-process
        mcrun jobs, all starting from the same beam file.

        Parameters
        ----------
        variable:
            Instrument parameter to scan, e.g. ``"omega"``.
        values:
            Values of ``variable``, e.g. ``range(-48, 83)``.
        folder_name:
            Run folder name. If ``None``, ``<time>_scan_<variable>_from_sample``.
        beam_file:
            Beam file to start from. If ``None``, :attr:`beam_file`.
        n_jobs:
            Number of jobs at a time. If ``None``, the number of cores.
        seed:
            Random seed; value ``n`` gets ``seed + n + 1``, so the jobs do not repeat each
            other's random numbers. If ``None``, a random seed.
        **instr_params:
            The other instrument parameters by McStas name.

        Returns
        -------
        :
            ``output_dir/<folder>/mccode.h5``, with one entry per value.

        Raises
        ------
        KeyError
            If a name is not an instrument parameter, before anything runs.
        ValueError
            If the instrument has no ``sample_pos``, before anything runs.
        FileNotFoundError
            If there is no beam file, before anything runs.
        """
        points = self.get_scan_points(variable, values, instr_params)  # checked before anything
        sample_pos = self._sample_pos_name()
        beam_file = self.find_beam_file(beam_file)
        output_path, inst = self._prepare(
            folder_name,
            f"scan_{variable}_from_sample",
            mpi=None,  # no MPI: the points run in parallel instead
        )
        inst.run_from(sample_pos, filename=f'"{beam_file}"')
        return _run_points_in_parallel(inst, output_path, points, seed=seed, n_jobs=n_jobs)
