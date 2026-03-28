# @author: Shaykh Siddique
# @email: shaykhsiddiqee@gmail.com


from __future__ import annotations

import csv
import math
import multiprocessing as mp
import time
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import numpy as np
from tqdm import tqdm

from HullParameterization import Hull_Parameterization as HP
import ModifiedMichellCw as MWCw
import MaxBox as MB
import Gaussian_Curvature as GC


class ShipHullProcessor:
    """
    Parent class:
    - loads hull design vectors from CSV
    - performs all calculations
    - saves outputs
    - generates STL files
    """

    def __init__(
        self,
        ds_path: str | Path = "./",
        input_vector_csv: str = "Input_Vectors.csv",
        geometry_dir: str = "GeometricMeasures",
        stl_dir: str = "stl",
        chunksize: int = 4,
        num_processes: Optional[int] = None,
    ) -> None:
        self._ds_path = Path(ds_path)
        self._input_vector_csv = self._ds_path / input_vector_csv
        self._geometry_dir = self._ds_path / geometry_dir
        self._stl_dir = self._ds_path / stl_dir

        self._chunksize = chunksize
        self._num_processes = num_processes if num_processes is not None else max(mp.cpu_count() - 2, 1)

        self._vec: Optional[np.ndarray] = None
        self._x_labels: Optional[np.ndarray] = None

        # Geometric property sampling locations
        self._z_idx = np.array([12, 25, 33, 38, 50, 62, 67, 75, 88, 100])
        self._z_idx_str = np.array([str(np.around(i / 100.0, decimals=2)) for i in self._z_idx])

        # Wave resistance settings
        self._z_ref = np.array([0.25, 0.33, 0.5, 0.67])
        self._fn_values = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])
        self._gravity = 9.81
        self._water_density = 1000.0
        self._num_sample_angles = 21
        self._test_hull_length = 10.0

    # ============================================================
    # Getters / Setters
    # ============================================================
    def get_ds_path(self) -> Path:
        return self._ds_path

    def set_ds_path(self, ds_path: str | Path) -> None:
        self._ds_path = Path(ds_path)

    def get_input_vector_csv(self) -> Path:
        return self._input_vector_csv

    def set_input_vector_csv(self, filename: str) -> None:
        self._input_vector_csv = self._ds_path / filename

    def get_geometry_dir(self) -> Path:
        return self._geometry_dir

    def set_geometry_dir(self, dirname: str) -> None:
        self._geometry_dir = self._ds_path / dirname

    def get_stl_dir(self) -> Path:
        return self._stl_dir

    def set_stl_dir(self, dirname: str) -> None:
        self._stl_dir = self._ds_path / dirname

    def get_chunksize(self) -> int:
        return self._chunksize

    def set_chunksize(self, chunksize: int) -> None:
        self._chunksize = int(chunksize)

    def get_num_processes(self) -> int:
        return self._num_processes

    def set_num_processes(self, num_processes: int) -> None:
        self._num_processes = int(num_processes)

    def get_vectors(self) -> Optional[np.ndarray]:
        return self._vec

    def get_x_labels(self) -> Optional[np.ndarray]:
        return self._x_labels

    # ============================================================
    # Printing helpers
    # ============================================================
    @staticmethod
    def print_header(title: str) -> None:
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)

    @staticmethod
    def print_subheader(title: str) -> None:
        print("\n" + "-" * 80)
        print(title)
        print("-" * 80)

    # ============================================================
    # File helpers
    # ============================================================
    def ensure_directories(self) -> None:
        self._geometry_dir.mkdir(parents=True, exist_ok=True)
        self._stl_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def save_csv(file_path: Path, header: List[str], rows: Iterable[Iterable]) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if header:
                writer.writerow(header)
            writer.writerows(rows)
        print(f"[SAVED] {file_path}")

    @staticmethod
    def load_vectors_from_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing file: {csv_path}")

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            data = list(reader)

        if len(data) < 2:
            raise ValueError(f"CSV file is empty or missing data rows: {csv_path}")

        x_labels = np.array(data[0], dtype=str)
        rows = np.array(data[1:], dtype=np.float64)
        return rows, x_labels

    # ============================================================
    # Dataset load
    # ============================================================
    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        self.print_header("Loading dataset from CSV")
        self._vec, self._x_labels = self.load_vectors_from_csv(self._input_vector_csv)

        print(f"CSV file                  : {self._input_vector_csv}")
        print(f"Loaded design vectors     : {self._vec.shape}")
        print(f"Loaded x_labels           : {self._x_labels.shape}")
        print(f"Using CPU processes       : {self._num_processes}")

        return self._vec, self._x_labels

    # ============================================================
    # Multiprocessing helper
    # ============================================================
    def run_imap_multiprocessing(self, func, argument_list, chunksize: int = 1, show_prog: bool = True):
        ctx = mp.get_context("spawn")
        indexed_args = list(enumerate(argument_list))
        results = [None] * len(indexed_args)

        with ctx.Pool(processes=self._num_processes) as pool:
            iterator = pool.imap_unordered(func, indexed_args, chunksize=chunksize)

            if show_prog:
                with tqdm(total=len(indexed_args), desc=f"Running {func.__name__}") as pbar:
                    for idx, result in iterator:
                        results[idx] = result
                        pbar.update(1)
            else:
                for idx, result in iterator:
                    results[idx] = result

        return results

    def run_imap_multiprocessing_fast(self, func, argument_list, chunksize: int = 1, show_prog: bool = True):
        ctx = mp.get_context("spawn")
        results = []

        with ctx.Pool(processes=self._num_processes) as pool:
            iterator = pool.imap_unordered(func, argument_list, chunksize=chunksize)

            if show_prog:
                with tqdm(total=len(argument_list), desc=f"Running {func.__name__}") as pbar:
                    for result in iterator:
                        results.append(result)
                        pbar.update(1)
            else:
                results = list(iterator)

        return results

    # ============================================================
    # Static worker methods for multiprocessing
    # ============================================================
    @staticmethod
    def _calc_geometric_properties_core(
        x: np.ndarray,
        z_idx: np.ndarray,
    ) -> np.ndarray:
        hull = HP(x)

        Z = hull.Calc_VolumeProperties(NUM_WL=101, PointsPerWL=1000)
        L = x[0]

        z = np.divide(Z[z_idx], L)
        vol = np.divide(hull.Volumes[z_idx], L**3.0)
        wp = np.divide(hull.Areas_WP[z_idx], L**2.0)
        lcf = np.divide(hull.LCFs[z_idx], L)
        ixx = np.divide(hull.I_WP[z_idx][:, 0], L**4.0)
        iyy = np.divide(hull.I_WP[z_idx][:, 1], L**4.0)

        vc = np.array(hull.VolumeCentroids)
        lcb = np.divide(vc[z_idx][:, 0], L)
        vcb = np.divide(vc[z_idx][:, 1], L)

        if hasattr(hull, "Area_WS"):
            wsa = np.divide(hull.Area_WS[z_idx], L**2.0)
        else:
            print("[WARNING] hull.Area_WS not found. Filling wetted surface area with zeros.")
            wsa = np.zeros(len(z_idx))

        if hasattr(hull, "WL_Lengths"):
            wl = np.divide(hull.WL_Lengths[z_idx], L)
        else:
            print("[WARNING] hull.WL_Lengths not found. Filling waterline lengths with zeros.")
            wl = np.zeros(len(z_idx))

        return np.concatenate((z, vol, wp, lcb, vcb, lcf, ixx, iyy, wsa, wl), axis=0)

    @staticmethod
    def calc_geometric_properties_worker(indexed_arg):
        idx, x = indexed_arg
        z_idx = np.array([12, 25, 33, 38, 50, 62, 67, 75, 88, 100])
        out_len = 10 * len(z_idx)

        try:
            result = ShipHullProcessor._calc_geometric_properties_core(x, z_idx)
            return idx, result
        except Exception as e:
            print(f"[WARNING] calc_geometric_properties failed for hull index {idx}: {e}")
            return idx, np.full(out_len, np.nan, dtype=np.float64)

    @staticmethod
    def _calc_rw_core(
        x: np.ndarray,
        z_ref: np.ndarray,
        fn_values: np.ndarray,
        gravity: float,
        water_density: float,
        num_sample_angles: int,
    ) -> np.ndarray:
        rw = np.zeros((len(z_ref) * len(fn_values)))
        hull = HP(x)

        for i, z_ratio in enumerate(z_ref):
            X, Z, Y, WL = hull.gen_PC_for_Cw(hull.Dd * z_ratio, NUM_WL=51, PointsPerWL=301)

            for j, fn in enumerate(fn_values):
                U = fn * math.sqrt(gravity * WL)
                rw[i * len(fn_values) + j] = MWCw.ModMichell(
                    Y, U, X, Z, water_density, num_sample_angles
                )

        return rw

    @staticmethod
    def calc_rw_worker(indexed_arg):
        idx, x = indexed_arg
        z_ref = np.array([0.25, 0.33, 0.5, 0.67])
        fn_values = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])
        gravity = 9.81
        water_density = 1000.0
        num_sample_angles = 21
        out_len = len(z_ref) * len(fn_values)

        try:
            result = ShipHullProcessor._calc_rw_core(
                x, z_ref, fn_values, gravity, water_density, num_sample_angles
            )
            return idx, result
        except Exception as e:
            print(f"[WARNING] calc_rw failed for hull index {idx}: {e}")
            return idx, np.full(out_len, np.nan, dtype=np.float64)

    @staticmethod
    def calc_maxbox_worker(indexed_arg):
        idx, x = indexed_arg
        try:
            result = MB.Run_BoxOpt(x)
            return idx, result
        except Exception as e:
            print(f"[WARNING] MaxBox failed for hull index {idx}: {e}")
            return idx, np.array([np.nan, np.nan, np.nan, np.nan, np.nan, True], dtype=object)

    @staticmethod
    def calc_gaussian_curvature_worker(indexed_arg):
        idx, x = indexed_arg
        try:
            result = GC.GaussianCurvature(x)
            return idx, result
        except Exception as e:
            print(f"[WARNING] Gaussian curvature failed for hull index {idx}: {e}")
            return idx, np.nan

    # ============================================================
    # Processing steps
    # ============================================================
    def benchmark_one_geometry_run(self) -> None:
        if self._vec is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        self.print_header("Benchmark: one geometric calculation")
        print(f"Single design vector shape: {self._vec[0].shape}")

        start = time.time()
        result = self._calc_geometric_properties_core(self._vec[0], self._z_idx)
        elapsed = time.time() - start

        print(f"One geometry result shape : {result.shape}")
        print(f"Time for one task         : {elapsed:.2f} seconds")

    def compute_and_save_geometric_measures(self) -> np.ndarray:
        if self._vec is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        self.print_header("Computing geometric measures")
        print(f"Z index ratios: {self._z_idx_str}")

        start = time.time()
        output = self.run_imap_multiprocessing(
            self.calc_geometric_properties_worker,
            self._vec,
            chunksize=self._chunksize,
            show_prog=True,
        )
        output = np.array(output, dtype=np.float64)
        elapsed = time.time() - start

        print(f"Geometric measures output shape: {output.shape}")
        print(f"Elapsed time                  : {elapsed:.2f} seconds")

        bad_rows = np.where(np.isnan(output).any(axis=1))[0]
        if len(bad_rows) > 0:
            print(f"[WARNING] Geometric calculation failed for hull indices: {bad_rows.tolist()}")

        L = len(self._z_idx)

        self.save_csv(
            self._geometry_dir / "z.csv",
            [f"z/LOA @ T/Dd = {i}" for i in self._z_idx_str],
            output[:, 0:L],
        )
        self.save_csv(
            self._geometry_dir / "Volume.csv",
            [f"Volume/LOA^3 @ T/Dd = {i}" for i in self._z_idx_str],
            output[:, L:2 * L],
        )
        self.save_csv(
            self._geometry_dir / "Area_WP.csv",
            [f"Area_WP/LOA^2 @ T/Dd = {i}" for i in self._z_idx_str],
            output[:, 2 * L:3 * L],
        )
        self.save_csv(
            self._geometry_dir / "LCB.csv",
            [f"LCB/LOA @ T/Dd = {i}" for i in self._z_idx_str],
            output[:, 3 * L:4 * L],
        )
        self.save_csv(
            self._geometry_dir / "VCB.csv",
            [f"VCB/LOA @ T/Dd = {i}" for i in self._z_idx_str],
            output[:, 4 * L:5 * L],
        )
        self.save_csv(
            self._geometry_dir / "LCF.csv",
            [f"LCF/LOA @ T/Dd = {i}" for i in self._z_idx_str],
            output[:, 5 * L:6 * L],
        )
        self.save_csv(
            self._geometry_dir / "Ixx.csv",
            [f"Ixx/LOA^4 @ T/Dd = {i}" for i in self._z_idx_str],
            output[:, 6 * L:7 * L],
        )
        self.save_csv(
            self._geometry_dir / "Iyy.csv",
            [f"Iyy/LOA^4 @ T/Dd = {i}" for i in self._z_idx_str],
            output[:, 7 * L:8 * L],
        )
        self.save_csv(
            self._geometry_dir / "Area_WS.csv",
            [f"Area_WS/LOA^2 @ T/Dd = {i}" for i in self._z_idx_str],
            output[:, 8 * L:9 * L],
        )
        self.save_csv(
            self._geometry_dir / "WL_Length.csv",
            [f"WL/LOA @ T/Dd = {i}" for i in self._z_idx_str],
            output[:, 9 * L:10 * L],
        )

        return output

    def compute_and_save_rw(self) -> np.ndarray:
        if self._vec is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        self.print_header("Computing wave resistance (Rw)")
        des_vec = self._vec.astype(np.float64)
        print(f"Design vector size: {des_vec.shape}")

        start = time.time()
        output = self.run_imap_multiprocessing(
            self.calc_rw_worker,
            des_vec,
            chunksize=self._chunksize,
            show_prog=True,
        )
        output = np.array(output, dtype=np.float64)
        elapsed = time.time() - start

        print(f"Rw output shape: {output.shape}")
        print(f"Elapsed time   : {elapsed:.2f} seconds")

        bad_rows = np.where(np.isnan(output).any(axis=1))[0]
        if len(bad_rows) > 0:
            print(f"[WARNING] Wave resistance calculation failed for hull indices: {bad_rows.tolist()}")

        np.save(self._geometry_dir / "Rw_Output.npy", output)
        print(f"[SAVED] {self._geometry_dir / 'Rw_Output.npy'}")
        return output

    def compute_and_save_cw(self, rw_output: np.ndarray) -> np.ndarray:
        self.print_header("Converting Rw to Cw")

        labels = []
        for i in range(len(self._z_ref)):
            for j in range(len(self._fn_values)):
                labels.append(f"draft={self._z_ref[i]} Fn={self._fn_values[j]}")

        print(f"Total Cw labels: {len(labels)}")

        wl_path = self._geometry_dir / "WL_Length.csv"
        if not wl_path.exists():
            raise FileNotFoundError(f"Missing required file: {wl_path}")

        wl_vec = []
        with open(wl_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                wl_vec.append(row)

        wl = np.array(wl_vec[1:]).astype(np.float64)
        print(f"WL_Length shape: {wl.shape}")

        z_ref_to_col = {}
        for z_ratio in self._z_ref:
            idx_value = int(round(z_ratio * 100))
            if idx_value not in self._z_idx:
                raise ValueError(
                    f"Draft ratio {z_ratio} -> index {idx_value} not found in Z_IDX {self._z_idx.tolist()}"
                )
            z_ref_to_col[z_ratio] = int(np.where(self._z_idx == idx_value)[0][0])

        print(f"Draft to WL column mapping: {z_ref_to_col}")

        cw = np.zeros((len(rw_output), len(rw_output[0])))
        print(f"Cw output shape : {cw.shape}")

        for h in range(len(rw_output)):
            if h % 100 == 0:
                print(f"Processing hull {h}/{len(rw_output)}")

            if np.isnan(rw_output[h]).any():
                cw[h, :] = np.nan
                continue

            for i, z_ratio in enumerate(self._z_ref):
                wl_col = z_ref_to_col[z_ratio]
                wl_value = wl[h, wl_col] * self._test_hull_length

                for j in range(len(self._fn_values)):
                    U = self._fn_values[j] * math.sqrt(self._gravity * wl_value)
                    denom = 0.5 * self._water_density * (U**2.0) * self._test_hull_length**2.0
                    cw[h, i * len(self._fn_values) + j] = rw_output[h, i * len(self._fn_values) + j] / denom

        self.save_csv(self._geometry_dir / "Cw.csv", labels, cw)
        return cw

    def compute_and_save_maxbox(self) -> np.ndarray:
        if self._vec is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        self.print_header("Computing MaxBox")
        output = self.run_imap_multiprocessing(
            self.calc_maxbox_worker,
            self._vec,
            chunksize=self._chunksize,
            show_prog=True,
        )
        boxes = np.array(output, dtype=object)

        violation_count = sum(1 for a in boxes[:, 5] if a)
        print(f"PC constraint violation count: {violation_count}")

        self.save_csv(
            self._geometry_dir / "MaxBox.csv",
            [
                "x/LOA of fwd end of Box",
                "length/LOA of Box",
                "height/LOA of Box",
                "width/LOA of Box",
                "Volume/LOA^3 of Box",
            ],
            boxes[:, 0:-1],
        )
        return boxes

    def compute_and_save_gaussian_curvature(self) -> np.ndarray:
        if self._vec is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        self.print_header("Computing Gaussian curvature")
        output = self.run_imap_multiprocessing(
            self.calc_gaussian_curvature_worker,
            self._vec,
            chunksize=self._chunksize,
            show_prog=True,
        )
        gk = np.array(output, dtype=np.float64)

        self.save_csv(
            self._geometry_dir / "GaussianCurvature.csv",
            ["Gaussian Curvature * LOA^2"],
            [[value] for value in gk],
        )
        return gk

    def generate_stl_files(self, limit: int = 1000) -> None:
        if self._vec is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        self.print_header("Generating STL files")
        count = min(limit, len(self._vec))
        print(f"Generating STL files for {count} hulls")

        for i in tqdm(range(count), desc="Generating STL"):
            try:
                hull = HP(self._vec[i])
                strpath = self._stl_dir / f"Hull_mesh_{i}"
                hull.gen_stl(NUM_WL=100, PointsPerWL=800, namepath=str(strpath))
            except Exception as e:
                print(f"[WARNING] STL generation failed for hull index {i}: {e}")

        print(f"STL generation complete. Files saved under: {self._stl_dir}")

    def run_all(self, stl_limit: int = 1000) -> None:
        self.ensure_directories()
        self.load_dataset()
        self.benchmark_one_geometry_run()

        _ = self.compute_and_save_geometric_measures()
        rw_output = self.compute_and_save_rw()

        try:
            _ = self.compute_and_save_cw(rw_output)
        except Exception as e:
            print(f"[WARNING] Cw calculation skipped: {e}")

        try:
            _ = self.compute_and_save_maxbox()
        except Exception as e:
            print(f"[WARNING] MaxBox calculation skipped: {e}")

        try:
            _ = self.compute_and_save_gaussian_curvature()
        except Exception as e:
            print(f"[WARNING] Gaussian curvature calculation skipped: {e}")

        try:
            self.generate_stl_files(limit=stl_limit)
        except Exception as e:
            print(f"[WARNING] STL generation skipped: {e}")

        self.print_header("All requested steps finished")


class ShipHullProcessorPreloaded(ShipHullProcessor):
    """
    Child class:
    - inherits all calculations from parent
    - additionally loads from NPY
    - samples N hulls
    - creates CSV file
    """

    def __init__(
        self,
        ds_path: str | Path = "./",
        input_vector_file: str = "InputVectors_30k.npy",
        x_labels_file: str = "X_LABELS.npy",
        output_csv: str = "Input_Vectors.csv",
        num_samples: Optional[int] = 10,
        geometry_dir: str = "GeometricMeasures",
        stl_dir: str = "stl",
        chunksize: int = 4,
        num_processes: Optional[int] = None,
    ) -> None:
        super().__init__(
            ds_path=ds_path,
            input_vector_csv=output_csv,
            geometry_dir=geometry_dir,
            stl_dir=stl_dir,
            chunksize=chunksize,
            num_processes=num_processes,
        )

        self._input_vector_file = self._ds_path / input_vector_file
        self._x_labels_file = self._ds_path / x_labels_file
        self._num_samples = num_samples

    # ============================================================
    # Getters / Setters
    # ============================================================
    def get_input_vector_file(self) -> Path:
        return self._input_vector_file

    def set_input_vector_file(self, filename: str) -> None:
        self._input_vector_file = self._ds_path / filename

    def get_x_labels_file(self) -> Path:
        return self._x_labels_file

    def set_x_labels_file(self, filename: str) -> None:
        self._x_labels_file = self._ds_path / filename

    def get_num_samples(self) -> Optional[int]:
        return self._num_samples

    def set_num_samples(self, num_samples: Optional[int]) -> None:
        self._num_samples = num_samples

    # ============================================================
    # NPY loading and CSV creation
    # ============================================================
    @staticmethod
    def load_vectors_from_npy(
        npy_path: Path,
        labels_path: Path,
        num_samples: Optional[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not npy_path.exists():
            raise FileNotFoundError(f"Missing file: {npy_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing file: {labels_path}")

        vec = np.load(npy_path)
        x_labels = np.load(labels_path)

        if num_samples is not None:
            vec = vec[:num_samples]

        x_labels = np.array(x_labels).astype(str)
        return vec.astype(np.float64), x_labels

    def create_csv_from_npy(self) -> Tuple[np.ndarray, np.ndarray]:
        self.print_header("Creating CSV from NPY")

        self._vec, self._x_labels = self.load_vectors_from_npy(
            self._input_vector_file,
            self._x_labels_file,
            self._num_samples,
        )

        print(f"Input NPY file             : {self._input_vector_file}")
        print(f"Input labels file          : {self._x_labels_file}")
        print(f"Loaded design vectors      : {self._vec.shape}")
        print(f"Loaded x_labels            : {self._x_labels.shape}")
        print(f"Number of samples used     : {self._num_samples}")

        self.save_csv(self._input_vector_csv, list(self._x_labels), self._vec)
        print(f"CSV created                : {self._input_vector_csv}")

        return self._vec, self._x_labels

    def run_all(self, stl_limit: int = 1000) -> None:
        self.ensure_directories()
        self.create_csv_from_npy()
        self.benchmark_one_geometry_run()

        _ = self.compute_and_save_geometric_measures()
        rw_output = self.compute_and_save_rw()

        try:
            _ = self.compute_and_save_cw(rw_output)
        except Exception as e:
            print(f"[WARNING] Cw calculation skipped: {e}")

        try:
            _ = self.compute_and_save_maxbox()
        except Exception as e:
            print(f"[WARNING] MaxBox calculation skipped: {e}")

        try:
            _ = self.compute_and_save_gaussian_curvature()
        except Exception as e:
            print(f"[WARNING] Gaussian curvature calculation skipped: {e}")

        try:
            self.generate_stl_files(limit=stl_limit)
        except Exception as e:
            print(f"[WARNING] STL generation skipped: {e}")

        self.print_header("All requested steps finished")


def main() -> None:
    """
    Example 1:
    Use existing CSV directly
    """
    processor = ShipHullProcessor(
        ds_path="./",
        input_vector_csv="Input_Vectors_SampleHulls.csv",
        geometry_dir="GeometricMeasures",
        stl_dir="stl",
        chunksize=4,
    )
    processor.run_all(stl_limit=1000)

    """
    Example 2:
    Create CSV from NPY first, then run everything
    """

    # preloaded_processor = ShipHullProcessorPreloaded(
    #     ds_path="./",
    #     input_vector_file="InputVectors_30k.npy",
    #     x_labels_file="X_LABELS.npy",
    #     output_csv="Input_Vectors.csv",
    #     num_samples=5,
    #     geometry_dir="GeometricMeasures",
    #     stl_dir="stl",
    #     chunksize=4,
    # )
    # preloaded_processor.run_all(stl_limit=1000)
    


if __name__ == "__main__":
    mp.freeze_support()
    main()
