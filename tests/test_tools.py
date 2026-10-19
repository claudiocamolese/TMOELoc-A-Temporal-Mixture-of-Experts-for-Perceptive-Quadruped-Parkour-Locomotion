"""Regression tests for offline metrics, archive discovery, and safe launcher helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
from _support import ROOT, load_tool

metrics = load_tool("analysis/episode_metrics.py")
episode_table, summarise = metrics.episode_table, metrics.summarise
newest_trace = load_tool("analysis/trace_utils.py").newest_trace
checkpoints = load_tool("export_slim_checkpoint.py")
select_checkpoints, slim = checkpoints.select_checkpoints, checkpoints.slim
load = load_tool("rl_verdict.py").load


class CheckpointTests(unittest.TestCase):
    """Selection and export must preserve final iterations and original training files."""

    def test_empty_run_with_interval_returns_no_checkpoints(self):
        """An empty directory must not raise an accidental IndexError."""
        with tempfile.TemporaryDirectory() as run:
            self.assertEqual(select_checkpoints(run, every=5000), [])

    def test_selection_uses_numeric_order_and_keeps_final(self):
        """Subsampling retains the final model even between interval boundaries."""
        with tempfile.TemporaryDirectory() as run:
            for name in ("model_9.pt", "model_100.pt", "model_23.pt", "model_latest.pt"):
                Path(run, name).touch()
            names = [Path(path).name for path in select_checkpoints(run, every=10)]
            self.assertEqual(names, ["model_100.pt"])
            names = [Path(path).name for path in select_checkpoints(run, every=9)]
            self.assertEqual(names, ["model_9.pt", "model_100.pt"])

    def test_invalid_interval_and_inplace_export_are_rejected(self):
        """Invalid requests should fail before reading or overwriting model data."""
        with self.assertRaises(ValueError):
            select_checkpoints("unused", every=0)
        with self.assertRaises(ValueError):
            slim("model.pt", "./model.pt")


class AnalysisTests(unittest.TestCase):
    """Protect exposure accounting and discovery alongside incomplete recorder output."""

    def test_censored_environments_contribute_time_and_progress(self):
        """A nonterminating trajectory must remain in the denominator and progress sum."""
        done = np.zeros((4, 2), dtype=bool)
        done[1, 0] = True
        goals = np.array([[0, 0], [7, 1], [0, 2], [1, 3]])
        report = summarise(episode_table(done, goals))
        self.assertEqual(report["segments"], 3)
        self.assertEqual(report["censored"], 2)
        self.assertAlmostEqual(report["exposure_min"], 8 * 0.02 / 60)
        self.assertAlmostEqual(report["goals_per_min"], 11 / (8 * 0.02) * 60)
        self.assertEqual(report["fall_hazard_per_1k_steps"], 0)

    def test_corrupt_and_unsuitable_archives_do_not_hide_valid_traces(self):
        """Discovery should close/skip partial archives and select a usable trace."""
        with tempfile.TemporaryDirectory() as directory:
            valid = Path(directory, "trace_valid.npz")
            np.savez(valid, attention=np.ones((1, 2)))
            np.savez(Path(directory, "trace_other.npz"), context=np.zeros((1, 2)))
            Path(directory, "trace_partial.npz").write_bytes(b"incomplete archive")
            with warnings.catch_warnings(record=True) as captured:
                result = newest_trace(directory, lambda archive: "attention" in archive.files)
            self.assertEqual(Path(result), valid)
            self.assertEqual(len(captured), 1)

    def test_log_parser_handles_missing_empty_and_carriage_return_records(self):
        """Read only complete numeric records and accept terminal carriage returns."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "ppo.log")
            self.assertIsNone(load(path))
            path.touch()
            self.assertIsNone(load(path))
            record = "it 2000 | reward -3.2 | ep_len 40.0 | kl 0.01 | lr 3.00e-04 | lvl 2.50"
            path.write_text(record + "\r" + record + "\n")
            self.assertEqual(load(path).shape, (2, 6))


class LauncherTests(unittest.TestCase):
    """Exercise pure shell helpers without starting or terminating any experiment."""

    def test_iteration_counter_returns_one_integer(self):
        """Missing files and zero grep matches must produce exactly one zero."""
        helper = ROOT / "scripts/lib/experiment_helpers.sh"
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory, "train.log")
            for contents, expected in (
                (None, "0\n"),
                ("no updates yet\n", "0\n"),
                ("Learning iteration 0\nLearning iteration 1\n", "2\n"),
            ):
                if contents is not None:
                    log.write_text(contents)
                result = subprocess.run(
                    [
                        "bash",
                        "-c",
                        'source "$1"; count_training_iterations "$2"',
                        "test",
                        str(helper),
                        str(log),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.assertEqual(result.stdout, expected)

    def test_checkpoint_selection_handles_spaces(self):
        """Shell selection must preserve filenames and choose the requested ordering."""
        helper = ROOT / "scripts/lib/experiment_helpers.sh"
        with tempfile.TemporaryDirectory(prefix="parkour run ") as directory:
            first = Path(directory, "model_1.pt")
            newer = Path(directory, "model_2.pt")
            first.touch()
            newer.touch()
            os.utime(first, (1, 1))
            os.utime(newer, (2, 2))
            env = dict(os.environ, PYTHON_BIN=sys.executable)
            for order, expected in (("first", first), ("newest", newer)):
                result = subprocess.run(
                    [
                        "bash",
                        "-c",
                        'source "$1"; select_checkpoint "$2" "$3"',
                        "test",
                        str(helper),
                        str(Path(directory, "model_*.pt")),
                        order,
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.assertEqual(result.stdout.strip(), str(expected))

    def test_environment_keeps_explicit_overrides(self):
        """Sourcing launcher defaults must preserve the caller's interpreter and teacher."""
        env = dict(
            os.environ,
            PYTHON_BIN="/configured/python",
            TEACHER_CKPT="/configured/model.pt",
            CUDA_VISIBLE_DEVICES="2",
            LD_LIBRARY_PATH="/configured/lib",
        )
        env.pop("CUDA_VISIBLE_DEVICES_OVERRIDE", None)
        result = subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; printf "%s\\n" "$PYTHON_BIN" "$TEACHER_CKPT" "$CUDA_VISIBLE_DEVICES" "$LD_LIBRARY_PATH"',
                "test",
                str(ROOT / "scripts/launch_env.sh"),
            ],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        lines = result.stdout.splitlines()
        self.assertEqual(lines[:3], ["/configured/python", "/configured/model.pt", "2"])
        self.assertIn("/configured/lib", lines[3].split(":"))

    def test_training_chain_propagates_failure(self):
        """A failing mock interpreter must stop the chain and preserve its exit status."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts/lib").mkdir(parents=True)
            for filename in ("run_experiments.sh", "launch_env.sh", "lib/experiment_helpers.sh"):
                shutil.copy(ROOT / "scripts" / filename, root / "scripts" / filename)
            interpreter = root / "mock-python"
            interpreter.write_text("#!/usr/bin/env bash\nexit 7\n")
            interpreter.chmod(0o755)
            result = subprocess.run(
                ["bash", str(root / "scripts/run_experiments.sh")],
                env=dict(os.environ, PYTHON_BIN=str(interpreter)),
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 7)
            self.assertIn("exit=7", result.stdout)
            self.assertFalse((root / "logs/run_console/baseline.log").exists())


if __name__ == "__main__":
    unittest.main()
