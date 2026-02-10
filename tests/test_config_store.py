from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import slurm_cli.config_store as config_store  # noqa: E402


class ConfigStoreTests(unittest.TestCase):
    def test_search_fields_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self._with_temp_config(tmpdir)
            cfg = config_store.Config(
                last_search_max_time_minutes=240,
                last_search_min_time_minutes=30,
                last_search_max_gpus=4,
                last_search_min_gpus=1,
            )
            cfg.save()
            loaded = config_store.Config.load()
            self.assertEqual(loaded.last_search_max_time_minutes, 240)
            self.assertEqual(loaded.last_search_min_time_minutes, 30)
            self.assertEqual(loaded.last_search_max_gpus, 4)
            self.assertEqual(loaded.last_search_min_gpus, 1)

    def test_search_fields_ignore_non_positive_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self._with_temp_config(tmpdir)
            payload = {
                "last_search_max_time_minutes": -1,
                "last_search_min_time_minutes": 0,
                "last_search_max_gpus": "bad",
                "last_search_min_gpus": -3,
            }
            config_store.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            config_store.CONFIG_PATH.write_text(json.dumps(payload), encoding="utf-8")
            loaded = config_store.Config.load()
            self.assertIsNone(loaded.last_search_max_time_minutes)
            self.assertIsNone(loaded.last_search_min_time_minutes)
            self.assertIsNone(loaded.last_search_max_gpus)
            self.assertIsNone(loaded.last_search_min_gpus)

    def _with_temp_config(self, tmpdir: str) -> None:
        path = Path(tmpdir)
        self.addCleanup(self._restore_paths)
        self._orig_dir = config_store.CONFIG_DIR
        self._orig_path = config_store.CONFIG_PATH
        config_store.CONFIG_DIR = path
        config_store.CONFIG_PATH = path / "config.json"

    def _restore_paths(self) -> None:
        config_store.CONFIG_DIR = self._orig_dir
        config_store.CONFIG_PATH = self._orig_path


if __name__ == "__main__":
    unittest.main()
