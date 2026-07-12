import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ingestion import rebuild_vector_store, replace_vector_store


class FailureSafeIngestionTests(unittest.TestCase):
    def test_successfully_replaces_existing_store(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            live = root / "chroma_db"
            live.mkdir()
            (live / "value.txt").write_text("old", encoding="utf-8")

            def builder(_chunks, _embedding, staging):
                staging.mkdir()
                (staging / "value.txt").write_text("new", encoding="utf-8")

            rebuild_vector_store([], None, live, builder=builder)

            self.assertEqual((live / "value.txt").read_text(), "new")
            self.assertEqual(list(root.glob(".chroma_db.*")), [])

    def test_build_failure_preserves_live_store_and_cleans_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            live = root / "chroma_db"
            live.mkdir()
            (live / "value.txt").write_text("old", encoding="utf-8")

            def failing_builder(_chunks, _embedding, staging):
                staging.mkdir()
                (staging / "partial.txt").write_text("partial", encoding="utf-8")
                raise RuntimeError("indexing failed")

            with self.assertRaisesRegex(RuntimeError, "indexing failed"):
                rebuild_vector_store([], None, live, builder=failing_builder)

            self.assertEqual((live / "value.txt").read_text(), "old")
            self.assertEqual(list(root.glob(".chroma_db.*")), [])

    def test_swap_failure_restores_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            live = root / "chroma_db"
            staging = root / ".chroma_db.staging-test"
            live.mkdir()
            staging.mkdir()
            (live / "value.txt").write_text("old", encoding="utf-8")
            (staging / "value.txt").write_text("new", encoding="utf-8")

            real_rename = Path.rename
            call_count = 0

            def fail_second_rename(source, destination):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise OSError("swap failed")
                return real_rename(source, destination)

            with patch("ingestion._rename_path", side_effect=fail_second_rename):
                with self.assertRaisesRegex(OSError, "swap failed"):
                    replace_vector_store(staging, live)

            self.assertEqual((live / "value.txt").read_text(), "old")
            self.assertTrue(staging.exists())
            self.assertEqual(list(root.glob(".chroma_db.backup-*")), [])


if __name__ == "__main__":
    unittest.main()
