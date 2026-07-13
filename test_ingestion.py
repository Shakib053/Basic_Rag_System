import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from ingestion import (
    build_chunker,
    get_chunking_strategy,
    rebuild_vector_store,
    replace_vector_store,
)


class ChunkerConfigurationTests(unittest.TestCase):
    @patch.dict("ingestion.os.environ", {}, clear=True)
    def test_semantic_chunking_is_default(self):
        self.assertEqual(get_chunking_strategy(), "semantic")

    def test_recursive_chunker_keeps_existing_settings(self):
        splitter = build_chunker("recursive", Mock())

        self.assertEqual(splitter._chunk_size, 700)
        self.assertEqual(splitter._chunk_overlap, 100)

    def test_invalid_chunking_strategy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported CHUNKING_STRATEGY"):
            get_chunking_strategy("fixed")

    @patch("ingestion._load_semantic_chunker_class")
    def test_semantic_chunker_uses_expected_configuration(self, loader):
        semantic_chunker = Mock()
        loader.return_value = semantic_chunker
        embeddings = Mock()

        build_chunker("semantic", embeddings)

        semantic_chunker.assert_called_once_with(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
            min_chunk_size=200,
        )


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
