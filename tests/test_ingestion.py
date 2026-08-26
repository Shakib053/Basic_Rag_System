import tempfile
import unittest

from pathlib import Path
from unittest.mock import Mock, patch

from ingestion.text_pipeline import (
    build_chunker,
    get_chunking_strategy,
)
from ingestion.store import (
    rebuild_vector_store,
    replace_vector_store,
)

class ChunkerConfigurationTests(unittest.TestCase):
    @patch.dict("ingestion.text_pipeline.os.environ", {}, clear=True)
    def test_semantic_chunking_is_default(self):
        self.assertEqual(get_chunking_strategy(), "semantic")

    @patch.dict(
        "ingestion.text_pipeline.os.environ",
        {"CHUNKING_STRATEGY": "semantic"},
        clear=True,
    )
    def test_semantic_strategy_can_be_selected_from_environment(self):
        self.assertEqual(get_chunking_strategy(), "semantic")

    @patch.dict(
        "ingestion.text_pipeline.os.environ",
        {"CHUNKING_STRATEGY": "recursive"},
        clear=True,
    )
    def test_recursive_strategy_can_be_selected_from_environment(self):
        self.assertEqual(get_chunking_strategy(), "recursive")

    @patch("ingestion.text_pipeline.build_recursive_chunker")
    def test_recursive_strategy_routes_to_recursive_chunker(self, builder):
        recursive_chunker = Mock()
        builder.return_value = recursive_chunker

        result = build_chunker("recursive", Mock())

        builder.assert_called_once_with()
        self.assertIs(result, recursive_chunker)

    def test_invalid_chunking_strategy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported CHUNKING_STRATEGY"):
            get_chunking_strategy("fixed")

    @patch("ingestion.text_pipeline._load_semantic_chunker_class")
    def test_semantic_chunker_uses_expected_configuration(self, loader):
        semantic_chunker = Mock()
        loader.return_value = semantic_chunker
        embeddings = Mock()

        build_chunker("semantic", embeddings)

        semantic_chunker.assert_called_once_with(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90,
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

            with patch("ingestion.store._rename_path", side_effect=fail_second_rename):
                with self.assertRaisesRegex(OSError, "swap failed"):
                    replace_vector_store(staging, live)

            self.assertEqual((live / "value.txt").read_text(), "old")
            self.assertTrue(staging.exists())
            self.assertEqual(list(root.glob(".chroma_db.backup-*")), [])


class IngestionEntrypointTests(unittest.TestCase):
    @patch("ingestion.run.run_image_pipeline")
    @patch("ingestion.run.run_text_pipeline")
    @patch("ingestion.run.verify_chroma_native_bindings")
    @patch("ingestion.run.load_dotenv")
    @patch("sys.argv", ["ingestion.run", "--text-only"])
    @patch.dict(
        "text_vectorstore.os.environ",
        {"VECTOR_STORE_PROVIDER": "chroma"},
        clear=True,
    )
    def test_text_only_chroma_verifies_chroma_bindings(
        self,
        _load_dotenv,
        verify_chroma,
        run_text,
        run_image,
    ):
        from ingestion.run import main

        main()

        verify_chroma.assert_called_once_with()
        run_text.assert_called_once_with(None)
        run_image.assert_not_called()

    @patch("ingestion.run.run_image_pipeline")
    @patch("ingestion.run.run_text_pipeline")
    @patch("ingestion.run.verify_chroma_native_bindings")
    @patch("ingestion.run.load_dotenv")
    @patch("sys.argv", ["ingestion.run", "--text-only"])
    @patch.dict(
        "text_vectorstore.os.environ",
        {"VECTOR_STORE_PROVIDER": "qdrant"},
        clear=True,
    )
    def test_text_only_qdrant_skips_chroma_binding_check(
        self,
        _load_dotenv,
        verify_chroma,
        run_text,
        run_image,
    ):
        from ingestion.run import main

        main()

        verify_chroma.assert_not_called()
        run_text.assert_called_once_with(None)
        run_image.assert_not_called()


if __name__ == "__main__":
    unittest.main()
