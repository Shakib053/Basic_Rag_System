import unittest

from langchain_core.documents import Document

from context_formatting import build_combined_context


class ContextFormattingTests(unittest.TestCase):
    def test_builds_combined_context_with_text_and_image_references(self):
        text_docs = [
            Document(page_content="First relevant text chunk."),
            Document(page_content="Second relevant text chunk."),
        ]
        image_doc = Document(
            page_content="data/extracted_images/sample/page_1_img_1.png",
            metadata={
                "source": "data/sample.pdf",
                "page": 1,
                "image_index": 1,
            },
        )

        context = build_combined_context(text_docs, [(image_doc, 0.42)])

        self.assertIn("Text context:\nFirst relevant text chunk.", context)
        self.assertIn("Second relevant text chunk.", context)
        self.assertIn("Image references:", context)
        self.assertIn("source: data/sample.pdf", context)
        self.assertIn("path: data/extracted_images/sample/page_1_img_1.png", context)
        self.assertIn("distance: 0.4200", context)

    def test_omits_image_section_when_no_images_are_available(self):
        text_docs = [Document(page_content="Only text context.")]

        context = build_combined_context(text_docs, [])

        self.assertEqual(context, "Only text context.")
        self.assertNotIn("Image references:", context)


if __name__ == "__main__":
    unittest.main()
