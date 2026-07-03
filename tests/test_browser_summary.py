import unittest

from browser_summary import extract_text_from_html, summarize_text


class BrowserSummaryTests(unittest.TestCase):
    def test_extract_text_from_html(self):
        html = "<html><body><h1>Project Summary</h1><p>This is a test paragraph.</p></body></html>"
        text = extract_text_from_html(html)
        self.assertIn("Project Summary", text)
        self.assertIn("This is a test paragraph.", text)

    def test_summarize_text_returns_concise_summary(self):
        text = "This project uses Streamlit and LangGraph for document intelligence. It can ingest PDFs and answer questions from them. The interface is simple and user friendly."
        summary = summarize_text(text, max_sentences=2)
        self.assertIn("Streamlit", summary)
        self.assertIn("LangGraph", summary)


if __name__ == "__main__":
    unittest.main()
