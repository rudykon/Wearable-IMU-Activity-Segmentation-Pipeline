"""Validate that MkDocs emits MathJax-ready academic equations."""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"


class DocumentationMathTests(unittest.TestCase):
    def test_math_configuration_is_enabled(self) -> None:
        config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        self.assertIn("pymdownx.arithmatex", config)
        self.assertIn("generic: true", config)
        self.assertIn("javascripts/mathjax.js", config)
        self.assertIn("mathjax@3.2.2", config)
        self.assertIn("stylesheets/math.css", config)

    def test_equation_sources_use_latex_not_html_subscripts(self) -> None:
        source_paths = [
            ROOT / "docs" / "guide" / "pipeline.md",
            ROOT / "docs" / "guide" / "pipeline.zh.md",
            ROOT / "docs" / "guide" / "data.md",
            ROOT / "docs" / "guide" / "data.zh.md",
            ROOT / "docs" / "guide" / "training.md",
            ROOT / "docs" / "guide" / "training.zh.md",
            ROOT / "docs" / "guide" / "evaluation.md",
            ROOT / "docs" / "guide" / "evaluation.zh.md",
            ROOT / "docs" / "deployment" / "hugging-face.md",
            ROOT / "docs" / "deployment" / "hugging-face.zh.md",
        ]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in source_paths)
        self.assertNotIn("<sub>", combined)
        self.assertNotIn("<sup>", combined)
        self.assertNotIn('<div class="method-chain">', combined)
        self.assertIn("\\operatorname{IoU}", combined)
        self.assertIn("\\mathcal{L}", combined)
        self.assertIn("\\widetilde{\\mathbf{p}}", combined)
        self.assertIn("\\(a_x,a_y,a_z\\)", combined)
        self.assertIn("\\(\\omega_x,\\omega_y,\\omega_z\\)", combined)

    def test_scientific_pages_use_paper_channel_notation(self) -> None:
        source_paths = [
            ROOT / "docs" / "guide" / "pipeline.md",
            ROOT / "docs" / "guide" / "pipeline.zh.md",
            ROOT / "docs" / "guide" / "data.md",
            ROOT / "docs" / "guide" / "data.zh.md",
        ]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in source_paths)
        self.assertNotIn("ACC_X", combined)
        self.assertNotIn("GYRO_X", combined)
        self.assertIn("a_x", combined)
        self.assertIn("\\omega_x", combined)

    def test_demo_pages_use_the_current_notation_screenshot(self) -> None:
        screenshot = ROOT / "docs" / "assets" / "demo-results-paper-notation.jpg"
        self.assertTrue(screenshot.is_file())
        self.assertGreater(screenshot.stat().st_size, 100_000)
        page_paths = [
            ROOT / "docs" / "index.md",
            ROOT / "docs" / "index.zh.md",
            ROOT / "docs" / "deployment" / "hugging-face.md",
            ROOT / "docs" / "deployment" / "hugging-face.zh.md",
        ]
        for page in page_paths:
            source = page.read_text(encoding="utf-8")
            self.assertIn("demo-results-paper-notation.jpg", source)
            self.assertNotIn("demo-results.jpg", source)

    def test_built_pages_contain_arithmatex_and_mathjax(self) -> None:
        pages = [
            SITE / "guide" / "pipeline" / "index.html",
            SITE / "guide" / "data" / "index.html",
            SITE / "guide" / "training" / "index.html",
            SITE / "guide" / "evaluation" / "index.html",
            SITE / "deployment" / "hugging-face" / "index.html",
            SITE / "zh" / "guide" / "pipeline" / "index.html",
            SITE / "zh" / "guide" / "data" / "index.html",
            SITE / "zh" / "guide" / "training" / "index.html",
            SITE / "zh" / "guide" / "evaluation" / "index.html",
            SITE / "zh" / "deployment" / "hugging-face" / "index.html",
        ]
        for page in pages:
            with self.subTest(page=page):
                self.assertTrue(page.is_file(), f"Missing built page: {page}")
                html = page.read_text(encoding="utf-8")
                self.assertIn('class="arithmatex"', html)
                self.assertIn("javascripts/mathjax.js", html)
                self.assertIn("mathjax@3.2.2", html)
                self.assertNotIn('<div class="method-chain">', html)


if __name__ == "__main__":
    unittest.main()
