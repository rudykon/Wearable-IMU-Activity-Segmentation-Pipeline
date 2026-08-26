"""Validate that MkDocs emits MathJax-ready academic equations."""

from __future__ import annotations

import json
import re
import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
SITE_BASE_PATH = "/Wearable-IMU-Activity-Segmentation-Pipeline/"


class ReferenceParser(HTMLParser):
    """Collect local references and anchors from one rendered page."""

    def __init__(self) -> None:
        super().__init__()
        self.references: list[tuple[str, str]] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs) -> None:
        attributes = dict(attrs)
        if attributes.get("id"):
            self.ids.add(attributes["id"])
        key = "href" if tag in {"a", "link"} else "src"
        if tag in {"a", "img", "script", "link", "source"} and attributes.get(key):
            self.references.append((tag, attributes[key]))


class DocumentationMathTests(unittest.TestCase):
    def test_primary_navigation_promotes_both_demos(self) -> None:
        config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        nav = config.split("\nnav:\n", maxsplit=1)[1]
        top_level_labels = re.findall(r"^  - ([^:]+):", nav, flags=re.MULTILINE)
        self.assertEqual(
            top_level_labels,
            [
                "Overview",
                "Method",
                "Results",
                "Reproduce",
                "Android Demo",
                "HF Demo",
                "GitHub",
            ],
        )
        self.assertNotIn("\n      - Android:", nav)

    def test_android_downloads_and_data_request_are_pinned(self) -> None:
        apk_url = (
            "https://github.com/rudykon/"
            "Wearable-IMU-Activity-Segmentation-Pipeline/releases/download/"
            "android-demo-v1.0-preview/"
            "hls-har-android-demo-v1.0-arm64-v8a-debug.apk"
        )
        release_url = (
            "https://github.com/rudykon/"
            "Wearable-IMU-Activity-Segmentation-Pipeline/releases/tag/"
            "android-demo-v1.0-preview"
        )
        form_url = "https://wj.qq.com/s2/26600660/1b91"

        for page in [
            ROOT / "docs" / "deployment" / "android.md",
            ROOT / "docs" / "deployment" / "android.zh.md",
        ]:
            with self.subTest(page=page):
                source = page.read_text(encoding="utf-8")
                self.assertIn(apk_url, source)
                self.assertIn(release_url, source)
                self.assertIn("assets/android/synthetic_activity_imu.tsv", source)

        for page in [
            ROOT / "docs" / "guide" / "data.md",
            ROOT / "docs" / "guide" / "data.zh.md",
        ]:
            with self.subTest(page=page):
                self.assertIn(form_url, page.read_text(encoding="utf-8"))

        canonical_sample = ROOT / "demo" / "examples" / "synthetic_activity_imu.tsv"
        download_sample = (
            ROOT / "docs" / "assets" / "android" / "synthetic_activity_imu.tsv"
        )
        self.assertTrue(download_sample.is_file())
        self.assertGreater(download_sample.stat().st_size, 500_000)
        self.assertEqual(download_sample.read_bytes(), canonical_sample.read_bytes())

        extra_css = (ROOT / "docs" / "stylesheets" / "extra.css").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            ":not(.md-button):not(.hero-button):not(.route-card):not(.demo-action)",
            extra_css,
        )

    def test_math_configuration_is_enabled(self) -> None:
        config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        self.assertIn("pymdownx.arithmatex", config)
        self.assertIn("generic: true", config)
        self.assertIn("javascripts/mathjax.js", config)
        self.assertIn("javascripts/vendor/mathjax/tex-mml-chtml.js", config)
        self.assertNotIn("cdn.jsdelivr.net", config)
        self.assertIn("stylesheets/math.css", config)

        vendor = ROOT / "docs" / "javascripts" / "vendor" / "mathjax"
        self.assertGreater((vendor / "tex-mml-chtml.js").stat().st_size, 1_000_000)
        self.assertTrue((vendor / "input" / "tex" / "extensions" / "ams.js").is_file())
        self.assertGreaterEqual(
            len(list((vendor / "output" / "chtml" / "fonts" / "woff-v2").glob("*.woff"))),
            20,
        )
        self.assertTrue((vendor / "LICENSE").is_file())

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

        for page in [
            ROOT / "docs" / "guide" / "pipeline.md",
            ROOT / "docs" / "guide" / "pipeline.zh.md",
        ]:
            with self.subTest(page=page):
                self.assertGreaterEqual(
                    page.read_text(encoding="utf-8").count("\\begin{aligned}"),
                    2,
                )

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

        chinese_home = (ROOT / "docs" / "index.zh.md").read_text(encoding="utf-8")
        self.assertIn('href="../assets/fig02_overall_framework.png"', chinese_home)
        self.assertIn('src="../assets/fig02_overall_framework.png"', chinese_home)
        self.assertIn('src="../assets/demo-results-paper-notation.jpg"', chinese_home)

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
                self.assertIn("javascripts/vendor/mathjax/tex-mml-chtml.js", html)
                self.assertNotIn("cdn.jsdelivr.net", html)
                self.assertNotIn('<div class="method-chain">', html)

    def test_built_chinese_home_assets_resolve(self) -> None:
        chinese_home = SITE / "zh" / "index.html"
        self.assertTrue(chinese_home.is_file())
        html = chinese_home.read_text(encoding="utf-8")
        for relative_path in [
            "../assets/fig02_overall_framework.png",
            "../assets/demo-results-paper-notation.jpg",
        ]:
            with self.subTest(relative_path=relative_path):
                self.assertIn(relative_path, html)
                self.assertTrue((chinese_home.parent / relative_path).resolve().is_file())

        homepage_css = (ROOT / "docs" / "stylesheets" / "showcase.css").read_text(
            encoding="utf-8"
        )
        self.assertRegex(
            homepage_css,
            r"\.home-section \.pipeline-frame img\s*\{[^}]*min-width:\s*0;",
        )

    def test_all_rendered_local_references_resolve(self) -> None:
        anchor_cache: dict[Path, set[str]] = {}
        for page in sorted(SITE.rglob("*.html")):
            if page.name == "404.html":
                continue
            parser = ReferenceParser()
            parser.feed(page.read_text(encoding="utf-8"))
            for tag, reference in parser.references:
                parsed = urlsplit(reference)
                if (
                    parsed.scheme
                    or parsed.netloc
                    or reference.startswith(("mailto:", "tel:", "data:", "#"))
                    or not parsed.path
                ):
                    continue

                path = unquote(parsed.path)
                if path.startswith(SITE_BASE_PATH):
                    target = SITE / path[len(SITE_BASE_PATH) :]
                elif path.startswith("/"):
                    continue
                else:
                    target = page.parent / path

                if path.endswith("/") or target.is_dir():
                    target /= "index.html"
                self.assertTrue(
                    target.exists(),
                    f"Broken {tag} reference in {page.relative_to(SITE)}: {reference}",
                )

                if parsed.fragment and target.suffix == ".html":
                    if target not in anchor_cache:
                        target_parser = ReferenceParser()
                        target_parser.feed(target.read_text(encoding="utf-8"))
                        anchor_cache[target] = target_parser.ids
                    self.assertIn(
                        unquote(parsed.fragment),
                        anchor_cache[target],
                        f"Broken anchor in {page.relative_to(SITE)}: {reference}",
                    )

    def test_legacy_pages_are_noindex_and_not_discoverable(self) -> None:
        legacy_pages = [
            "ASSETS/index.html",
            "USAGE/index.html",
            "context/use-cases/index.html",
            "zh/ASSETS/index.html",
            "zh/USAGE/index.html",
            "zh/context/use-cases/index.html",
        ]
        for relative_path in legacy_pages:
            with self.subTest(relative_path=relative_path):
                page = SITE / relative_path
                self.assertTrue(page.is_file())
                self.assertIn(
                    '<meta name="robots" content="noindex, follow">',
                    page.read_text(encoding="utf-8"),
                )

        for search_index in [
            SITE / "search" / "search_index.json",
            SITE / "zh" / "search" / "search_index.json",
        ]:
            if search_index.is_file():
                locations = {
                    item["location"]
                    for item in json.loads(search_index.read_text(encoding="utf-8"))["docs"]
                }
                for legacy_prefix in ["ASSETS/", "USAGE/", "context/use-cases/"]:
                    self.assertFalse(
                        any(location.startswith(legacy_prefix) for location in locations),
                        f"Legacy page remained in {search_index}: {legacy_prefix}",
                    )

        sitemap = (SITE / "sitemap.xml").read_text(encoding="utf-8")
        for legacy_path in [
            "/ASSETS/",
            "/USAGE/",
            "/context/use-cases/",
            "/zh/ASSETS/",
            "/zh/USAGE/",
            "/zh/context/use-cases/",
        ]:
            self.assertNotIn(legacy_path, sitemap)

    def test_pages_emit_social_preview_metadata(self) -> None:
        image_url = (
            "https://rudykon.github.io/Wearable-IMU-Activity-Segmentation-Pipeline/"
            "assets/fig02_overall_framework.png"
        )
        for page in [SITE / "index.html", SITE / "zh" / "index.html"]:
            with self.subTest(page=page):
                html = page.read_text(encoding="utf-8")
                self.assertIn('<meta property="og:type" content="website">', html)
                self.assertIn(f'<meta property="og:image" content="{image_url}">', html)
                self.assertIn('<meta name="twitter:card" content="summary_large_image">', html)

    def test_repository_has_machine_readable_citation(self) -> None:
        citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        self.assertIn("cff-version: 1.2.0", citation)
        self.assertIn('family-names: "Kong"', citation)
        self.assertIn("version: 0.1.0-research-preview", citation)
        self.assertIn("date-released: 2026-08-26", citation)
        self.assertIn(
            "releases/tag/v0.1.0-research-preview",
            citation,
        )
        self.assertIn("license: Apache-2.0", citation)

    def test_built_demo_and_data_pages_keep_their_actions(self) -> None:
        android_pages = [
            SITE / "deployment" / "android" / "index.html",
            SITE / "zh" / "deployment" / "android" / "index.html",
        ]
        for page in android_pages:
            with self.subTest(page=page):
                self.assertTrue(page.is_file(), f"Missing built page: {page}")
                html = page.read_text(encoding="utf-8")
                self.assertIn("android-demo-v1.0-preview", html)
                self.assertIn("synthetic_activity_imu.tsv", html)
                self.assertIn('id="download-and-try"', html)
                self.assertIn('id="capabilities"', html)

        chinese_data = SITE / "zh" / "guide" / "data" / "index.html"
        self.assertTrue(chinese_data.is_file(), f"Missing built page: {chinese_data}")
        self.assertIn(
            "https://wj.qq.com/s2/26600660/1b91",
            chinese_data.read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    unittest.main()
