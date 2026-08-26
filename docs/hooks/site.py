"""Post-build hygiene for compatibility-only documentation pages."""

from __future__ import annotations

import gzip
from pathlib import Path
from urllib.parse import urlsplit
from xml.etree import ElementTree

SITEMAP_NAMESPACE = "http://www.sitemaps.org/schemas/sitemap/0.9"
LEGACY_PATHS = {
    "/ASSETS/",
    "/USAGE/",
    "/context/use-cases/",
    "/zh/ASSETS/",
    "/zh/USAGE/",
    "/zh/context/use-cases/",
}


def _remove_legacy_urls(payload: bytes, site_url: str) -> bytes:
    """Return one sitemap without compatibility-only page URLs."""

    root = ElementTree.fromstring(payload)
    base_path = urlsplit(site_url).path.rstrip("/")
    namespace = {"sitemap": SITEMAP_NAMESPACE}

    for entry in list(root.findall("sitemap:url", namespace)):
        location = entry.findtext("sitemap:loc", default="", namespaces=namespace)
        path = urlsplit(location).path
        if base_path and path.startswith(base_path):
            path = path[len(base_path) :]
        normalized = "/" + path.strip("/") + "/"
        if normalized in LEGACY_PATHS:
            root.remove(entry)

    ElementTree.register_namespace("", SITEMAP_NAMESPACE)
    return ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)


def on_post_build(*, config, **_kwargs) -> None:
    """Keep preserved legacy pages out of the public sitemap."""

    site_dir = Path(config.site_dir)
    plain_sitemap = site_dir / "sitemap.xml"
    if plain_sitemap.is_file():
        plain_sitemap.write_bytes(
            _remove_legacy_urls(plain_sitemap.read_bytes(), config.site_url)
        )

    compressed_sitemap = site_dir / "sitemap.xml.gz"
    if compressed_sitemap.is_file():
        with gzip.open(compressed_sitemap, "rb") as handle:
            payload = handle.read()
        with gzip.open(compressed_sitemap, "wb") as handle:
            handle.write(_remove_legacy_urls(payload, config.site_url))
