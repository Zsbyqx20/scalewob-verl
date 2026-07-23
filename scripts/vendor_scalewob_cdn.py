#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Vendor external CDN assets used by ScaleWoB env pages into a local _cdn/ directory.

ScaleWoB env pages under ``data/scalewob-env/<env>/index.html`` pull stylesheets,
fonts, and runtime libraries from several public CDNs (Tailwind, Google Fonts,
cdnjs, unpkg). On an offline machine a Python server can serve the env files,
but it must also serve these dependencies locally. This script downloads them
once on a network-enabled machine and writes them under
``data/scalewob-env/_cdn/`` in a layout the local server expects:

    data/scalewob-env/_cdn/
        tailwind.js                          # cdn.tailwindcss.com runtime
        animate.min.css                      # cdnjs animate.css 4.1.1
        fontawesome.min.css                  # cdnjs font-awesome 6.4.0 css
        webfonts/<file>                      # font-awesome woff2 binaries
        phosphor-web.js                      # unpkg @phosphor-icons/web
        googlefonts/<sha>.css                # one CSS per distinct User-Agent-agnostic URL
        googlefonts/<sha>.<ext>              # binary font files referenced inside

The Google Fonts CSS itself only contains ``@font-face`` declarations pointing
at ``fonts.gstatic.com``; this script resolves those URLs, downloads the binary
fonts, and rewrites the CSS to reference them locally.

Usage:
    python scripts/vendor_scalewob_cdn.py --out data/scalewob-env/_cdn
    # produces data/scalewob-env/_cdn/ plus data/scalewob-env/_cdn.tar.gz

The script is intentionally dependency-free (stdlib only) so it can be run on
any host with outbound HTTPS.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urlencode, urlparse


# ----------------------------------------------------------------------------
# Targets observed in data/scalewob-env/*/index.html across the env corpus.
# Each entry: (kind, url, output_subpath).
#   kind = "raw"        -> download as-is to output_subpath
#   kind = "gf-css"     -> download CSS, parse @font-face urls, fetch fonts, rewrite CSS
# ----------------------------------------------------------------------------

USER_AGENT = (
    "Mozilla/5.0 (Linux; Android 10; SM-G981B) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36"
)


RAW_TARGETS: list[tuple[str, str]] = [
    # Tailwind runtime: cdn.tailwindcss.com serves a single JS file.
    ("raw", "https://cdn.tailwindcss.com"),
    # Animate.css 4.1.1
    ("raw", "https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css"),
    # FontAwesome 6.4.0 css
    ("raw", "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"),
    # Phosphor icons web bundle (used by 11 envs including dingtalk / car_xiaomi)
    ("raw", "https://unpkg.com/@phosphor-icons/web"),
]


# Google Fonts CSS URLs are discovered by scanning the env corpus (see
# _scan_google_fonts_urls) rather than hardcoded here, since the corpus grows
# faster than any static list can be kept in sync with. Keep this constant as
# a manually-curated fallback: URLs seen in the past but that might not be
# scannable (e.g. built dynamically in JS rather than a literal string).
GOOGLE_FONTS_CSS_TARGETS: list[str] = [
    "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap",
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
    "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Instagram+Sans:wght@400;700&display=swap",
    "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Noto+Sans+SC:wght@300;400;500;700&display=swap",
]


_GOOGLE_FONTS_URL_RE = re.compile(r"https?://fonts\.googleapis\.com/css2\?[^\"'\s<>]*")


def _scan_google_fonts_urls(env_root: Path) -> list[str]:
    """Walk the env corpus and return every distinct Google Fonts CSS URL referenced."""
    urls: set[str] = set()
    for env_dir in sorted(env_root.iterdir()):
        if not env_dir.is_dir() or env_dir.name.startswith(("_", "__")):
            continue
        for f in env_dir.rglob("*"):
            if f.suffix not in {".js", ".html"}:
                continue
            try:
                content = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            urls.update(_GOOGLE_FONTS_URL_RE.findall(content))
    return sorted(urls)


def _fetch(url: str, timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _safe_name(url: str) -> str:
    # Stable, filesystem-safe identifier for a URL (used for gstatic font filenames).
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    suffix = re.sub(r"[^A-Za-z0-9._-]", "_", url.split("/")[-1])
    return f"{h}.{suffix}" if suffix else h


def _vendor_raw(url: str, out_root: Path) -> None:
    name = url.rsplit("/", 1)[-1] or "index"
    # Tailwind's cdn.tailwindcss.com returns a JS; the URL has no extension so
    # we normalise it to tailwind.js for predictable serving.
    if url.endswith("cdn.tailwindcss.com") or url == "https://cdn.tailwindcss.com":
        out = out_root / "tailwind.js"
    elif "animate.min.css" in url:
        out = out_root / "animate.min.css"
    elif "font-awesome" in url and url.endswith(".css"):
        out = out_root / "fontawesome.min.css"
    elif "phosphor-icons/web" in url:
        out = out_root / "phosphor-web.js"
    else:
        out = out_root / name
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetch {url} -> {out.relative_to(out_root.parent)}", flush=True)
    data = _fetch(url)
    out.write_bytes(data)


# Google Fonts CSS @font-face -> fetch each gstatic binary, rewrite the CSS.
_GFONT_URL_RE = re.compile(r"url\((https://fonts\.gstatic\.com/[^)]+)\)")


def _vendor_google_fonts_css(url: str, out_root: Path) -> None:
    css_dir = out_root / "googlefonts"
    css_dir.mkdir(parents=True, exist_ok=True)
    css_name = _safe_name(url)
    css_path = css_dir / css_name
    print(f"  fetch Google Fonts CSS: {url}", flush=True)
    try:
        css_text = _fetch(url).decode("utf-8")
    except urllib.error.URLError as exc:
        print(f"  WARN: failed to fetch Google Fonts CSS {url}: {exc}", file=sys.stderr)
        return

    seen: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        font_url = match.group(1)
        if font_url in seen:
            return f"url(googlefonts/{seen[font_url]})"
        local_name = _safe_name(font_url)
        seen[font_url] = local_name
        out = css_dir / local_name
        if not out.exists():
            print(f"    fetch font {font_url} -> googlefonts/{local_name}", flush=True)
            try:
                data = _fetch(font_url)
            except urllib.error.URLError as exc:
                print(f"    WARN: failed to fetch {font_url}: {exc}", file=sys.stderr)
                return match.group(0)
            out.write_bytes(data)
        return f"url(googlefonts/{local_name})"

    rewritten = _GFONT_URL_RE.sub(replace, css_text)
    css_path.write_text(rewritten, encoding="utf-8")


def _repack_fonts_in_fa_css(out_root: Path) -> None:
    """FontAwesome's all.min.css references webfonts via ../webfonts/..."""
    fa_css = out_root / "fontawesome.min.css"
    if not fa_css.exists():
        return
    css_text = fa_css.read_text(encoding="utf-8")
    # Find ../webfonts/<file> URLs by walking the css for url(...)
    urls = set(re.findall(r"url\(\.\./webfonts/([^)]+)\)", css_text))
    if not urls:
        return
    webfonts_dir = out_root / "webfonts"
    webfonts_dir.mkdir(parents=True, exist_ok=True)
    base = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/webfonts/"
    for rel in sorted(urls):
        out = webfonts_dir / rel
        if out.exists():
            continue
        url = base + rel
        print(f"  fetch FA webfont {url}", flush=True)
        try:
            data = _fetch(url)
        except urllib.error.URLError as exc:
            print(f"    WARN: failed to fetch {url}: {exc}", file=sys.stderr)
            continue
        out.write_bytes(data)


def _tar(out_root: Path, tar_path: Path) -> None:
    print(f"  tarring -> {tar_path}", flush=True)
    if tar_path.exists():
        tar_path.unlink()
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(out_root, arcname=out_root.name)


# ----------------------------------------------------------------------------
# Unsplash image vendoring
# ----------------------------------------------------------------------------

# Match references like
#   https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?w=800&q=80&fit=crop
#   https://images.unsplash.com/flagged/photo-1572987337946-de0a00159d91?...
# The id is the leading numeric-hash suffix; everything after `?` is query.
_UNSPLASH_RE = re.compile(
    r"https?://images\.unsplash\.com/(?:flagged/)?(photo-[0-9]+-[a-z0-9]+)(?:\?([^'\" <>]+))?"
)


def _scan_unsplash_variants(env_root: Path) -> dict[tuple[str, str, str, str, str], int]:
    """Walk the env corpus and return a map of (photo_id, w, h, q, fit) -> ref count.

    We treat each ``(photo_id, w, h, q, fit)`` tuple as a distinct image fetch.
    Other query keys (``ixid``, ``ixlib``, ``auto``, ``fm``) are dropped because
    they either are analytics or affect response format — we let unsplash pick
    JPEG by default and rewrite ``fm`` later if needed.
    """
    variants: dict[tuple[str, str, str, str, str], int] = {}
    photo_ids: set[str] = set()
    scanned = 0
    for env_dir in sorted(env_root.iterdir()):
        if not env_dir.is_dir():
            continue
        if env_dir.name.startswith(("_", "__")):
            continue
        for f in env_dir.rglob("*"):
            if f.suffix not in {".js", ".html"}:
                continue
            try:
                content = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            scanned += 1
            for m in _UNSPLASH_RE.finditer(content):
                photo_id = m.group(1)
                photo_ids.add(photo_id)
                qs = parse_qs(m.group(2) or "")
                w = (qs.get("w") or [""])[0]
                h = (qs.get("h") or [""])[0]
                q = (qs.get("q") or [""])[0]
                fit = (qs.get("fit") or [""])[0]
                key = (photo_id, w, h, q, fit)
                variants[key] = variants.get(key, 0) + 1
    print(
        f"  scanned {scanned} files; {len(photo_ids)} distinct photo IDs; "
        f"{len(variants)} distinct (id, w, h, q, fit) variants",
        flush=True,
    )
    return variants


def _ext_from_content_type(ct: str | None) -> str:
    if not ct:
        return "jpg"
    ct = ct.split(";")[0].strip().lower()
    return {
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/png": "png",
        "image/webp": "webp",
        "image/avif": "avif",
        "image/gif": "gif",
    }.get(ct, "jpg")


def _variant_filename(photo_id: str, w: str, h: str, q: str, fit: str, ext: str) -> str:
    parts = [photo_id]
    parts.append(f"w{w}" if w else "w-")
    parts.append(f"h{h}" if h else "h-")
    parts.append(f"q{q}" if q else "q-")
    parts.append(f"fit{fit}" if fit else "fit-")
    return "_".join(parts) + f".{ext}"


def _vendor_unsplash(env_root: Path, out_root: Path) -> None:
    variants = _scan_unsplash_variants(env_root)
    if not variants:
        print("  no unsplash references found; skipping", flush=True)
        return

    img_root = out_root / "unsplash"
    img_root.mkdir(parents=True, exist_ok=True)
    index_lines: list[str] = ["photo_id\tw\th\tq\tfit\text\tpath\tbytes"]
    total_bytes = 0
    failures: list[str] = []
    # Sort for deterministic output and easier debugging.
    for (photo_id, w, h, q, fit), count in sorted(variants.items()):
        ext = "jpg"
        path_rel = ""  # filled once we know the extension
        # Build the URL keeping only params that have values; use `auto=format`
        # so unsplash picks the best format for the request Accept header.
        params: list[tuple[str, str]] = [("auto", "format")]
        for k, v in (("w", w), ("h", h), ("q", q), ("fit", fit)):
            if v:
                params.append((k, v))
        url = f"https://images.unsplash.com/{photo_id}?{urlencode(params)}"
        out_dir = img_root / photo_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / _variant_filename(photo_id, w, h, q, fit, ext)
        if out_path.exists():
            size = out_path.stat().st_size
        else:
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": USER_AGENT, "Accept": "image/*"}
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = resp.read()
                    ct = resp.headers.get("Content-Type")
                ext = _ext_from_content_type(ct)
                out_path = out_dir / _variant_filename(photo_id, w, h, q, fit, ext)
                if not out_path.exists():
                    out_path.write_bytes(data)
                size = len(data)
            except Exception as exc:
                failures.append(f"{photo_id} w={w} h={h} q={q} fit={fit}: {exc}")
                continue
        total_bytes += size
        index_lines.append(
            f"{photo_id}\t{w}\t{h}\t{q}\t{fit}\t{ext}\t{out_path.relative_to(out_root)}\t{size}"
        )

    (img_root / "INDEX.tsv").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print(
        f"  fetched {len(index_lines) - 1} variants, {total_bytes / (1024 * 1024):.1f} MB total",
        flush=True,
    )
    if failures:
        print(f"  WARN: {len(failures)} unsplash fetches failed", file=sys.stderr)
        for f in failures[:20]:
            print(f"    {f}", file=sys.stderr)


def _script_relative(rel: str) -> Path:
    """Resolve ``rel`` against this script's own directory, not cwd.

    Lets the script be invoked from any working directory while still finding
    the env corpus that ships alongside it in the repo.
    """
    here = Path(__file__).resolve().parent
    return (here / rel).resolve()


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=_script_relative("../data/scalewob-env/_cdn"),
        help="Output directory for vendored assets.",
    )
    p.add_argument(
        "--env-root",
        type=Path,
        default=_script_relative("../data/scalewob-env"),
        help="Root of the env corpus to scan for unsplash URLs.",
    )
    p.add_argument(
        "--no-tar",
        action="store_true",
        help="Skip producing the .tar.gz archive (only write the directory).",
    )
    p.add_argument(
        "--skip-unsplash",
        action="store_true",
        help="Skip vendoring unsplash images (only vendor the JS/CSS/font assets).",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    out_root: Path = args.out.resolve()
    if out_root.exists():
        print(f"WARN: removing existing {out_root}", flush=True)
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print("Vendoring raw CDN assets...", flush=True)
    for kind, url in RAW_TARGETS:
        assert kind == "raw"
        _vendor_raw(url, out_root)

    print("Vendoring FontAwesome webfonts referenced in fontawesome.min.css...", flush=True)
    _repack_fonts_in_fa_css(out_root)

    print(f"Scanning {args.env_root} for Google Fonts CSS references...", flush=True)
    scanned_gfont_urls = _scan_google_fonts_urls(args.env_root)
    gfont_urls = sorted(set(scanned_gfont_urls) | set(GOOGLE_FONTS_CSS_TARGETS))
    print(f"  found {len(scanned_gfont_urls)} distinct URLs in corpus, {len(gfont_urls)} total to vendor", flush=True)
    print("Vendoring Google Fonts CSS + binaries...", flush=True)
    for url in gfont_urls:
        _vendor_google_fonts_css(url, out_root)

    if not args.skip_unsplash:
        print(
            f"Scanning {args.env_root} for unsplash image references...",
            flush=True,
        )
        _vendor_unsplash(args.env_root, out_root)
    else:
        print("Skipping unsplash (--skip-unsplash).", flush=True)

    if not args.no_tar:
        _tar(out_root, out_root.with_suffix(".tar.gz"))

    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())