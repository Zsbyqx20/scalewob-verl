# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Local stdlib HTTP server hosting the ScaleWoB env corpus.

This server replaces the hosted ``niumascript.com/scalewob-env`` static site
for environments without internet access. It serves the per-env bundles from
``data/scalewob-env/<env>/`` plus the shared bridge/recorder JS, and rewrites
``index.html`` so the external CDN resources (Tailwind, Google Fonts,
animate.css, font-awesome, phosphor-icons) point at the locally-vendored
copies under ``data/scalewob-env/_cdn/``.

URL contract (matches ``scalewob.automation.ScaleWoBAutomation``):

    GET /scalewob-env/<env_id>/index.html[?...cache-busting...]
    GET /scalewob-env/<env_id>/<relative-asset-path>
    GET /scalewob-env/_shared/scalewob-bridge.js
    GET /scalewob-env/_shared/event-recorder.js
    GET /scalewob-env/_cdn/tailwind.js
    GET /scalewob-env/_cdn/<vendor-file>

The HTML rewriting is conservative: only external URLs that we have a vendored
replacement for are rewritten; anything we cannot map is left alone (and will
404 in headless Chrome — harmless if the env doesn't depend on it).

Usage:
    python scripts/serve_scalewob.py --root data/scalewob-env --port 8000
    # then, in the trainer config:
    #   actor_rollout_ref.rollout.scalewob.base_url=http://localhost:8000/scalewob-env
"""

from __future__ import annotations

import argparse
import logging
import mimetypes
import os
import re
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger("serve_scalewob")


# ----------------------------------------------------------------------------
# Unsplash index loader
# ----------------------------------------------------------------------------

# The vendor script writes _cdn/unsplash/INDEX.tsv:
#   photo_id\tw\th\tq\tfit\text\tpath\tbytes
# We load it once at startup and build a (photo_id, w, h, q, fit) -> path map.


class _UnsplashIndex:
    def __init__(self) -> None:
        self._exact: dict[tuple[str, str, str, str, str], Path] = {}
        # For each photo_id, all variants we have, sorted by size desc; used as
        # a fallback when an exact (w, h, q, fit) match isn't vendored.
        self._per_photo: dict[str, list[tuple[tuple[str, str, str, str], Path]]] = {}

    def load(self, tsv_path: Path) -> int:
        if not tsv_path.is_file():
            logger.info("no unsplash index at %s; unsplash rewrites disabled", tsv_path)
            return 0
        with tsv_path.open(encoding="utf-8") as f:
            header = f.readline()  # discard
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 7:
                    continue
                photo_id, w, h, q, fit, ext, rel_path = parts[:7]
                # rel_path is relative to _cdn/ root.
                abs_path = tsv_path.parent.parent / rel_path
                key = (photo_id, w, h, q, fit)
                self._exact[key] = abs_path
                self._per_photo.setdefault(photo_id, []).append(((w, h, q, fit), abs_path))
        logger.info("loaded unsplash index: %d variants, %d photos", len(self._exact), len(self._per_photo))
        return len(self._exact)

    def lookup(self, photo_id: str, w: str, h: str, q: str, fit: str) -> Path | None:
        path = self._exact.get((photo_id, w, h, q, fit))
        if path is not None:
            return path
        candidates = self._per_photo.get(photo_id)
        if not candidates:
            return None
        # No exact match — pick the closest vendored variant.
        #
        # Priority: match the requested w (closest), then h, then any variant
        # of the same photo. q and fit are dropped first (we don't vendor
        # quality/fit as separate files).
        try:
            req_w = int(w) if w else -1
        except ValueError:
            req_w = -1
        try:
            req_h = int(h) if h else -1
        except ValueError:
            req_h = -1

        def score(item: tuple[tuple[str, str, str, str], Path]) -> tuple[int, int, int]:
            (cw, ch, _cq, _cf), _p = item
            try:
                iw = int(cw) if cw else -1
            except ValueError:
                iw = -1
            try:
                ih = int(ch) if ch else -1
            except ValueError:
                ih = -1
            # Lower is better.
            wd = abs(iw - req_w) if req_w >= 0 and iw >= 0 else 10**9
            hd = abs(ih - req_h) if req_h >= 0 and ih >= 0 else 10**9
            # Prefer variants that have a size over no-size (-1).
            size_match = 0 if iw >= 0 or ih >= 0 else 1
            return (size_match, wd, hd)

        candidates.sort(key=score)
        return candidates[0][1]


_UNSPLASH_URL_RE = re.compile(
    r"https?://images\.unsplash\.com/(?:flagged/)?(photo-[0-9]+-[a-z0-9]+)"
    r"(?:\?([^\"'\s<>]*))?"
)


def rewrite_unsplash_urls(text: str, index: "_UnsplashIndex | None") -> str:
    """Replace ``https://images.unsplash.com/...`` with local server URLs.

    When ``index`` is loaded, the rewritten URL has the photo_id + query string
    intact (e.g. ``/scalewob-env/_cdn/unsplash/photo-XYZ?w=800&q=80&fit=crop``)
    so the server can do the lookup at request time and serve the right
    vendored variant.
    """

    def replace(m: re.Match[str]) -> str:
        photo_id = m.group(1)
        qs = m.group(2) or ""
        if index is None:
            # No vendored images — emit a same-origin URL the server will 404,
            # which at least keeps the page origin consistent.
            return f"/scalewob-env/_cdn/unsplash/{photo_id}?{qs}" if qs else f"/scalewob-env/_cdn/unsplash/{photo_id}"
        return f"/scalewob-env/_cdn/unsplash/{photo_id}?{qs}" if qs else f"/scalewob-env/_cdn/unsplash/{photo_id}"

    return _UNSPLASH_URL_RE.sub(replace, text)


# ----------------------------------------------------------------------------
# URL rewriting for index.html
# ----------------------------------------------------------------------------

# Order matters: more specific patterns first.
HTML_REWRITES: list[tuple[re.Pattern[str], str]] = [
    # Shared bridge / recorder (originally at niumascript.com/scalewob-env/...)
    (
        re.compile(r"https?://niumascript\.com/scalewob-env/(scalewob-bridge\.js)"),
        r"/scalewob-env/_shared/\1",
    ),
    (
        re.compile(r"https?://niumascript\.com/scalewob-env/(event-recorder\.js)"),
        r"/scalewob-env/_shared/\1",
    ),
    # Tailwind runtime
    (re.compile(r"https?://cdn\.tailwindcss\.com"), "/scalewob-env/_cdn/tailwind.js"),
    # Animate.css 4.1.1
    (
        re.compile(r"https?://cdnjs\.cloudflare\.com/ajax/libs/animate\.css/4\.1\.1/animate\.min\.css"),
        "/scalewob-env/_cdn/animate.min.css",
    ),
    # FontAwesome 6.4.0 css. We rewrite to a local file; webfonts are resolved
    # relative to that, so the FA CSS uses ../webfonts/<file> which our server
    # handles by serving from _cdn/webfonts/ as long as the URL is /scalewob-env/_cdn/webfonts/...
    # We instead rewrite to a path whose parent is /scalewob-env/_cdn/ so the
    # FA-relative ../webfonts resolves there.
    (
        re.compile(r"https?://cdnjs\.cloudflare\.com/ajax/libs/font-awesome/6\.4\.0/css/all\.min\.css"),
        "/scalewob-env/_cdn/fontawesome.min.css",
    ),
    # Phosphor icons
    (
        re.compile(r"https?://unpkg\.com/@phosphor-icons/web"),
        "/scalewob-env/_cdn/phosphor-web.js",
    ),
    # Google Fonts CSS — we have one vendor CSS per distinct URL. To keep this
    # generic, we point every Google Fonts CSS request at a "router" endpoint
    # that picks the right vendored file based on the query string. See
    # _GoogleFontsCSSHandler. No trailing slash: the vendored CSS's relative
    # `url(googlefonts/<file>)` refs are resolved by the browser against this
    # URL's directory, which must be _cdn/ (not _cdn/googlefonts/) so they
    # land on _cdn/googlefonts/<file> and not _cdn/googlefonts/googlefonts/<file>.
    (
        re.compile(r'https?://fonts\.googleapis\.com/css2\?([^"\' <>]*)'),
        r'/scalewob-env/_cdn/googlefonts?\1',
    ),
]


def rewrite_index_html(html: str) -> str:
    for pattern, replacement in HTML_REWRITES:
        html = pattern.sub(replacement, html)
    return html


# Google Fonts CSS bodies reference gstatic binaries via url(...). The vendor
# script rewrites those to relative paths under googlefonts/. We do the same
# rewrite here so an unvendored fallback (a CSS fetched at request time) would
# also be safe — but in practice the vendor script handles this.
_GFONT_URL_RE = re.compile(r"url\((https://fonts\.gstatic\.com/[^)]+)\)")


def rewrite_googlefonts_css(css: str) -> str:
    # No-op place-holder; vendors already produce resolved CSS. We still strip
    # gstatic URLs defensively in case someone fetches raw CSS at runtime.
    return _GFONT_URL_RE.sub(lambda m: "url(/scalewob-env/_cdn/_gstatic_proxy)", css)


# ----------------------------------------------------------------------------
# HTTP handler
# ----------------------------------------------------------------------------


class _ScaleWoBHandler(SimpleHTTPRequestHandler):
    """Serve ScaleWoB env bundles from a local root directory."""

    root: Path  # set on the class by main()
    shared_dir: Path  # data/scalewob-env (for _shared/* and _cdn/*)
    cdn_dir: Path  # data/scalewob-env/_cdn
    unsplash_index: _UnsplashIndex  # set on the class by main()

    # ------------------------------------------------------------------
    # Boilerplate
    # ------------------------------------------------------------------
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        logger.info("%s - %s", self.address_string(), format % args)

    def end_headers(self) -> None:
        # Disable browser cache so reloads see the latest assets (matches the
        # hosted service behaviour, which is why scalewob.automation appends
        # a nonce query).
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        # /scalewob-env/_shared/<file>
        if path.startswith("/scalewob-env/_shared/"):
            self._serve_from(self.shared_dir, path[len("/scalewob-env/_shared/"):])
            return
        # /scalewob-env/_cdn/...
        if path.startswith("/scalewob-env/_cdn/"):
            self._serve_cdn(path[len("/scalewob-env/_cdn/"):])
            return
        # /scalewob-env/webfonts/<file> — fontawesome.min.css references its
        # webfonts via "../webfonts/<file>" relative to
        # /scalewob-env/_cdn/fontawesome.min.css, which the browser resolves
        # to this path (one level above _cdn/), not /scalewob-env/_cdn/webfonts/.
        if path.startswith("/scalewob-env/webfonts/"):
            self._serve_from(self.cdn_dir / "webfonts", path[len("/scalewob-env/webfonts/"):])
            return
        # /scalewob-env/<env_id>/<asset>
        if path.startswith("/scalewob-env/"):
            sub = path[len("/scalewob-env/"):]
            # top-level helper endpoints
            if sub.startswith("_shared/") or sub.startswith("_cdn/") or sub.startswith("webfonts/"):
                # Should have been caught above; defensive 404.
                self._send_error(HTTPStatus.NOT_FOUND, f"unknown helper path: {sub}")
                return
            env_id, _, rest = sub.partition("/")
            if not env_id:
                self._send_error(HTTPStatus.BAD_REQUEST, "missing env_id")
                return
            env_dir = self.root / env_id
            if not env_dir.is_dir():
                self._send_error(HTTPStatus.NOT_FOUND, f"unknown env: {env_id}")
                return
            self._serve_from(env_dir, rest)
            return
        self._send_error(HTTPStatus.NOT_FOUND, f"unknown path: {path}")

    # ------------------------------------------------------------------
    # Path resolvers
    # ------------------------------------------------------------------
    def _serve_from(self, base: Path, rel: str) -> None:
        rel = rel.lstrip("/")
        if not rel:
            self._send_error(HTTPStatus.BAD_REQUEST, "missing asset path")
            return
        target = (base / rel).resolve()
        # Prevent path traversal
        try:
            target.relative_to(base.resolve())
        except ValueError:
            self._send_error(HTTPStatus.FORBIDDEN, "path traversal blocked")
            return
        if not target.exists():
            self._send_error(HTTPStatus.NOT_FOUND, f"missing asset: {rel}")
            return
        # HTML rewriting: only on index.html
        if target.name == "index.html":
            self._serve_html(target)
            return
        # JS rewriting: every .js file under an env (most data lives in store.js).
        if target.suffix == ".js":
            self._serve_js(target)
            return
        self._serve_static(target)

    def _serve_cdn(self, rel: str) -> None:
        rel = rel.lstrip("/")
        # Google Fonts CSS: served as a routing endpoint that picks the vendored
        # file whose query string matches. This is the CSS entrypoint only
        # ("googlefonts" or "googlefonts/", no further path segment) — actual
        # font binaries referenced from that CSS (googlefonts/<hash>.woff2)
        # fall through to static serving below.
        if rel in ("googlefonts", "googlefonts/"):
            self._serve_googlefonts(rel)
            return
        if rel == "_gstatic_proxy":
            self._send_error(HTTPStatus.NOT_FOUND, "gstatic proxy not vendored")
            return
        # Unsplash image lookup: /scalewob-env/_cdn/unsplash/<photo_id>?<query>
        if rel == "unsplash" or rel.startswith("unsplash/"):
            self._serve_unsplash(rel)
            return
        target = (self.cdn_dir / rel).resolve()
        try:
            target.relative_to(self.cdn_dir.resolve())
        except ValueError:
            self._send_error(HTTPStatus.FORBIDDEN, "path traversal blocked")
            return
        if not target.exists():
            self._send_error(HTTPStatus.NOT_FOUND, f"missing vendor asset: {rel}")
            return
        # Apply CSS rewriting for vendor CSS files
        if target.suffix == ".css":
            self._serve_css(target)
            return
        self._serve_static(target)

    def _serve_unsplash(self, rel: str) -> None:
        # Path form: unsplash/<photo_id>  (query string carries w/h/q/fit)
        photo_id = rel[len("unsplash/"):].lstrip("/") if rel.startswith("unsplash/") else ""
        if not photo_id:
            self._send_error(HTTPStatus.BAD_REQUEST, "missing photo_id")
            return
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query or "")
        w = (qs.get("w") or [""])[0]
        h = (qs.get("h") or [""])[0]
        q = (qs.get("q") or [""])[0]
        fit = (qs.get("fit") or [""])[0]
        path = self.unsplash_index.lookup(photo_id, w, h, q, fit)
        if path is None:
            # A handful of unsplash photo IDs referenced by the env corpus 404
            # upstream (the photos were deleted) and were never vendored — no
            # retry will fix that. Render a same-origin placeholder instead of
            # a broken image so pages still lay out correctly.
            logger.debug("unsplash not vendored, serving placeholder: %s w=%s h=%s", photo_id, w, h)
            self._serve_placeholder_image(w, h)
            return
        self._serve_static(path)

    _PLACEHOLDER_BG = "#d9dde3"
    _PLACEHOLDER_FG = "#9aa1ac"

    def _serve_placeholder_image(self, w: str, h: str) -> None:
        try:
            width = int(w) if w else 400
        except ValueError:
            width = 400
        try:
            height = int(h) if h else max(1, width * 3 // 4)
        except ValueError:
            height = max(1, width * 3 // 4)
        font_size = max(10, min(width, height) // 8)
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
            f'<rect width="100%" height="100%" fill="{self._PLACEHOLDER_BG}"/>'
            f'<text x="50%" y="50%" fill="{self._PLACEHOLDER_FG}" font-family="sans-serif" '
            f'font-size="{font_size}" text-anchor="middle" dominant-baseline="middle">image</text>'
            "</svg>"
        )
        body = svg.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/svg+xml")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_googlefonts(self, rel: str) -> None:
        # URL form: /scalewob-env/_cdn/googlefonts?<query>
        # The vendor script names each CSS file by sha256 of the *full*
        # "https://fonts.googleapis.com/css2?<query>" URL (see _safe_name in
        # vendor_scalewob_cdn.py), so we must reconstruct that same URL here
        # rather than hashing the query string alone.
        parsed = urlparse(self.path)
        query = parsed.query or ""
        if not query:
            self._send_error(HTTPStatus.BAD_REQUEST, "googlefonts endpoint requires ?<query>")
            return
        import hashlib

        full_url = f"https://fonts.googleapis.com/css2?{query}"
        h = hashlib.sha256(full_url.encode("utf-8")).hexdigest()[:16]
        # The vendor script appends a safe-name suffix; we look up by prefix.
        gf_dir = self.cdn_dir / "googlefonts"
        if not gf_dir.is_dir():
            self._send_error(HTTPStatus.NOT_FOUND, "googlefonts/ vendor dir missing")
            return
        candidates = sorted(gf_dir.glob(f"{h}.*"))
        if not candidates:
            self._send_error(
                HTTPStatus.NOT_FOUND,
                f"googlefonts CSS for query {query!r} not vendored (hash {h})",
            )
            return
        self._serve_css(candidates[0])

    # ------------------------------------------------------------------
    # Body writers
    # ------------------------------------------------------------------
    def _serve_static(self, path: Path) -> None:
        ctype, _ = mimetypes.guess_type(str(path))
        if ctype is None:
            ctype = "application/octet-stream"
        try:
            body = path.read_bytes()
        except OSError as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"read error: {exc}")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_html(self, path: Path) -> None:
        try:
            html = path.read_text(encoding="utf-8")
        except OSError as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"read error: {exc}")
            return
        rewritten = rewrite_index_html(html)
        rewritten = rewrite_unsplash_urls(rewritten, self.unsplash_index)
        body = rewritten.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_js(self, path: Path) -> None:
        try:
            js = path.read_text(encoding="utf-8")
        except OSError as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"read error: {exc}")
            return
        js = rewrite_unsplash_urls(js, self.unsplash_index)
        body = js.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/javascript; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_css(self, path: Path) -> None:
        try:
            css = path.read_text(encoding="utf-8")
        except OSError as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"read error: {exc}")
            return
        css = rewrite_googlefonts_css(css)
        body = css.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/css; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ------------------------------------------------------------------
    def _send_error(self, status: HTTPStatus, msg: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(msg)))
        self.end_headers()
        self.wfile.write(msg.encode("utf-8"))


def _script_relative(rel: str) -> Path:
    """Resolve ``rel`` against this script's own directory, not cwd."""
    here = Path(__file__).resolve().parent
    return (here / rel).resolve()


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=_script_relative("../data/scalewob-env"),
        help="Directory containing the env corpus.",
    )
    p.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    p.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    root = args.root.resolve()
    if not root.is_dir():
        print(f"error: root not a directory: {root}", file=sys.stderr)
        return 2
    cdn_dir = root / "_cdn"
    if not cdn_dir.is_dir():
        print(
            f"warning: _cdn/ vendor directory missing at {cdn_dir}; "
            "env pages will fail to load external assets until you run "
            "scripts/vendor_scalewob_cdn.py",
            file=sys.stderr,
        )

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    unsplash_index = _UnsplashIndex()
    unsplash_index.load(cdn_dir / "unsplash" / "INDEX.tsv")

    handler_cls = type(
        "_ScaleWoBHandlerBound",
        (_ScaleWoBHandler,),
        {
            "root": root,
            "shared_dir": root,
            "cdn_dir": cdn_dir,
            "unsplash_index": unsplash_index,
        },
    )

    server = ThreadingHTTPServer((args.host, args.port), handler_cls)
    logger.info("ScaleWoB server listening on http://%s:%d (root=%s)", args.host, args.port, root)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())