"""The ``aerox`` command.

aerox view                     serve the aircraft viewer and open the browser
aerox view A320 B789           only these aircraft
aerox view --port 9000 --no-open
aerox view --out viewer.html   write the page to a file instead of serving it
"""

from __future__ import annotations

import argparse
import tempfile
import webbrowser
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def viewer_page(codes: Sequence[str] = ()) -> bytes:
    """The viewer page for the given ICAO codes, or for all aircraft."""

    from .aircraft import load_aircraft
    from .plot import write_viewer

    aircraft = [load_aircraft(code) for code in codes] or None
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "viewer.html"
        write_viewer(path, aircraft)
        return path.read_bytes()


def make_server(page: bytes, host: str = "127.0.0.1", port: int = 8765):
    """An HTTP server that returns the page at ``/``.

    When the port is taken, a free port is used instead.
    """

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - name required by BaseHTTPRequestHandler
            if self.path.split("?")[0] not in ("/", "/index.html"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(page)

        def log_message(self, format, *args):  # keep the terminal quiet
            pass

    try:
        return ThreadingHTTPServer((host, port), Handler)
    except OSError:
        return ThreadingHTTPServer((host, 0), Handler)


def _view(args: argparse.Namespace) -> int:
    page = viewer_page(args.icao)
    if args.out:
        args.out.write_bytes(page)
        print(args.out.resolve())
        if not args.no_open:
            webbrowser.open(args.out.resolve().as_uri())
        return 0
    server = make_server(page, args.host, args.port)
    url = f"http://{args.host}:{server.server_address[1]}/"
    print(f"Aerox aircraft viewer at {url}  (Ctrl+C to stop)")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aerox", description="Aerox aircraft tools.")
    commands = parser.add_subparsers(dest="command", required=True)
    view = commands.add_parser(
        "view",
        help="view aircraft in 2D and 3D in the browser",
        description=(
            "Serve the aircraft viewer on localhost and open it in the browser. "
            "The page loads three.js from a CDN, so it needs an internet connection."
        ),
    )
    view.add_argument(
        "icao", nargs="*", help="ICAO type codes; all aircraft when empty"
    )
    view.add_argument("--host", default="127.0.0.1")
    view.add_argument("--port", type=int, default=8765)
    view.add_argument("--no-open", action="store_true", help="do not open a browser")
    view.add_argument("--out", type=Path, help="write the page to a file instead")
    view.set_defaults(handler=_view)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
