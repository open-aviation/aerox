import threading
import urllib.request

import pytest

from aerox.cli import build_parser, main, make_server, viewer_page


def test_view_writes_the_page_to_a_file(tmp_path):
    out = tmp_path / "viewer.html"
    assert main(["view", "A320", "--out", str(out), "--no-open"]) == 0
    html = out.read_text(encoding="utf-8")
    assert '"A320":' in html and "__AIRCRAFT_DATA__" not in html


def test_view_server_returns_the_page_and_404_elsewhere():
    page = viewer_page(["A320"])
    server = make_server(page, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        with urllib.request.urlopen(base + "/") as response:
            assert response.status == 200
            assert response.read() == page
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(base + "/other")
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()


def test_a_command_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
