from __future__ import annotations

import argparse
import contextlib
import mimetypes
import os
import re
import socket
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import build_run_viewer


class RangeRequestHandler(SimpleHTTPRequestHandler):
    """Static file handler with byte-range support for browser audio seeking."""

    def send_head(self):
        path = Path(self.translate_path(self.path))
        if path.is_dir():
            for index in ("index.html", "index.htm"):
                index_path = path / index
                if index_path.exists():
                    path = index_path
                    break
            else:
                return self.list_directory(str(path))

        if not path.exists():
            self.send_error(404, "File not found")
            return None

        ctype = self.guess_type(str(path))
        file_size = path.stat().st_size
        range_header = self.headers.get("Range")

        if range_header:
            match = re.match(r"bytes=(\d*)-(\d*)$", range_header.strip())
            if match:
                start_text, end_text = match.groups()
                start = int(start_text) if start_text else 0
                end = int(end_text) if end_text else file_size - 1
                end = min(end, file_size - 1)
                if start <= end and start < file_size:
                    file_obj = open(path, "rb")
                    file_obj.seek(start)
                    self.range = (start, end)
                    self.send_response(206)
                    self.send_header("Content-type", ctype)
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                    self.send_header("Content-Length", str(end - start + 1))
                    self.end_headers()
                    return file_obj

            self.send_error(416, "Requested Range Not Satisfiable")
            return None

        self.range = None
        file_obj = open(path, "rb")
        self.send_response(200)
        self.send_header("Content-type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(file_size))
        self.end_headers()
        return file_obj

    def copyfile(self, source, outputfile):
        byte_range = getattr(self, "range", None)
        if not byte_range:
            return super().copyfile(source, outputfile)

        start, end = byte_range
        remaining = end - start + 1
        while remaining > 0:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        return super().end_headers()


def find_free_port(preferred_port: int) -> int:
    for port in [preferred_port, *range(preferred_port + 1, preferred_port + 50)]:
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No free port found near {preferred_port}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve run_full viewer with audio range support.")
    parser.add_argument("--root", default=".", help="Workspace root containing run_full* directories.")
    parser.add_argument("--out", default=".run_viewer/index.html", help="Generated viewer HTML path.")
    parser.add_argument("--port", type=int, default=8765, help="Preferred local server port.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_path = Path(args.out).expanduser()
    if not out_path.is_absolute():
        out_path = root / out_path

    build_run_viewer.build_html(root, out_path)
    port = find_free_port(args.port)
    mimetypes.add_type("audio/wav", ".wav")
    mimetypes.add_type("audio/mpeg", ".mp3")
    mimetypes.add_type("audio/flac", ".flac")
    mimetypes.add_type("audio/mp4", ".m4a")

    handler = partial(RangeRequestHandler, directory=str(root))
    server = ThreadingHTTPServer((args.host, port), handler)
    rel_index = os.path.relpath(out_path, root).replace(os.sep, "/")
    url = f"http://{args.host}:{port}/{rel_index}"
    print(f"Serving run viewer: {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
