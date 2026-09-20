"""Serve pinned Faster-Whisper and Kokoro specialists from the RTX computer."""

from __future__ import annotations

import argparse
import base64
import hmac
import json
import os
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from moira.brand import DISPLAY_NAME  # noqa: E402
from moira.edge_components import OpenCVCameraSource  # noqa: E402
from moira.rtx_voice import FasterWhisperRTX, KokoroRTX  # noqa: E402

MAX_REQUEST_BYTES = 12 * 1024 * 1024


class VoiceApplication:
    def __init__(
        self,
        cache_dir: Path,
        *,
        camera: int | str | None = None,
        camera_id: str = "co6-usb",
    ) -> None:
        self.stt = FasterWhisperRTX(cache_dir=cache_dir / "whisper")
        self.tts = KokoroRTX(cache_dir=cache_dir / "kokoro")
        self.camera = (
            OpenCVCameraSource(camera, camera_id=camera_id) if camera is not None else None
        )

    def predict(self, path: str, request: dict[str, Any]) -> dict[str, Any]:
        if path == "/v1/stt":
            return self.stt.predict(request)
        if path == "/v1/tts":
            return self.tts.predict(request)
        raise KeyError(path)

    def capture(self) -> dict[str, Any]:
        if self.camera is None:
            raise RuntimeError("Laptop camera service was not enabled")
        frame = self.camera.capture()
        return {
            "camera_id": frame.camera_id,
            "captured_at": frame.captured_at,
            "media_type": frame.media_type,
            "data_base64": base64.b64encode(frame.data).decode("ascii"),
        }

    def close(self) -> None:
        if self.camera is not None:
            self.camera.close()


def handler_type(application: VoiceApplication, token: str | None) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "MoIRARTXVoice/1"

        def _json(self, status: HTTPStatus, value: Any) -> None:
            encoded = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _authorized(self) -> bool:
            if token is None:
                return True
            supplied = self.headers.get("Authorization", "")
            return hmac.compare_digest(supplied, f"Bearer {token}")

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/v1/camera":
                if not self._authorized():
                    self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                    return
                try:
                    self._json(HTTPStatus.OK, application.capture())
                except Exception as exc:
                    self._json(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {"error": type(exc).__name__, "message": str(exc)},
                    )
                return
            if self.path != "/health":
                self._json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
                return
            self._json(
                HTTPStatus.OK,
                {
                    "status": "ready",
                    "stt_loaded": application.stt._model is not None,
                    "tts_loaded": application.tts._pipeline is not None,
                    "camera_enabled": application.camera is not None,
                },
            )

        def do_POST(self) -> None:  # noqa: N802
            if not self._authorized():
                self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return
            if self.path not in {"/v1/stt", "/v1/tts"}:
                self._json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > MAX_REQUEST_BYTES:
                    raise ValueError("request body size is invalid")
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError("request body must be a JSON object")
                result = application.predict(self.path, request)
            except (TypeError, ValueError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            except Exception as exc:
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": type(exc).__name__, "message": str(exc)},
                )
                return
            self._json(HTTPStatus.OK, result)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"{self.address_string()} {format % args}")

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--camera",
        help="Optional OpenCV camera index or path to expose over HTTP; omit to disable",
    )
    parser.add_argument("--camera-id", default="co6-usb")
    default_cache = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "moira" / "models"
    parser.add_argument("--cache-dir", type=Path, default=default_cache)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=False)
    token = os.environ.get("MOIRA_LAN_TOKEN")
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not token:
        parser.error("MOIRA_LAN_TOKEN is required when serving beyond loopback")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    camera: int | str | None = args.camera
    if isinstance(camera, str) and camera.isdigit():
        camera = int(camera)
    application = VoiceApplication(
        args.cache_dir.resolve(),
        camera=camera,
        camera_id=args.camera_id,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler_type(application, token))
    print(f"{DISPLAY_NAME} RTX voice service listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        application.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
