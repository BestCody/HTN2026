"""Send one local audio file through the configured RTX LAN STT specialist."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from moira.cloud import JsonComponent, JsonEndpoint, JsonHttpClient
from moira.contracts import decode_physical_response
from moira.physical import SpeechInput


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path, help="WAV, MP3, or M4A command recording")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=300.0,
        help="Maximum LAN inference time (default: 300)",
    )
    args = parser.parse_args()

    load_dotenv()
    url = os.environ.get("MOIRA_STT_URL")
    if not url:
        raise SystemExit("Set MOIRA_STT_URL in .env")

    audio_path = args.audio.resolve()
    if not audio_path.is_file():
        raise SystemExit(f"Audio file does not exist: {audio_path}")

    component = JsonComponent(
        JsonEndpoint(
            url,
            token=os.environ.get("MOIRA_LAN_TOKEN"),
            http=JsonHttpClient(
                timeout_seconds=args.timeout_seconds,
                attempts=1,
                allow_private_http=True,
            ),
        ),
        decode=decode_physical_response,
    )
    transcript = component.run(SpeechInput(audio_path.read_bytes()))
    print(transcript)


if __name__ == "__main__":
    main()
