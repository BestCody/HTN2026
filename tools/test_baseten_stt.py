"""Send one local audio file through the deployed MoIRA STT specialist."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from moira.cloud import BasetenComponent, BasetenEndpoint
from moira.contracts import decode_physical_response
from moira.physical import SpeechInput


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path, help="WAV, MP3, or M4A command recording")
    args = parser.parse_args()

    load_dotenv()
    model_id = os.environ.get("BASETEN_STT_MODEL_ID")
    if not model_id:
        raise SystemExit("Set BASETEN_STT_MODEL_ID in .env after deployment")
    environment = os.environ.get("BASETEN_STT_ENVIRONMENT", "development")

    audio_path = args.audio.resolve()
    if not audio_path.is_file():
        raise SystemExit(f"Audio file does not exist: {audio_path}")

    component = BasetenComponent(
        BasetenEndpoint(model_id, environment=environment),
        decode=decode_physical_response,
    )
    transcript = component.run(SpeechInput(audio_path.read_bytes()))
    print(transcript)


if __name__ == "__main__":
    main()

