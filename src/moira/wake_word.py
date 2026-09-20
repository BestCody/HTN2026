"""Strict text-level wake-word detection for locally transcribed microphone audio."""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+")


def normalized_words(text: str) -> tuple[str, ...]:
    if not isinstance(text, str):
        raise TypeError("wake-word transcript must be text")
    return tuple(_WORD.findall(text.casefold()))


def wake_word_detected(transcript: str, wake_word: str) -> bool:
    wake = normalized_words(wake_word)
    if len(wake) != 1:
        raise ValueError("wake word must contain exactly one word")
    return wake[0] in normalized_words(transcript)


def command_follows_wake_word(transcript: str, wake_word: str) -> bool:
    """Return whether the activation segment also contains a command."""

    wake = normalized_words(wake_word)
    if len(wake) != 1:
        raise ValueError("wake word must contain exactly one word")
    words = normalized_words(transcript)
    try:
        index = words.index(wake[0])
    except ValueError:
        return False
    trailing = words[index + 1 :]
    # Polite fillers alone do not count as a robot instruction.
    return any(word not in {"hey", "hi", "okay", "ok", "please"} for word in trailing)
