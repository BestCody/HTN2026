import pytest

from moira.wake_word import command_follows_wake_word, wake_word_detected


@pytest.mark.parametrize(
    "transcript",
    (
        "Hey Charlie, pick up the red block.",
        "CHARLIE",
        "Okay, Charlie please move the block",
    ),
)
def test_exact_charlie_wake_word_is_detected(transcript: str) -> None:
    assert wake_word_detected(transcript, "charlie")


@pytest.mark.parametrize("transcript", ("tomorrow", "more", "hey robot", ""))
def test_partial_or_missing_wake_word_is_rejected(transcript: str) -> None:
    assert not wake_word_detected(transcript, "charlie")


def test_activation_can_include_command_or_request_followup() -> None:
    assert command_follows_wake_word("Hey Charlie, pick up the block", "charlie")
    assert not command_follows_wake_word("Hey Charlie", "charlie")
    assert not command_follows_wake_word("Charlie, please", "charlie")
