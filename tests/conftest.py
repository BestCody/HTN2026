import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="Run pretrained model E2E tests (downloads MiniLM and SmolLM2-1.7B)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-e2e"):
        return
    skip = pytest.mark.skip(reason="Pass --run-e2e to download and run pretrained models")
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip)
