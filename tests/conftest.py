# Taken from https://doc.pytest.org/en/latest/example/simple.html?highlight=skipping

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runlocal", action="store_true", default=False, help="run local only tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "local: mark test as local only")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runlocal"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_local = pytest.mark.skip(reason="need --runlocal option to run")
    for item in items:
        if "local" in item.keywords:
            item.add_marker(skip_local)
