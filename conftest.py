import pytest


def pytest_addoption(parser):
    parser.addoption("--rungpu", action="store_true", default=False, help="run gpu tests")
    parser.addoption("--perf-tests", action="store_true", default=False, help="run performance tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--rungpu"):
        return
    skip_gpu = pytest.mark.skip(reason="need --rungpu option to run")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
