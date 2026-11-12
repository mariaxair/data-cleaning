"""Tests must never contact a provider, even if .env contains credentials."""

import pytest
import requests


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network access is forbidden during tests.")
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
