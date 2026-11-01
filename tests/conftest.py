"""The test suite is offline: accidental real dataset downloads fail immediately."""
import socket
import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError('Unit tests must not access the network or download datasets')
    monkeypatch.setattr(socket, 'create_connection', blocked)
