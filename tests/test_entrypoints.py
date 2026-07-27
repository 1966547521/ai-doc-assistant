"""Tests for command-line application entry points."""


def test_api_server_uses_environment_host_and_port(monkeypatch):
    from api import server

    captured = {}
    monkeypatch.setenv("API_HOST", "127.0.0.1")
    monkeypatch.setenv("API_PORT", "8123")
    monkeypatch.setattr(
        "uvicorn.run",
        lambda app, host, port, reload: captured.update(
            app=app, host=host, port=port, reload=reload
        ),
    )

    server.main()

    assert captured == {
        "app": "api.main:app",
        "host": "127.0.0.1",
        "port": 8123,
        "reload": False,
    }
