from konopro_backend.config import BackendSettings


def test_fingerprint_settings_can_be_overridden(monkeypatch, tmp_path):
    monkeypatch.setenv("KONOPRO_FINGERPRINT_PROVIDER", "audd")
    monkeypatch.setenv("KONOPRO_FINGERPRINT_WINDOW_S", "30")
    monkeypatch.setenv("KONOPRO_PROCESSING_ROOT", str(tmp_path / "processing"))

    settings = BackendSettings()

    assert settings.fingerprint_provider == "audd"
    assert settings.fingerprint_window_s == 30.0
    assert settings.processing_path == (tmp_path / "processing").resolve()


def test_backend_can_import_research_segmentation_package():
    from konopro_research.session_segmentation import segment_long_recording

    assert callable(segment_long_recording)


def test_local_environment_allows_all_cors_origins_by_default():
    settings = BackendSettings()

    assert settings.environment == "local"
    assert settings.cors_origins == ["*"]


def test_explicit_cors_origins_are_preserved_in_local():
    settings = BackendSettings(cors_allow_origins="https://example.com,http://localhost:5173")

    assert settings.cors_origins == ["https://example.com", "http://localhost:5173"]


def test_non_local_environment_uses_explicit_cors_origins():
    settings = BackendSettings(environment="production")

    assert "http://localhost:5173" in settings.cors_origins
    assert settings.cors_origins != ["*"]
