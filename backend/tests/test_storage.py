from io import BytesIO

import pytest

from konopro_backend.storage import LocalAudioStorage, StorageValidationError


def test_save_audio_generates_safe_key_and_metadata(tmp_path):
    storage = LocalAudioStorage(tmp_path)
    payload = b"abc123"

    stored = storage.save(
        BytesIO(payload),
        original_filename="../unsafe/song.wav",
        content_type="audio/wav",
        max_bytes=100,
    )

    assert ".." not in stored.storage_key
    assert "unsafe" not in stored.storage_key
    assert stored.storage_key.endswith(".wav")
    assert stored.size_bytes == len(payload)
    assert stored.sha256 == "6ca13d52ca70c883e0f0bb101e425a89e8624de51db2d2392593af6a84118090"
    assert storage.path_for(stored.storage_key).exists()
    assert storage.open(stored.storage_key).read() == payload


def test_delete_is_idempotent(tmp_path):
    storage = LocalAudioStorage(tmp_path)
    stored = storage.save(BytesIO(b"audio"), "song.mp3", "audio/mpeg", max_bytes=100)

    assert storage.exists(stored.storage_key)
    storage.delete(stored.storage_key)
    storage.delete(stored.storage_key)

    assert not storage.exists(stored.storage_key)


def test_rejects_unsupported_content_type(tmp_path):
    storage = LocalAudioStorage(tmp_path)

    with pytest.raises(StorageValidationError, match="Unsupported audio content type"):
        storage.save(BytesIO(b"not audio"), "song.txt", "text/plain", max_bytes=100)


def test_rejects_files_above_max_size(tmp_path):
    storage = LocalAudioStorage(tmp_path)

    with pytest.raises(StorageValidationError, match="exceeds"):
        storage.save(BytesIO(b"too large"), "song.wav", "audio/wav", max_bytes=3)


def test_rejects_storage_key_path_traversal(tmp_path):
    storage = LocalAudioStorage(tmp_path)

    with pytest.raises(StorageValidationError):
        storage.path_for("../outside.wav")
