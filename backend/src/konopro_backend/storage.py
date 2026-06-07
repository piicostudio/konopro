from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from uuid import uuid4


class StorageValidationError(ValueError):
    """Raised when an upload cannot be stored safely."""


@dataclass(frozen=True)
class StoredAudio:
    storage_key: str
    size_bytes: int
    sha256: str
    content_type: str
    path: Path


CONTENT_TYPE_EXTENSIONS = {
    "audio/aac": ".aac",
    "audio/flac": ".flac",
    "audio/m4a": ".m4a",
    "audio/mp4": ".m4a",
    "audio/mpeg": ".mp3",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/wave": ".wav",
    "audio/x-m4a": ".m4a",
    "audio/x-wav": ".wav",
}

ALLOWED_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}
CHUNK_SIZE = 1024 * 1024


class LocalAudioStorage:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        fileobj,
        original_filename: str,
        content_type: str,
        max_bytes: int,
    ) -> StoredAudio:
        extension = self._extension_for(original_filename, content_type)
        storage_key = f"audio/{uuid4().hex}{extension}"
        path = self.path_for(storage_key)
        path.parent.mkdir(parents=True, exist_ok=True)

        hasher = hashlib.sha256()
        size_bytes = 0
        with path.open("wb") as output:
            while True:
                chunk = fileobj.read(CHUNK_SIZE)
                if not chunk:
                    break
                size_bytes += len(chunk)
                if size_bytes > max_bytes:
                    output.close()
                    path.unlink(missing_ok=True)
                    raise StorageValidationError(
                        f"Upload exceeds maximum size of {max_bytes} bytes"
                    )
                hasher.update(chunk)
                output.write(chunk)

        return StoredAudio(
            storage_key=storage_key,
            size_bytes=size_bytes,
            sha256=hasher.hexdigest(),
            content_type=content_type,
            path=path,
        )

    def open(self, storage_key: str):
        return self.path_for(storage_key).open("rb")

    def exists(self, storage_key: str) -> bool:
        return self.path_for(storage_key).exists()

    def delete(self, storage_key: str) -> None:
        path = self.path_for(storage_key)
        path.unlink(missing_ok=True)
        self._remove_empty_parents(path.parent)

    def path_for(self, storage_key: str) -> Path:
        posix_path = PurePosixPath(storage_key)
        if posix_path.is_absolute() or ".." in posix_path.parts:
            raise StorageValidationError("Invalid storage key")
        path = (self.root / Path(*posix_path.parts)).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise StorageValidationError("Storage key escapes storage root") from exc
        return path

    def _extension_for(self, original_filename: str, content_type: str) -> str:
        normalized_content_type = (content_type or "").split(";")[0].strip().lower()
        extension = CONTENT_TYPE_EXTENSIONS.get(normalized_content_type)
        if extension is None:
            raise StorageValidationError(f"Unsupported audio content type: {content_type}")

        filename_extension = Path(original_filename or "").suffix.lower()
        if filename_extension and filename_extension not in ALLOWED_EXTENSIONS:
            raise StorageValidationError(f"Unsupported audio extension: {filename_extension}")
        return filename_extension or extension

    def _remove_empty_parents(self, path: Path) -> None:
        current = path
        while current != self.root and current.exists():
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent


def copy_storage_object(source: LocalAudioStorage, target: LocalAudioStorage, storage_key: str) -> None:
    target_path = target.path_for(storage_key)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source.path_for(storage_key), target_path)
