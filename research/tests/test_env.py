from __future__ import annotations

import os

from konopro_research.env import load_research_env


def test_load_research_env_sets_missing_values_without_overwriting(tmp_path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "AUDD_API_TOKEN='from-file'",
                'ACRCLOUD_HOST="host-from-file"',
                "EXISTING_VALUE=file-value",
                "# ignored comment",
            ]
        )
    )
    monkeypatch.delenv("AUDD_API_TOKEN", raising=False)
    monkeypatch.delenv("ACRCLOUD_HOST", raising=False)
    monkeypatch.setenv("EXISTING_VALUE", "already-set")

    load_research_env(env_path)

    assert os.environ["AUDD_API_TOKEN"] == "from-file"
    assert os.environ["ACRCLOUD_HOST"] == "host-from-file"
    assert os.environ["EXISTING_VALUE"] == "already-set"
