from pathlib import Path

from src.ingestion.visual_assets import (
    VisualAssetManager,
)


def test_asset_manager_creates_directory(
    tmp_path: Path,
):
    output_dir = (
        tmp_path / "images"
    )

    manager = VisualAssetManager(
        output_dir=output_dir
    )

    assert output_dir.exists()