from src.config.settings import (
    PROJECT_ROOT,
    DATA_DIR,
    RAW_DATA_DIR,
)


def test_project_root_exists():
    assert PROJECT_ROOT.exists()


def test_data_directory_path():
    assert DATA_DIR.name == "data"


def test_raw_data_directory_path():
    assert RAW_DATA_DIR.name == "raw"