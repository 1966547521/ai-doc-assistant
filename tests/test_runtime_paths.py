from src.runtime_paths import CACHE_DIR, CHROMA_DIR, LOG_DIR, SESSIONS_FILE


def test_runtime_artifacts_are_grouped_under_project_data_directory():
    paths = (CACHE_DIR, CHROMA_DIR, LOG_DIR, SESSIONS_FILE)

    assert all(path.parent.name == "data" for path in paths)
