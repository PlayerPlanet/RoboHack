import sys
from pathlib import Path
import pytest


# Make repository root importable during tests and provide a fixture
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return the repository root as a Path for tests that need it."""
    return REPO_ROOT
