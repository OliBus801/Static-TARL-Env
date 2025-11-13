# tests/conftest.py
import sys
from pathlib import Path
import warnings

# Add the src directory to sys.path for module resolution
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Configure pytest to ignore specific warnings
def pytest_configure():
    warnings.filterwarnings(
        "ignore",
        message=".*torch_geometric\\.distributed.*",
        category=DeprecationWarning,
    )