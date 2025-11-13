# tests/conftest.py
import json
import sys
from pathlib import Path

import pytest

# Add the src directory to sys.path for module resolution
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def sample_files(tmp_path):
    network = {
        "nodes": [
            {"id": 0, "x": 0, "y": 0},
            {"id": 1, "x": 2, "y": 0},
        ],
        "edges": [
            {"source": 0, "target": 1, "capacity": 10, "freeflow_travel_time": 60},
            {"source": 0, "target": 1, "capacity": 5, "freeflow_travel_time": 30},
        ],
    }
    demand = {
        "zones": ["Z1", "Z2"],
        "period": {"start": "07:00", "end": "08:00"},
        "matrix": [
            [0, 10],
            [0, 0],
        ],
    }
    network_path = tmp_path / "network.json"
    demand_path = tmp_path / "demand.json"
    network_path.write_text(json.dumps(network))
    demand_path.write_text(json.dumps(demand))
    return network_path, demand_path