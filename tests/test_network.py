import json
import networkx as nx
import numpy as np
import pytest


def test_network(tmp_path):
    from simulation.network import TapScenario
    
    # Prepare temporary test JSONs
    network = {
        "nodes": [
            {"id": 0, "x": 0, "y": 0},
            {"id": 1, "x": 2, "y": 0}
        ],
        "edges": [
            {"source": 0, "target": 1, "capacity": 10, "freeflow_travel_time": 60},
            {"source": 0, "target": 1, "capacity": 5,  "freeflow_travel_time": 30}
        ]
        }
    demand = {
        "zones": ["Z1","Z2"],
        "period": {"start": "07:00", "end": "08:00"},
        "matrix": [
            [0,   10],
            [0,  0]
        ]
    }
    network_path = tmp_path / "network.json"
    demand_path = tmp_path / "demand.json"
    network_path.write_text(json.dumps(network))
    demand_path.write_text(json.dumps(demand))

    # Create the TapScenario object
    G = TapScenario()
    G.load_network_from_json(network_path)
    G.load_demand_from_json(demand_path)

    # Basic network checks ------------------------
    assert isinstance(G.graph, nx.MultiDiGraph), "the graph should be an instance of MultiDiGraph."
    assert G.graph.is_directed(), "The graph should be directed."

    # Check nodes
    assert set(G.graph.nodes) == {0, 1}, "Nodes 0 and 1 should be present."

    # Check edges
    assert G.graph.number_of_edges(0, 1) == 2, "There should be two parallel edges from 0 to 1."

    # Check edge attributes exist for all edges
    for u, v, attrs in G.graph.edges(data=True):
        assert "capacity" in attrs, f"Edge ({u},{v}) missing 'capacity'."
        assert "freeflow_travel_time" in attrs, f"Edge ({u},{v}) missing 'freeflow_travel_time'."

    # Basic demand checks ------------------------
    assert isinstance(G.od, np.ndarray), "OD should be a numpy array."
    assert np.array_equal(G.od, np.array([[0, 10], [0, 0]])), "Demand matrix does not match expected."
    assert G.total_agents == 10, "Incorrect total number of agents. Expected 10."

    # Basic flow consistency check ------------------------
    good_assignment = [10, 0]  # All flow on first edge
    bad_assignment_1 = [5, 0]    # Not enough flow to satisfy demand
    bad_assignment_2 = [10, 5]   # Too much flow
    assert G.check_flow_consistency(good_assignment) is True, "Good assignment should pass flow consistency."
    assert G.check_flow_consistency(bad_assignment_1) is False, "Bad assignment 1 should fail flow consistency. Not enough flow."
    assert G.check_flow_consistency(bad_assignment_2) is False, "Bad assignment 2 should fail flow consistency. Too much flow."