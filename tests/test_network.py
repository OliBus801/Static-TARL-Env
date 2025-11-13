import numpy as np
import pytest
from torch_geometric.data import Data

def test_load_network_and_demand(sample_files):
    from simulation.network import TapScenario

    network_path, demand_path = sample_files
    scenario = TapScenario()
    graph = scenario.load_network_from_json(network_path)
    scenario.load_demand_from_json(demand_path)

    assert isinstance(graph, Data)
    assert graph.num_nodes == 2
    assert graph.num_edges == 2
    assert pytest.approx(graph.capacity.tolist()) == [10.0, 5.0]
    assert pytest.approx(graph.freeflow_travel_time.tolist()) == [60.0, 30.0]
    assert pytest.approx(graph.length.tolist()) == [2.0, 2.0]
    assert graph.edge_attr.shape == (2, 3)

    assert isinstance(scenario.od, np.ndarray)
    assert np.array_equal(scenario.od, np.array([[0, 10], [0, 0]]))
    assert scenario.total_agents == 10


def test_flow_consistency(sample_files):
    from simulation.network import TapScenario

    network_path, demand_path = sample_files
    scenario = TapScenario()
    scenario.load_network_from_json(network_path)
    scenario.load_demand_from_json(demand_path)

    assert scenario.check_flow_consistency(np.array([10, 0], dtype=float)) is True
    assert scenario.check_flow_consistency(np.array([5, 0], dtype=float)) is False
    assert scenario.check_flow_consistency(np.array([10, 5], dtype=float)) is False


def test_system_cost_requires_consistent_flow(sample_files):
    from simulation.network import TapScenario

    network_path, demand_path = sample_files
    scenario = TapScenario()
    scenario.load_network_from_json(network_path)
    scenario.load_demand_from_json(demand_path)

    scenario.check_flow_consistency(np.array([10, 0], dtype=float))
    total_cost = scenario.calculate_system_cost(np.array([10, 0], dtype=float))
    assert total_cost == 690.0 # f(10) = 60*(1 + 0.15*(10/10)^4) = 69 per unit flow, total 690

    with pytest.raises(ValueError):
        scenario.calculate_system_cost(np.array([0, 0], dtype=float))
