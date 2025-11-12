import json
import networkx as nx
import numpy as np
from networkx.readwrite import json_graph
import yaml
from pathlib import Path

# Utils --------------------
def load_params(filepath: str) -> dict:
    with open(filepath, "r") as f:
        params = yaml.safe_load(f)
        return params

# Load parameters from the YAML file --------------------
config_path = Path(__file__).resolve().parent.parent / "data" / "configs" / "params.yaml"
params = load_params(config_path)

# Parameters
ALPHA = params.get("alpha", 0.15)
BETA = params.get("beta", 4.0)

class TapScenario:

    def __init__(self):
        self.graph = None
        self.od = None
        self.total_agents = 0

    def load_network_from_json(self, filepath: str) -> nx.MultiDiGraph:
        """
        Load a network from a JSON file into a MultiDiGraph.

        Args:
            filepath (str): Path to the JSON file.

        Returns:
            nx.MultiDiGraph: The loaded graph.
        """

        with open(filepath, "r") as f:
            data = json.load(f)

            self.graph = json_graph.node_link_graph(data, edges='edges', directed=True, multigraph=True)
            num_nodes = self.graph.number_of_nodes()
            num_edges = self.graph.number_of_edges()
            print(f"✅ Successfully loaded network from JSON file!")
            print(f"🌐 Network Info: {num_nodes} nodes, {num_edges} edges.")
            
    def load_demand_from_json(self, filepath: str) -> dict:
        """
        Load demand from a JSON file into a dictionary.

        Args:
            filepath (str): Path to the JSON file.

        Returns:
            dict: The loaded demand.
        """
        with open(filepath, "r") as f:
            demand_data = json.load(f)
            self.od = np.array(demand_data['matrix'])
            self.total_agents = np.sum(self.od)
            print(f"✅ Successfully loaded demand from JSON file!")
            print(f"👥 Demand Info: Total agents = {self.total_agents}.")

    def check_flow_consistency(self, assignment: list) -> bool:
        """
        Verify if the traffic assignment candidate solution verifies the flow consistency assumptions.
        It calculates the "in" flow for each edge minus the demand "destination" for the destination node and makes sure that it's equal to the "out" flow.

        Args:
            assignment (list): A list of traffic flow (number of vehicles) on each edge.

        Returns:
            bool: True if the flow consistency is verified, False otherwise.
        """
        if len(assignment) != self.graph.number_of_edges():
            raise ValueError(f"You must provide an assignment for exactly the number of edges in the graph. (Expected {self.graph.number_of_edges()}, got {len(assignment)})")
        
        flows = np.zeros(self.graph.number_of_edges())

        # TODO : Add a verification that the sum of assignment is equal to total_agents
        # It's not as easy as it seems because flows move across multiple edges.

        # Add origin demand to flows
        for i, demand in enumerate(np.sum(self.od, axis=1)):  # Add origin demand to flows
            flows[i] += demand
        for j, demand in enumerate(np.sum(self.od, axis=0)):  # Subtract destination demand from flows
            flows[j] -= demand


        for (u, v, _), flow in zip(self.graph.edges(data=True), assignment):
            # Out flow for node u
            flows[u] -= flow
            # In flow for node v
            flows[v] += flow
        
        if not np.all(flows == 0):
            inconsistent_nodes = np.where(flows != 0)[0]
            print(f"❌ Flow consistency check failed for nodes: {inconsistent_nodes}, with flow values: {flows[inconsistent_nodes]}")
            return False
        else:
            print(f"✅ Flow consistency check passed.")
            return True
    
    def calculate_system_cost(self, assignment: list) -> int:
        """
        Calculate total system cost from a traffic assignment candidate solution.
        This function uses the BPR non-linear function to calculate cost on each link.
        Since the problem is static, we assume all agents have equal travel time.

        Args:
            assignment (list): A list of traffic flow (number of vehicles) on each edge.

        Returns:
            int: The total system cost.
        """
        if not self.check_flow_consistency(assignment):
            raise ValueError("The provided assignment does not verify flow consistency.")
        
        total_cost = 0

        for i, (origin, destination, data) in enumerate(self.graph.edges(data=True)):
            print(f"---------- Link #{i} ----------")
            print(f"Calculating travel time on link {origin} -> {destination} with capacity {data['capacity']} and freeflow_travel_time {data['freeflow_travel_time']}")
            cost = data['freeflow_travel_time']*(1 + ALPHA * (assignment[i] / data['capacity'])**BETA)
            print(f"Coût : {cost} sec. * {assignment[i]} agents = {cost*assignment[i]}")

            total_cost += cost*assignment[i]

        print(f"Coût total du système : {total_cost}")
        return total_cost        



