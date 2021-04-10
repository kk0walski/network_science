import random
import numpy as np
import networkx as nx
import multiprocessing as mp


def get_state(nodes_number, physical_network, hidden_network, 
        hidden_transition_prob, physical_transition_prob,
        infectivity, factor, _lambda, level_limit, initial_infecteds=None,
        rho=None):
    
    layer_network = {}
    layer_network['hidden_transition_prob'] = hidden_transition_prob
    layer_network['physical_transition_prob'] = physical_transition_prob
    layer_network['infectivity'] = infectivity
    layer_network['factor'] = factor
    layer_network['lambda'] = _lambda
    layer_network['limit'] = level_limit
    layer_network['nodes'] = {}
    physical_dict = nx.to_dict_of_lists(physical_layer)
    hidden_dict = nx.to_dict_of_lists(hidden_layer)
    for node in range(nodes_number):
        layer_network['nodes'][node] = {'hidden_status': 'U', 'hidden': hidden_dict[node],
                            'physical_status': 'S', 'physical': physical_dict[node]}

    if initial_infecteds is None:
        if rho is None:
            initial_number = 1
        else:
            initial_number = int(round(nodes_number*rho))
        initial_infecteds = random.sample(range(nodes_number), initial_number)
    
    for i in initial_infecteds:
        if i < nodes_number:
            layer_network['nodes'][node]['physical_status'] = 'I'
            layer_network['nodes'][node]['hidden_status'] = 'A'

    return layer_network

def filter_node_rec(level, node):
    if state['nodes'][node]["hidden_status"] == "A":
        return True
    elif level == state['limit']:
        return False
    else:
        boolean_status = list(
            map(
                lambda neighbour: filter_node_rec(level + 1, neighbour),
                state['nodes'][node]['hidden'],
            )
        )
        if len(boolean_status) == 0:
            return False
        else:
            return max(boolean_status)

def filter_node(node):
    print("FILTROWANIE!!!")
    if state['nodes'][node]['hidden_status'] == 'A':
        return node
    elif filter_node_rec(0, node):
        return node
    else:
        return -1


def run_chain(nodes_number):
    status_counts = {"S": 0, "I": 0, "A": 0, "U": 0}
    infected_nodes = [k for k, v in state['nodes'].items() if v['physical_status'] == 'I']

    if len(infected_nodes) > 0:
        aware_nodes = [k for k, v in state['nodes'].items() if v['hidden_status'] == 'A']
        unaware_nodes = list(set(range(nodes_number)) - set(aware_nodes))

        print("WEZLY CZESCIOWO WYFILTROWANE!")
        with mp.Pool(processes=(10)) as pool:
            filtered_unaware = []
            for node in pool.imap_unordered(
                    filter_node, unaware_nodes, chunksize=5
                ):
                if node >= 0:
                    filtered_unaware.append(node)

            rest_number = nodes_number - len(filtered_unaware) - len(aware_nodes)
            exam_nodes = list(set(filtered_unaware) + set(aware_nodes))
        
    return exam_nodes

def random_edge(graph):
    edges = list(graph.edges)
    nonedges = list(nx.non_edges(graph))

    # random edge choice
    chosen_edge = random.choice(edges)
    chosen_nonedge = random.choice([x for x in nonedges if chosen_edge[0] == x[0]])
    graph.add_edge(chosen_nonedge[0], chosen_nonedge[1])
    return graph

if __name__ == "__main__":
    nodes_number = 1000
    physical_layer = nx.barabasi_albert_graph(nodes_number, 10)
    hidden_layer = physical_layer.copy()
    for i in range(400):
        hidden_layer = random_edge(hidden_layer)
    _lambda = 1
    rho = 0.2
    hidden_transition_prob = 0.6
    physical_transition_prob = 0.4
    factor = 1e-3
    level_limit = 2
    infectivity = 0.1

    manager = mp.Manager()
    temp_state = get_state(nodes_number,
        physical_layer,
        hidden_layer,
        hidden_transition_prob,
        physical_transition_prob,
        infectivity,
        factor,
        _lambda,
        level_limit,
        rho=rho)

    state = manager.dict(temp_state)
    print("STAN GOTOWY!")
    exam_nodes = run_chain(nodes_number)
    print(exam_nodes)