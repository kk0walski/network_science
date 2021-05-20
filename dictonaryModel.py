from os import stat
import random
import itertools
import numpy as np
import networkx as nx
from collections import Counter, defaultdict


class MarkovModel:

    def __init__(
        self,
        nodes_number,
        network,
        initial_infecteds=None,
        rho=None,
        tmin=0,
        tmax=10,
    ):

        self.nodes_number = nodes_number
        self.network = network
        self.tmin = tmin
        self.tmax = tmax
        self.m = 0
        self.a = [value['degree'] for value in network.values()]
        self.hidden_status_original = np.array(['U' for _ in range(nodes_number)])
        self.physical_status_original = np.array(['S' for _ in range(nodes_number)])

        if initial_infecteds is None:
            if rho is None:
                self.initial_number = 1
            else:
                self.initial_number = int(round(len(list(network.keys())) * rho))
            random.seed(100)
            self.initial_infecteds = random.sample(
                list(network.keys()), self.initial_number
            )
        else:
            self.initial_infecteds = [infected for infected in initial_infecteds if infected < nodes_number and infected > 0]

        self.init_S = self.nodes_number - len(self.initial_infecteds)
        self.init_A =  len(self.initial_infecteds)
        for node in self.initial_infecteds:
            self.hidden_status_original[node] = "A"
            self.physical_status_original[node] = "I"

    def rnd(self):
        exp = np.random.randint(-11, -1)
        significand = 0.9 * np.random.random() + 0.1
        return significand * 10 ** exp

    def init_simulation(self):
        self.tmin = 0
        self.times = [self.tmin]
        self.hidden_status = (self.hidden_status_original == "A")
        self.hidden_status_copy = self.hidden_status[:]
        self.physical_status = (self.physical_status_original == "I")
        self.physical_status_copy = self.physical_status[:]
        self.S_t = [self.init_S]
        self.U_t = [self.init_S]
        self.A_t = [self.init_A]
        self.I_t = [self.init_A]

    def set_infectivity(self, infectivity):
        self.infectivity_unaware = infectivity
        self.infectivity_aware = self.factor * infectivity

    def set_level(self, level):
        self.level_limit = level

    def set_lambda(self, _lambda):
        self._lambda = _lambda

    def set_media(self, m):
        self.m = m

    def set_physical_trans_prob(self, prob):
        self.physical_transition_prob = prob
        self.probability_AI = 1 - self.physical_transition_prob

    def set_hidden_trans_prob(self, prob):
        self.hidden_transition_prob = prob
        self.probability_A = (
            1 - self.physical_transition_prob * self.hidden_transition_prob + self.physical_transition_prob * self.hidden_transition_prob*self.m
        )

    def set_factor(self, factor):
        self.factor = factor

    def markov_chain_AI(self):
        change_hidden = np.random.choice(
            ["U", "A"],
            p=[self.hidden_transition_prob, (1 - self.hidden_transition_prob)],
        )
        if change_hidden == "U":
            change_hidden = np.random.choice(
            ["U", "A"],
            p=[(1 - self.m), self.m],
        )
        change_physical = np.random.choice(
            ["S", "I"],
            p=[self.physical_transition_prob, (1 - self.physical_transition_prob)],
        )
        return change_hidden, change_physical

    def neighbour_statuses(self, node, layer, parent=None):
        if parent:
            arr = self.network[node][layer]
            temp_neighbours = np.delete(arr, np.where(arr == parent)) 
        else:
            temp_neighbours = self.network[node][layer]
        return temp_neighbours, self.hidden_status[temp_neighbours], self.physical_status[temp_neighbours]

    def r_prob(self, level, parent, node):
        probability_hidden = 1
        if level < self.level_limit:
            for neighbour, hidden_status, physical_status in zip(*self.neighbour_statuses(node, 'hidden', parent)):
                if hidden_status:
                    if physical_status:
                        probability_hidden = probability_hidden * (
                            1 - self.a[neighbour]*self.probability_A * self._lambda
                        )
                    else:
                        prob_US = self.probability_US(level + 1, node, neighbour)
                        probability_temp = 1 - prob_US * self.hidden_transition_prob*(1 - self.m)
                        probability_hidden = probability_hidden * (
                            1 -  self.a[neighbour]*probability_temp * self._lambda
                        )
                else:
                    prob_US = self.probability_US(level + 1, node, neighbour)
                    prob_r = self.r_prob(level + 1, node, neighbour)
                    probability_temp = 1 - prob_r * prob_US*(1 - self.m)
                    probability_hidden = probability_hidden * (
                    1 -  self.a[neighbour]*probability_temp * self._lambda
                    )
                        
            return probability_hidden
        else:
            _sum = (
                1
                - len(list(self.network[node]['hidden'])) * self.rnd() * self._lambda
            )
            return _sum

    def probability_AS(self, level, parent, node):
        probability = 1
        if level < self.level_limit:
            for neighbour, hidden_status, physical_status in zip(*self.neighbour_statuses(node, 'physical', parent)):
                if hidden_status:
                    if physical_status:
                        probability = probability * (
                            1 - self.probability_AI * self.infectivity_aware
                        )
                    else:
                        prob_AS = self.probability_AS(level + 1, node, neighbour)
                        prob_US = self.probability_US(level + 1, node, neighbour)
                        probability_temp = (
                            1
                            - prob_AS
                            + self.hidden_transition_prob * ((prob_AS - prob_US) + self.m*(prob_US - prob_AS))
                        )
                        probability = probability * (
                            1 - probability_temp * self.infectivity_aware
                        )
                else:
                    prob_US = self.probability_US(level + 1, node, neighbour)
                    prob_AS = self.probability_AS(level + 1, node, neighbour)
                    prob_r = self.r_prob(level + 1, node, neighbour)
                    probability_temp = 1 - prob_AS + prob_r * ((prob_AS - prob_US) + self.m*(prob_US - prob_AS))
                    probability = probability * (
                        1 - probability_temp * self.infectivity_aware
                    )
            return probability
        else:
            _sum = (
                1
                - len(list(self.network[node]['physical']))
                * self.rnd()
                * self.infectivity_aware
            )
            return _sum

    def probability_US(self, level, parent, node):
        probability = 1
        if level < self.level_limit:
            for neighbour, hidden_status, physical_status in zip(*self.neighbour_statuses(node, 'physical', parent)):
                if hidden_status:
                    if physical_status:
                        probability = probability * (
                            1 - self.probability_AI * self.infectivity_unaware
                        )
                    else:
                        prob_AS = self.probability_AS(level + 1, node, neighbour)
                        prob_US = self.probability_US(level + 1, node, neighbour)
                        probability_temp = (
                            1
                            - prob_AS
                            + self.hidden_transition_prob * ((prob_AS - prob_US) + self.m*(prob_US - prob_AS))
                        )
                        probability = probability * (
                            1 - probability_temp * self.infectivity_unaware
                        )
                else:
                    prob_US = self.probability_US(level + 1, node, neighbour)
                    prob_AS = self.probability_AS(level + 1, node, neighbour)
                    prob_r = self.r_prob(level + 1, node, neighbour)
                    probability_temp = 1 - prob_AS + prob_r * ((prob_AS - prob_US) + self.m*(prob_US - prob_AS))
                    probability = probability * (
                        1 - probability_temp * self.infectivity_unaware
                    )
            return probability
        else:
            _sum = (
                1
                - len(list(self.network[node]['physical']))
                * self.rnd()
                * self.infectivity_unaware
            )
            return _sum


    def markov_chain_AS(self, node):
        hidden_states = [("U", self.infectivity_unaware), ("A", self.infectivity_aware)]
        status_index = np.random.choice(
            2,
            p=[self.hidden_transition_prob, (1 - self.hidden_transition_prob)],
        )
        probability = 1
        if hidden_states[status_index][0] == "U":
             status_index = np.random.choice(
            2,
            p=[(1-self.m), self.m],
            )

        chosen_infectivity = hidden_states[status_index][1]

        for neighbour, hidden_status, physical_status in zip(*self.neighbour_statuses(node, 'physical')):
            if hidden_status:
                if physical_status:
                    probability = probability * (
                        1 - self.probability_AI * chosen_infectivity
                    )
                else:
                    prob_AS = self.probability_AS(0, node, neighbour)
                    probability = probability * (1 - prob_AS * chosen_infectivity)
            else:
                prob_r = self.r_prob(0, node, neighbour)
                prob_AS = self.probability_AS(0, node, neighbour)
                prob_US = self.probability_US(0, node, neighbour)
                probability_temp = 1 - prob_AS + prob_r * (prob_AS - prob_US)
                probability = probability * (1 - probability_temp * chosen_infectivity)

        if probability <= 0:
            probability = 0
        elif probability > 1:
            probability = 1
        change_physical = np.random.choice(
            ["S", "I"], p=[probability, (1 - probability)]
        )
        return hidden_states[status_index][0], change_physical

    def markov_chain_US(self, node):
        probability_hidden = 1
        probability_physical = 1
        for neighbour, hidden_status, physical_status in zip(*self.neighbour_statuses(node, 'hidden')):
            if hidden_status:
                if physical_status:
                    probability_hidden = probability_hidden * (
                        1 - self.probability_A * self._lambda
                    )
                else:
                    prob_US = self.probability_US(0, node, neighbour)
                    probability_temp = 1 - prob_US * self.hidden_transition_prob
                    probability_hidden = probability_hidden * (
                        1 - probability_temp * self._lambda
                    )
            else:
                prob_r = self.r_prob(0, node, neighbour)
                prob_US = self.probability_US(0, node, neighbour)
                probability_temp = 1 - prob_r * prob_US
                probability_hidden = probability_hidden * (
                    1 - probability_temp * self._lambda
                )

        if probability_hidden <= 0:
            probability_hidden = 0
        elif probability_hidden > 1:
            probability_hidden = 1

        hidden_states = [("U", self.infectivity_unaware), ("A", self.infectivity_aware)]
        status_index = np.random.choice(
            2,
            p=[self.hidden_transition_prob, (1 - self.hidden_transition_prob)],
        )

        if hidden_states[status_index][0] == "U":
             status_index = np.random.choice(
            2,
            p=[(1-self.m), self.m],
            )

        chosen_infectivity = hidden_states[status_index][1]

        for neighbour, hidden_status, physical_status in zip(*self.neighbour_statuses(node, 'physical')):
            if hidden_status:
                if physical_status:
                    probability_physical = probability_physical * (
                        1 - self.probability_AI * chosen_infectivity
                    )
                else:
                    prob_AS = self.probability_AS(0, node, neighbour)
                    probability_physical = probability_physical * (
                        1 - prob_AS * chosen_infectivity
                    )

            else:
                prob_r = self.r_prob(0, node, neighbour)
                prob_AS = self.probability_AS(0, node, neighbour)
                prob_US = self.probability_US(0, node, neighbour)
                probability_temp = 1 - prob_AS + prob_r * (prob_AS - prob_US)
                probability_physical = probability_physical * (
                    1 - probability_temp * chosen_infectivity
                )

        if probability_physical <= 0:
            probability_physical = 0
        elif probability_physical > 1:
            probability_physical = 1
        change_physical = np.random.choice(
            ["S", "I"], p=[probability_physical, (1 - probability_physical)]
        )
        return hidden_states[status_index][0], change_physical

    def hidden_chain(self, node):
        if self.hidden_status[node]:
            if self.physical_status[node]:
                hidden_status, physical_status = self.markov_chain_AI()
            else:
                hidden_status, physical_status = self.markov_chain_AS(node)
        else:
            if self.physical_status[node]:
                hidden_status, physical_status = "A", "I"
            else:
                hidden_status, physical_status = self.markov_chain_US(node)

        if hidden_status == "U" and physical_status == "I":
            hidden_status = "A"

        self.physical_status_copy[node] = (physical_status == "I")
        self.hidden_status_copy[node] = (hidden_status == "A")
        return hidden_status, physical_status

    def filter_node_rec(self, level, node):
        if self.hidden_status[node]:
            return True
        elif level == self.level_limit:
            return False
        else:
            boolean_status = list(
                map(
                    lambda neighbour: self.filter_node_rec(level + 1, neighbour),
                    list(self.network[node]['hidden']),
                )
            )
            if len(boolean_status) == 0:
                return False
            else:
                return max(boolean_status)

    def filter_node(self, node):
        if self.hidden_status[node]:
            return True
        else:
            return self.filter_node_rec(0, node)

    def run_chain(self):
        infected_nodes = np.where(self.physical_status)[0]

        if len(infected_nodes) > 0:
            status_counts = defaultdict(int, Counter(itertools.chain(*map(self.hidden_chain, filter(self.filter_node, range(self.nodes_number))))))
            self.physical_status = self.physical_status_copy[:]
            self.hidden_status = self.hidden_status_copy[:]
            self.S_t.append(self.nodes_number - status_counts["I"])
            self.I_t.append(status_counts["I"])
            self.U_t.append(self.nodes_number - status_counts["A"])
            self.A_t.append(status_counts["A"])
        else:
            self.S_t.append(self.nodes_number)
            self.I_t.append(0)
            self.U_t.append(self.nodes_number)
            self.A_t.append(0)

    def run(self):
        for i in range(self.tmin, self.tmax):
            self.run_chain()
            self.times.append(i)

def multiplex_network(nodes_number, physical_network, hidden_network):
    
    layer_network = {}
    physical_dict = nx.to_dict_of_lists(physical_network)
    hidden_dict = nx.to_dict_of_lists(hidden_network)
    for node in range(nodes_number):
        layer_network[node] = {'hidden': hidden_dict[node], 'physical': physical_dict[node]}

    return layer_network

def multiplex_network_from_file(nodes_number, physical_network, hidden_network):
    
    layer_network = {}
    physical_dict = nx.to_dict_of_lists(physical_network)
    physical_dict = {int(k):[int(i) for i in v] for k,v in physical_dict.items()}
    hidden_dict = nx.to_dict_of_lists(hidden_network)
    hidden_dict = {int(k):[int(i) for i in v] for k,v in hidden_dict.items()}
    for node in range(nodes_number):
        layer_network[node] = {'hidden': [a-1 for a in hidden_dict[node+1]], 'physical': [p-1 for p in physical_dict[node+1]]}

    return layer_network

def from_file(name):
    my_graph = nx.Graph()
    edges = nx.read_edgelist("networks/out." + name)
    new_edges = [(int(a) - 1, int(b) - 1) for a, b in edges.edges()]
    my_graph.add_edges_from(new_edges)
    return my_graph


def random_edge(graph):
    edges = list(graph.edges)
    nonedges = list(nx.non_edges(graph))

    # random edge choice
    chosen_edge = random.choice(edges)
    exam_nodes_length = 0
    while exam_nodes_length <= 0:
        chosen_edge = random.choice(edges)
        exam_nodes = [x for x in nonedges if chosen_edge[0] == x[0]]
        exam_nodes_length = len(exam_nodes)

    chosen_nonedge = random.choice(exam_nodes)
    graph.add_edge(chosen_nonedge[0], chosen_nonedge[1])
    return graph