import random
import progressbar
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import multiprocessing as mp


class MarkovModel:
    def __init__(
        self,
        nodes_number,
        physical_layer,
        hidden_layer,
        initial_infecteds=None,
        rho=None,
        tmin=0,
        tmax=30,
    ):
        self.nodes_number = nodes_number
        self.physical_nw = physical_layer
        self.hidden_nw = hidden_layer
        self.tmin = tmin
        self.tmax = tmax

        if initial_infecteds is None:
            if rho is None:
                self.initial_number = 1
            else:
                self.initial_number = int(round(self.physical_nw.order() * rho))
            self.initial_infecteds = random.sample(
                self.physical_nw.nodes(), self.initial_number
            )
        elif self.physical_nw.has_node(initial_infecteds):
            self.initial_infecteds = [initial_infecteds]

    # def __init__(
    #     self,
    #     nodes_number,
    #     physical_layer,
    #     hidden_layer,
    #     hidden_transition_prob,
    #     physical_transition_prob,
    #     infectivity,
    #     factor,
    #     _lambda,
    #     initial_infecteds=None,
    #     rho=None,
    #     level_limit=2,
    #     tmin=0,
    #     tmax=100,
    # ):

    #     self.nodes_number = nodes_number
    #     self.physical_nw = physical_layer
    #     self.hidden_nw = hidden_layer
    #     self.physical_transition_prob = physical_transition_prob
    #     self.hidden_transition_prob = hidden_transition_prob
    #     self.infectivity_unaware = infectivity
    #     self.infectivity_aware = factor * infectivity
    #     self._lambda = _lambda
    #     self.tmin = tmin
    #     self.tmax = tmax
    #     self.probability_AI = 1 - self.physical_transition_prob
    #     self.level_limit = level_limit
    #     self.probability_A = (
    #         1 - self.physical_transition_prob * self.hidden_transition_prob
    #     )

    #     if initial_infecteds is None:
    #         if rho is None:
    #             self.initial_number = 1
    #         else:
    #             self.initial_number = int(round(self.physical_nw.order() * rho))
    #         self.initial_infecteds = random.sample(
    #             self.physical_nw.nodes(), self.initial_number
    #         )
    #     elif self.physical_nw.has_node(initial_infecteds):
    #         self.initial_infecteds = [initial_infecteds]
    #     self.init_simulation()

    def rnd(self):
        exp = np.random.randint(-5, -1)
        significand = 0.9 * np.random.random() + 0.1
        return significand * 10 ** exp

    def init_simulation(self):
        np.random.seed(100)
        self.tmin = 0
        self.times = [self.tmin]
        nx.set_node_attributes(self.hidden_nw, "U", "status")
        nx.set_node_attributes(self.physical_nw, "S", "status")
        for node in self.initial_infecteds:
            self.hidden_nw.nodes[node]["status"] = "A"
            self.physical_nw.nodes[node]["status"] = "I"

        self.S_t = [self.nodes_number - len(self.initial_infecteds)]
        self.U_t = [self.nodes_number - len(self.initial_infecteds)]
        self.A_t = [len(self.initial_infecteds)]
        self.I_t = [len(self.initial_infecteds)]

    def set_infectivity(self, infectivity):
        self.infectivity_unaware = infectivity
        self.infectivity_aware = self.factor * infectivity

    def set_level(self, level):
        self.level_limit = level

    def set_lambda(self, _lambda):
        self._lambda = _lambda

    def set_hidden_layer(self, layer):
        self.hidden_nw = layer

    def set_physical_layer(self, layer):
        self.physical_nw = layer

    def set_physical_trans_prob(self, prob):
        self.physical_transition_prob = prob
        self.probability_AI = 1 - self.physical_transition_prob

    def set_hidden_trans_prob(self, prob):
        self.hidden_transition_prob = prob
        self.probability_A = (
            1 - self.physical_transition_prob * self.hidden_transition_prob
        )

    def set_factor(self, factor):
        self.factor = factor

    def markov_chain_AI(self):
        change_hidden = np.random.choice(
            ["U", "A"],
            p=[self.hidden_transition_prob, (1 - self.hidden_transition_prob)],
        )
        change_physical = np.random.choice(
            ["S", "I"],
            p=[self.physical_transition_prob, (1 - self.physical_transition_prob)],
        )
        return change_hidden, change_physical

    def r_prob(self, level, parent, node):
        probability_hidden = 1
        if level < self.level_limit:
            for neighbour in self.hidden_nw.neighbors(node):
                if neighbour != parent:
                    if self.hidden_nw.nodes[neighbour]["status"] == "A":
                        if self.physical_nw.nodes[neighbour]["status"] == "I":
                            probability_hidden = probability_hidden * (
                                1 - self.probability_A * self._lambda
                            )
                        else:
                            prob_US = self.probability_US(level + 1, node, neighbour)
                            probability_temp = 1 - prob_US * self.hidden_transition_prob
                            probability_hidden = probability_hidden * (
                                1 - probability_temp * self._lambda
                            )
                    else:
                        prob_US = self.probability_US(level + 1, node, neighbour)
                        prob_r = self.r_prob(level + 1, node, neighbour)
                        probability_temp = 1 - prob_r * prob_US
                        probability_hidden = probability_hidden * (
                            1 - probability_temp * self._lambda
                        )
                        
            return probability_hidden
        else:
            _sum = (
                1
                - len(list(self.hidden_nw.neighbors(node))) * self.rnd() * self._lambda
            )
            return _sum

    def probability_AS(self, level, parent, node):
        probability = 1
        if level < self.level_limit:
            for neighbour in self.physical_nw.neighbors(node):
                if neighbour != parent:
                    if self.hidden_nw.nodes[neighbour]["status"] == "A":
                        if self.physical_nw.nodes[neighbour]["status"] == "I":
                            probability = probability * (
                                1 - self.probability_AI * self.infectivity_aware
                            )
                        else:
                            prob_AS = self.probability_AS(level + 1, node, neighbour)
                            prob_US = self.probability_US(level + 1, node, neighbour)
                            probability_temp = (
                                1
                                - prob_AS
                                + self.hidden_transition_prob * (prob_AS - prob_US)
                            )
                            probability = probability * (
                                1 - probability_temp * self.infectivity_aware
                            )
                    else:
                        prob_US = self.probability_US(level + 1, node, neighbour)
                        prob_AS = self.probability_AS(level + 1, node, neighbour)
                        prob_r = self.r_prob(level + 1, node, neighbour)
                        probability_temp = 1 - prob_AS + prob_r * (prob_AS - prob_US)
                        probability = probability * (
                            1 - probability_temp * self.infectivity_aware
                        )
            return probability
        else:
            _sum = (
                1
                - len(list(self.physical_nw.neighbors(node)))
                * self.rnd()
                * self.infectivity_aware
            )
            return _sum

    def probability_US(self, level, parent, node):
        probability = 1
        if level < self.level_limit:
            for neighbour in self.physical_nw.neighbors(node):
                if neighbour != parent:
                    if self.hidden_nw.nodes[neighbour]["status"] == "A":
                        if self.physical_nw.nodes[neighbour]["status"] == "I":
                            probability = probability * (
                                1 - self.probability_AI * self.infectivity_unaware
                            )
                        else:
                            prob_AS = self.probability_AS(level + 1, node, neighbour)
                            prob_US = self.probability_US(level + 1, node, neighbour)
                            probability_temp = (
                                1
                                - prob_AS
                                + self.hidden_transition_prob * (prob_AS - prob_US)
                            )
                            probability = probability * (
                                1 - probability_temp * self.infectivity_unaware
                            )
                    else:
                        prob_US = self.probability_US(level + 1, node, neighbour)
                        prob_AS = self.probability_AS(level + 1, node, neighbour)
                        prob_r = self.r_prob(level + 1, node, neighbour)
                        probability_temp = 1 - prob_AS + prob_r * (prob_AS - prob_US)
                        probability = probability * (
                            1 - probability_temp * self.infectivity_unaware
                        )
            return probability
        else:
            _sum = (
                1
                - len(list(self.physical_nw.neighbors(node)))
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
        chosen_infectivity = hidden_states[status_index][1]

        for neighbour in self.physical_nw.neighbors(node):
            if self.hidden_nw.nodes[neighbour]["status"] == "A":
                if self.physical_nw.nodes[neighbour]["status"] == "I":
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

        if probability < 0:
            probability = 0
        change_physical = np.random.choice(
            ["S", "I"], p=[probability, (1 - probability)]
        )
        return hidden_states[status_index][0], change_physical

    def markov_chain_US(self, node):
        probability_hidden = 1
        probability_physical = 1
        for neighbour in self.hidden_nw.neighbors(node):
            if self.hidden_nw.nodes[neighbour]["status"] == "A":
                if self.physical_nw.nodes[neighbour]["status"] == "I":
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

        if probability_hidden < 0:
            probability_hidden = 0

        hidden_states = [("U", self.infectivity_unaware), ("A", self.infectivity_aware)]
        status_index = np.random.choice(
            2,
            p=[self.hidden_transition_prob, (1 - self.hidden_transition_prob)],
        )

        chosen_infectivity = hidden_states[status_index][1]

        for neighbour in self.physical_nw.neighbors(node):
            if self.hidden_nw.nodes[neighbour]["status"] == "A":
                if self.physical_nw.nodes[neighbour]["status"] == "I":
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

        if probability_physical < 0:
            probability_physical = 0
        change_physical = np.random.choice(
            ["S", "I"], p=[probability_physical, (1 - probability_physical)]
        )
        return hidden_states[status_index][0], change_physical

    def hidden_chain(self, node):
        physical_status = self.physical_nw.nodes[node]["status"]
        hidden_status = self.hidden_nw.nodes[node]["status"]
        if hidden_status == "A":
            if physical_status == "S":
                hidden_status, physical_status = self.markov_chain_AS(node)
            else:
                hidden_status, physical_status = self.markov_chain_AI()
        else:
            if physical_status == "S":
                hidden_status, physical_status = self.markov_chain_US(node)
            else:
                hidden_status, physical_status = "A", "I"

        if hidden_status == "U" and physical_status == "I":
            return node, "A", "I"
        return node, hidden_status, physical_status

    def filter_node_rec(self, level, node):
        if self.hidden_nw.nodes[node]["status"] == "A":
            return True
        elif level == self.level_limit:
            return False
        else:
            boolean_status = list(
                map(
                    lambda neighbour: self.filter_node_rec(level + 1, neighbour),
                    self.hidden_nw.neighbors(node),
                )
            )
            if len(boolean_status) == 0:
                return False
            else:
                return max(boolean_status)

    def filter_node(self, node):
        if self.hidden_nw.nodes[node]["status"] == "A":
            return (node, True)
        else:
            return (node, self.filter_node_rec(0, node))

    def run_chain(self, nodes, processes, nodes_number=1000):

        aware_nodes = list(
            map(
                lambda y: y[0],
                filter(lambda x: x[1] == "A", self.hidden_nw.nodes(data="status")),
            )
        )

        infected_nodes = list(
            filter(lambda x: self.physical_nw.nodes[x]["status"] == "I", aware_nodes)
        )
        if len(infected_nodes) > 0:
            status_counts = {"S": 0, "I": 0, "A": 0, "U": 0}

            unaware_nodes = self.hidden_nw.nodes() - aware_nodes

            with mp.Pool(processes=(10)) as filtering:
                filtered_unaware = []
                for node, to_process in filtering.imap_unordered(
                    self.filter_node, unaware_nodes, chunksize=5
                ):
                    if to_process:
                        filtered_unaware.append(node)

            rest_number = nodes_number - len(filtered_unaware) - len(aware_nodes)
            exam_nodes = filtered_unaware + aware_nodes

            with mp.Pool(processes=(10)) as pool:
                for node, hidden_status, physical_status in pool.imap_unordered(
                    self.hidden_chain, exam_nodes, chunksize=10
                ):
                    status_counts[hidden_status] = status_counts[hidden_status] + 1
                    self.physical_nw.nodes[node]["status"] = physical_status
                    status_counts[physical_status] = status_counts[physical_status] + 1
                    self.hidden_nw.nodes[node]["status"] = hidden_status

            self.S_t.append(status_counts["S"] + rest_number)
            self.I_t.append(status_counts["I"])
            self.U_t.append(status_counts["U"] + rest_number)
            self.A_t.append(status_counts["A"])
        else:
            self.S_t.append(len(nodes))
            self.I_t.append(0)
            self.U_t.append(len(nodes))
            self.A_t.append(0)

    def run(self, processes=None):
        all_nodes = list(self.physical_nw.nodes())
        for i in range(self.tmin, self.tmax):
            self.run_chain(all_nodes, processes, len(all_nodes))
            self.times.append(i)


def generate_graph(nodes):
    while True:
        s = []
        while len(s) < nodes:
            nextval = int(
                nx.utils.powerlaw_sequence(1, 2.5)[0]
            )  # 100 nodes, power-law exponent 2.5
            if nextval != 0:
                s.append(nextval)
        if sum(s) % 2 == 0:
            break
    G = nx.configuration_model(s)
    G = nx.Graph(G)  # remove parallel edges
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


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
    infectivity = 0.1

    sim = MarkovModel(
        nodes_number,
        physical_layer,
        hidden_layer,
        hidden_transition_prob,
        physical_transition_prob,
        infectivity,
        factor,
        _lambda,
        rho=rho,
    )
    sim.run(processes=4)
    print(sim.I_t)
    print(np.mean(np.array(sim.I_t) / nodes_number))

    # i_probs = []
    # a_probs = []
    # infectivities = []
    # sim = MarkovModel(nodes_number, physical_layer, hidden_layer,
    #                 hidden_transition_prob, physical_transition_prob,
    #                 infectivity, factor, _lambda, rho=rho)

    # for infectivity in np.linspace(0, 1, 20):
    #     sim.set_infectivity(infectivity)
    #     infectivities.append(infectivity)
    #     sim.run()
    #     i_probs.append(np.mean(sim.I_t)/nodes_number)
    #     a_probs.append(np.mean(sim.A_t)/nodes_number)
    #     sim.init_simulation()
    # plt.plot(infectivities, i_probs)
    # print(i_probs)
    # plt.plot(infectivities, a_probs)
    # plt.show()