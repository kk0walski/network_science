import random
import progressbar
import numpy as np
from pymnet import *
import multiprocessing as mp
from collections import defaultdict

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
        physical_dict = nx.to_dict_of_lists(physical_layer)
        physical_dict = {k: len(v) for k, v in physical_dict.items()}
        hidden_dict = nx.to_dict_of_lists(hidden_layer)
        hidden_dict = {k: len(v) for k, v in hidden_dict.items()}
        self.network = models.conf({'physical': physical_dict, 'hidden': hidden_dict}, degstype='nodes')
        self.tmin = tmin
        self.tmax = tmax

        if initial_infecteds is None:
            if rho is None:
                self.initial_number = 1
            else:
                self.initial_number = int(round(len(list(self.network.A['physical'])) * rho))
            self.initial_infecteds = random.sample(
                list(self.network.A['physical']), self.initial_number
            )
        elif min([infected in self.network.A['physical'] for infected in initial_infecteds]):
            self.initial_infecteds = [initial_infecteds]

    def rnd(self):
        exp = np.random.randint(-5, -1)
        significand = 0.9 * np.random.random() + 0.1
        return significand * 10 ** exp

    def default_value(self):
        return {"hidden": "U", "physical": "S"}

    def init_simulation(self):
        np.random.seed(100)
        self.tmin = 0
        self.times = [self.tmin]
        self.statuses = defaultdict(self.default_value)
        for node in self.initial_infecteds:
            self.statuses[node]["hidden"] = "A"
            self.statuses[node]["physical"] = "I"

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
            for neighbour in self.network.A['hidden'][node]:
                if neighbour != parent:
                    if self.statuses[neighbour]['hidden'] == "A":
                        if self.statuses[neighbour]['physical'] == "I":
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
                - len(list(self.network.A['hidden'][node])) * self.rnd() * self._lambda
            )
            return _sum

    def probability_AS(self, level, parent, node):
        probability = 1
        if level < self.level_limit:
            for neighbour in self.network.A['physical'][node]:
                if neighbour != parent:
                    if self.statuses[neighbour]['hidden'] == "A":
                        if self.statuses[neighbour]['physical'] == "I":
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
                - len(list(self.network.A['physical'][node]))
                * self.rnd()
                * self.infectivity_aware
            )
            return _sum

    def probability_US(self, level, parent, node):
        probability = 1
        if level < self.level_limit:
            for neighbour in self.network.A['physical'][node]:
                if neighbour != parent:
                    if self.statuses[neighbour]['hidden'] == "A":
                        if self.statuses[neighbour]['physical'] == "I":
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
                - len(list(self.network.A['physical'][node]))
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

        for neighbour in self.network.A['physical'][node]:
            if self.statuses[neighbour]['hidden'] == "A":
                if self.statuses[neighbour]['physical'] == "I":
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
        for neighbour in self.network.A['hidden'][node]:
            if self.statuses[neighbour]['hidden'] == "A":
                if self.statuses[neighbour]['physical'] == "I":
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

        for neighbour in self.network.A['physical'][node]:
            if self.statuses[neighbour]['hidden'] == "A":
                if self.statuses[neighbour]['physical'] == "I":
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
        physical_status = self.statuses[node]['physical']
        hidden_status = self.statuses[node]['hidden']
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
        if self.statuses[node]['hidden'] == "A":
            return True
        elif level == self.level_limit:
            return False
        else:
            boolean_status = list(
                map(
                    lambda neighbour: self.filter_node_rec(level + 1, neighbour),
                    list(self.network.A['hidden'][node]),
                )
            )
            if len(boolean_status) == 0:
                return False
            else:
                return max(boolean_status)

    def filter_node(self, node):
        if self.statuses[node]['hidden'] == "A":
            return (node, True)
        else:
            return (node, self.filter_node_rec(0, node))


    def run_chain(self, nodes, processes, nodes_number=1000):

        aware_nodes = list(
            filter(lambda x: self.statuses[x]['hidden'] == "A", list(self.network.A['hidden']),
            )
        )

        infected_nodes = list(
            filter(lambda x: self.statuses[x]['physical'] == "I", aware_nodes)
        )
        if len(infected_nodes) > 0:
            status_counts = {"S": 0, "I": 0, "A": 0, "U": 0}

            unaware_nodes = list(set(list(self.network.A['hidden'])) - set(aware_nodes))


            with mp.Pool(processes=(10)) as pool:
                filtered_unaware = []
                for node, to_process in pool.imap_unordered(
                    self.filter_node, unaware_nodes, chunksize=5
                ):
                    if to_process:
                        filtered_unaware.append(node)

                rest_number = nodes_number - len(filtered_unaware) - len(aware_nodes)
                exam_nodes = filtered_unaware + aware_nodes

                for node, hidden_status, physical_status in pool.imap_unordered(
                    self.hidden_chain, exam_nodes, chunksize=10
                ):
                    status_counts[hidden_status] = status_counts[hidden_status] + 1
                    self.statuses[node]['physical'] = physical_status
                    status_counts[physical_status] = status_counts[physical_status] + 1
                    self.statuses[node]['hidden'] = hidden_status

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
        all_nodes = list(self.network.A['physical'])
        for i in progressbar.progressbar(range(self.tmin, self.tmax)):
            self.run_chain(all_nodes, processes, len(all_nodes))
            self.times.append(i)