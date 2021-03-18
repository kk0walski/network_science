import random
import progressbar
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class MarkovModel():

    def __init__(self, physical_layer, hidden_layer, hidden_transition_prob, physical_transition_prob,
        infectivity, factor, _lambda, initial_infecteds=None,
        rho = None, tmin=0, tmax=100, epsilon = 1e-3):

        self.physical_nw = physical_layer
        self.hidden_nw = hidden_layer
        self.physical_transition_prob = physical_transition_prob
        self.hidden_transition_prob = hidden_transition_prob
        self.infectivity_unaware = infectivity
        self.infectivity_aware = factor*infectivity
        self._lambda = _lambda
        self.tmin = tmin
        self.tmax = tmax
        self.epsilon = epsilon
        self.init_simulation(initial_infecteds, rho, tmin)

    def init_simulation(self, initial_infecteds=None, rho = None, tmin=0):
        if initial_infecteds is None:
            if rho is None:
                self.initial_number = 1
            else:
                self.initial_number = int(round(self.physical_nw.order()*rho))
            self.initial_infecteds=random.sample(self.physical_nw.nodes(), self.initial_number)
        elif self.physical_nw.has_node(initial_infecteds):
            self.initial_infecteds=[initial_infecteds]

        self.times = [tmin]
        all_nodes = list(self.physical_nw.nodes())
        self.S = list(set(all_nodes) - set(self.initial_infecteds)) #Sustainable
        self.S_t = [len(self.S)]
        self.I = self.initial_infecteds[:] #Infected
        self.I_t = [len(self.I)]
        self.U = list(set(all_nodes) - set(self.initial_infecteds)) #Unaware
        self.U_t = [len(self.U)]
        self.A = self.initial_infecteds[:] #Aware
        self.A_t = [len(self.A)]

    def markov_chain_AI(self):
        change_hidden = np.random.choice(['U', 'A'],replace=True,p=[self.hidden_transition_prob, (1 - self.hidden_transition_prob)])
        change_physical = np.random.choice(['S', 'I'],replace=True,p=[self.physical_transition_prob, (1 - self.physical_transition_prob)])
        if change_hidden == 'U' and change_physical == 'I':
            return 'AI'
        else:
            return change_hidden + change_physical

    def probability_AS(self, node):
        _sum = 0
        for neighbour in self.physical_nw.neighbors(node):
            _sum = _sum + self.infectivity_aware*random.random()/(10*5)
        return _sum

    def probability_US(self, node):
        _sum = 0
        for neighbour in self.physical_nw.neighbors(node):
             _sum = _sum + self.infectivity_unaware*random.random()/(10*5)
        return _sum

    def markov_chain_AS(self, node):
        change_hidden = np.random.choice(['U', 'A'],replace=True,p=[self.hidden_transition_prob, (1 - self.hidden_transition_prob)])
        probability_AI = (1 - self.physical_transition_prob)
        probability = 1
        if change_hidden == 'U':
            for neighbour in self.physical_nw.neighbors(node):
                if neighbour in self.I and neighbour in self.A:
                    probability = probability*(1 - probability_AI*self.infectivity_unaware)
                elif neighbour in self.S and neighbour in self.A:
                    prob_AS = self.probability_AS(neighbour)
                    probability = probability*(1 - prob_AS*self.infectivity_unaware)
                else:
                    prob_US = self.probability_US(neighbour)
                    probability = probability*(1 - prob_US*self.infectivity_unaware)
        else:
            for neighbour in self.physical_nw.neighbors(node):
                if neighbour in self.I and neighbour in self.A:
                    probability = probability*(1 - probability_AI*self.infectivity_aware)
                elif neighbour in self.S and neighbour in self.A:
                    prob_AS = self.probability_AS(neighbour)
                    probability = probability*(1 - prob_AS*self.infectivity_aware)
                else:
                    prob_US = self.probability_US(neighbour)
                    probability = probability*(1 - prob_US*self.infectivity_aware)
        change_physical = np.random.choice(['S', 'I'],replace=True,p=[probability, (1 - probability)])
        if change_hidden == 'U' and change_physical == 'I':
            return 'AI'
        else:
            return change_hidden + change_physical

    def markov_chain_US(self, node):
        probability_hidden = 1
        probability_physical = 1
        probability_AI = (1 - self.physical_transition_prob)
        probability_A = probability_AI + (1 - self.hidden_transition_prob)*self.physical_transition_prob
        for neighbour in self.hidden_nw.neighbors(node):
            if neighbour in self.I and neighbour in self.A:
                probability_hidden = probability_hidden*(1 - probability_A*self._lambda)
            elif neighbour in self.S and neighbour in self.A:
                prob_AS = self.probability_AS(neighbour)
                probability_hidden = probability_hidden*(1 - self._lambda + prob_AS*self.hidden_transition_prob*self._lambda)
            else:
                prob_US = self.probability_US(neighbour)
                probability_hidden = probability_hidden*(1 - self._lambda + prob_US*self.hidden_transition_prob*self._lambda)
        change_hidden = np.random.choice(['U', 'A'],replace=True,p=[probability_hidden, (1 - probability_hidden)])
        if change_hidden == 'U':
            for neighbour in self.physical_nw.neighbors(node):
                if neighbour in self.I and neighbour in self.A:
                    probability_physical = probability_physical*(1 - probability_physical*self.infectivity_unaware)
                elif neighbour in self.S and neighbour in self.A:
                    prob_AS = self.probability_AS(neighbour)
                    probability_physical = probability_physical*(1 - prob_AS*self.infectivity_unaware)
                else:
                    prob_US = self.probability_US(neighbour)
                    probability = probability_physical*(1 - prob_US*self.infectivity_unaware)
        else:
            for neighbour in self.physical_nw.neighbors(node):
                if neighbour in self.I and neighbour in self.A:
                    probability_physical = probability_physical*(1 - probability_AI*self.infectivity_aware)
                elif neighbour in self.S and neighbour in self.A:
                    prob_AS = self.probability_AS(neighbour)
                    probability_physical = probability_physical*(1 - prob_AS*self.infectivity_aware)
                else:
                    prob_US = self.probability_US(neighbour)
                    probability = probability_physical*(1 - prob_US*self.infectivity_aware)
        change_physical = np.random.choice(['S', 'I'],replace=True,p=[probability_physical, (1 - probability_physical)])
        if change_hidden == 'U' and change_physical == 'I':
            return 'AI'
        else:
            return change_hidden + change_physical

    def hidden_chain(self, nodes):

        new_S = []
        new_I = []
        new_U = []
        new_A = []

        for node in nodes:
            if node in self.A and node in self.I:
                status = self.markov_chain_AI()
            elif node in self.A and node in self.S:
                status = self.markov_chain_AS(node)
            elif node in self.U and node in self.S:
                status = self.markov_chain_US(node)
            else:
                status = 'AI'

            if status[1] == 'S':
                new_S.append(node)
            else:
                new_I.append(node)

            if status[0] == 'U':
                new_U.append(node)
            else:
                new_A.append(node)

        self.S = new_S[:]
        self.A = new_A[:]
        self.I = new_I[:]
        self.U = new_U[:]
        self.S_t.append(len(new_S))
        self.I_t.append(len(new_I))
        self.U_t.append(len(new_U))
        self.A_t.append(len(new_A))


    def run(self):
        all_nodes = list(self.physical_nw.nodes())
        while self.tmin < self.tmax:
            self.hidden_chain(all_nodes)
            self.tmin = self.tmin+1
            self.times.append(self.tmin)

if __name__ == "__main__":
    nodes_number=1000
    physical_layer = nx.scale_free_graph(nodes_number)
    hidden_layer = nx.scale_free_graph(nodes_number,  alpha=0.4, beta=0.45, gamma=0.15)
    _lambda = 0.1
    rho = 0.2
    hidden_transition_prob = 0.6
    physical_transition_prob = 0.4
    factor = 0.15

    infectivity = 0.2
    sim = MarkovModel(hidden_layer, physical_layer, hidden_transition_prob, physical_transition_prob,
                                                    infectivity, _lambda, factor, rho=rho)
    sim.run()
    print(sim.I_t)
    print(np.mean(np.array(sim.I_t)/nodes_number))


    # i_probs = []
    # infectivities = []
    # for infectivity in progressbar.progressbar(np.linspace(0, 1, 100)):
    #     s_times, i_times, u_times, a_times, times = run(hidden_layer, physical_layer, hidden_transition_prob, physical_transition_prob,
    #                                                     infectivity, _lambda, factor, rho=rho)
    #     infectivities.append(infectivity)
    #     i_probs.append(np.mean(i_times)/nodes_number)
    # plt.plot(infectivities, i_probs)
    # plt.show()