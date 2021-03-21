import random
import progressbar
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class MarkovModel():

    def __init__(self, physical_layer, hidden_layer, hidden_transition_prob, physical_transition_prob,
        infectivity, factor, _lambda, initial_infecteds=None,
        rho = None, tmin=0, tmax=100, epsilon = 1e-5):

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
        self.probability_AI = (1 - self.physical_transition_prob)
        self.probability_A = self.probability_AI + (1 - self.hidden_transition_prob)*self.physical_transition_prob

        if initial_infecteds is None:
            if rho is None:
                self.initial_number = 1
            else:
                self.initial_number = int(round(self.physical_nw.order()*rho))
            self.initial_infecteds=random.sample(self.physical_nw.nodes(), self.initial_number)
        elif self.physical_nw.has_node(initial_infecteds):
            self.initial_infecteds=[initial_infecteds]
        self.init_simulation()

    def init_simulation(self):
        self.tmin = 0
        self.times = [self.tmin]
        all_nodes = list(self.physical_nw.nodes())
        self.S = list(set(all_nodes) - set(self.initial_infecteds)) #Sustainable
        self.S_t = [len(self.S)]
        self.I = self.initial_infecteds[:] #Infected
        self.I_t = [len(self.I)]
        self.U = list(set(all_nodes) - set(self.initial_infecteds)) #Unaware
        self.U_t = [len(self.U)]
        self.A = self.initial_infecteds[:] #Aware
        self.A_t = [len(self.A)]

    def set_infectivity(self, infectivity):
        self.infectivity_unaware = infectivity
        self.infectivity_aware = factor*infectivity

    def markov_chain_AI(self):
        change_hidden = np.random.choice(['U', 'A'],replace=True,p=[self.hidden_transition_prob, (1 - self.hidden_transition_prob)])
        change_physical = np.random.choice(['S', 'I'],replace=True,p=[self.physical_transition_prob, (1 - self.physical_transition_prob)])
        if change_hidden == 'U' and change_physical == 'I':
            return 'AI'
        else:
            return change_hidden + change_physical

    def r_prob_second(self, node):
        probability_hidden = 1
        for neighbour in self.hidden_nw.neighbors(node):
            if neighbour in self.I and neighbour in self.A:
                probability_hidden = probability_hidden*(1 - self.probability_A*self._lambda)
            elif neighbour in self.S and neighbour in self.A:
                prob_US = self.probability_US_second(neighbour)
                probability_temp = (1 - self.hidden_transition_prob + (1 - prob_US)*self.hidden_transition_prob)
                probability_hidden = probability_hidden*(1 - probability_temp*self._lambda)
        return probability_hidden

    def r_prob(self, node):
        probability_hidden = 1
        for neighbour in self.hidden_nw.neighbors(node):
            if neighbour in self.I and neighbour in self.A:
                probability_hidden = probability_hidden*(1 - self.probability_A*self._lambda)
            elif neighbour in self.S and neighbour in self.A:
                prob_US = self.probability_US_second(neighbour)
                probability_temp = (1 - self.hidden_transition_prob + (1 - prob_US)*self.hidden_transition_prob)
                probability_hidden = probability_hidden*(1 - probability_temp*self._lambda)
            else:
                prob_US = self.probability_US_second(neighbour)
                prob_r = self.r_prob_second(neighbour)
                probability_temp = (1 - prob_r*prob_US)
                probability_hidden = probability_hidden*(1 - probability_temp*self._lambda)
        return probability_hidden


    def probability_AS_second(self, node):
        _sum = (1 - len(list(self.physical_nw.neighbors(node)))*self.epsilon*self.infectivity_aware)
        return _sum

    def probability_US_second(self, node):
        _sum = (1 - len(list(self.physical_nw.neighbors(node)))*self.epsilon*self.infectivity_unaware)
        return _sum

    def probability_AS(self, node):
        probability = 1
        for neighbour in self.physical_nw.neighbors(node):
            if neighbour in self.I and neighbour in self.A:
                probability = probability*(1 - self.probability_AI*self.infectivity_aware)
            elif neighbour in self.S and neighbour in self.A:
                prob_AS = self.probability_AS_second(neighbour)
                prob_US = self.probability_US_second(neighbour)
                probability_temp = ((1-self.hidden_transition_prob)*prob_AS + self.hidden_transition_prob*prob_US)
                probability = probability*(1 - probability_temp*self.infectivity_aware)
            else:
                prob_US = self.probability_US_second(neighbour)
                prob_AS = self.probability_AS_second(neighbour)
                prob_r = self.r_prob_second(neighbour)
                probability_temp = ((1-prob_r)*(1 - prob_AS) + prob_r*(1-prob_US))
                probability = probability*(1 - probability_temp*self.infectivity_aware)
        return probability

    def probability_US(self, node):
        probability = 1
        for neighbour in self.physical_nw.neighbors(node):
            if neighbour in self.I and neighbour in self.A:
                probability = probability*(1 - self.probability_AI*self.infectivity_unaware)
            elif neighbour in self.S and neighbour in self.A:
                prob_AS = self.probability_AS_second(neighbour)
                prob_US = self.probability_US_second(neighbour)
                probability_temp = ((1-self.hidden_transition_prob)*prob_AS + self.hidden_transition_prob*prob_US)
                probability = probability*(1 - probability_temp*self.infectivity_unaware)
            else:
                prob_US = self.probability_US_second(neighbour)
                prob_AS = self.probability_AS_second(neighbour)
                prob_r = self.r_prob_second(neighbour)
                probability_temp = ((1-prob_r)*(1 - prob_AS) + prob_r*(1-prob_US))
                probability = probability*(1 - probability_temp*self.infectivity_unaware)
        return probability

    def markov_chain_AS(self, node):
        change_hidden = np.random.choice(['U', 'A'],replace=True,p=[self.hidden_transition_prob, (1 - self.hidden_transition_prob)])
        probability = 1
        if change_hidden == 'U':
            for neighbour in self.physical_nw.neighbors(node):
                if neighbour in self.I and neighbour in self.A:
                    probability = probability*(1 - self.probability_AI*self.infectivity_unaware)
                elif neighbour in self.S and neighbour in self.A:
                    prob_AS = self.probability_AS(neighbour)
                    probability = probability*(1 - prob_AS*self.infectivity_unaware)
                else:
                    prob_r = self.r_prob(neighbour)
                    prob_AS = self.probability_AS(neighbour)
                    prob_US = self.probability_US(neighbour)
                    probability_temp = ((1-prob_r)*(1 - prob_AS) + prob_r*(1-prob_US))
                    probability = probability*(1 - probability_temp*self.infectivity_unaware)
        else:
            for neighbour in self.physical_nw.neighbors(node):
                if neighbour in self.I and neighbour in self.A:
                    probability = probability*(1 - self.probability_AI*self.infectivity_aware)
                elif neighbour in self.S and neighbour in self.A:
                    prob_AS = self.probability_AS(neighbour)
                    probability = probability*(1 - prob_AS*self.infectivity_aware)
                else:
                    prob_r = self.r_prob(neighbour)
                    prob_AS = self.probability_AS(neighbour)
                    prob_US = self.probability_US(neighbour)
                    probability_temp = ((1-prob_r)*(1 - prob_AS) + prob_r*(1-prob_US))
                    probability = probability*(1 - probability_temp*self.infectivity_unaware)
        change_physical = np.random.choice(['S', 'I'],replace=True,p=[probability, (1 - probability)])
        if change_hidden == 'U' and change_physical == 'I':
            return 'AI'
        else:
            return change_hidden + change_physical

    def markov_chain_US(self, node):
        probability_hidden = 1
        probability_physical = 1
        for neighbour in self.hidden_nw.neighbors(node):
            if neighbour in self.I and neighbour in self.A:
                probability_hidden = probability_hidden*(1 - self.probability_A*self._lambda)
            elif neighbour in self.S and neighbour in self.A:
                prob_AS = self.probability_AS(neighbour)
                probability_temp = (1 - self.hidden_transition_prob + (1 - prob_AS)*self.hidden_transition_prob)
                probability_hidden = probability_hidden*(1 - probability_temp*self._lambda)
            else:
                prob_r = self.r_prob(neighbour)
                prob_US = self.probability_US(neighbour)
                probability_temp = (1 - prob_r*prob_US)
                probability_hidden = probability_hidden*(1 - probability_temp*self._lambda)
        change_hidden = np.random.choice(['U', 'A'],replace=True,p=[probability_hidden, (1 - probability_hidden)])
        if change_hidden == 'U':
            for neighbour in self.physical_nw.neighbors(node):
                if neighbour in self.I and neighbour in self.A:
                    probability_physical = probability_physical*(1 - probability_physical*self.infectivity_unaware)
                elif neighbour in self.S and neighbour in self.A:
                    prob_AS = self.probability_AS(neighbour)
                    probability_physical = probability_physical*(1 - prob_AS*self.infectivity_unaware)
                else:
                    prob_r = self.r_prob(neighbour)
                    prob_AS = self.probability_AS(neighbour)
                    prob_US = self.probability_US(neighbour)
                    probability_temp = ((1-prob_r)*(1 - prob_AS) + prob_r*(1-prob_US))
                    probability_physical = probability_physical*(1 - probability_temp*self.infectivity_unaware)
        else:
            for neighbour in self.physical_nw.neighbors(node):
                if neighbour in self.I and neighbour in self.A:
                    probability_physical = probability_physical*(1 - self.probability_AI*self.infectivity_aware)
                elif neighbour in self.S and neighbour in self.A:
                    prob_AS = self.probability_AS(neighbour)
                    probability_physical = probability_physical*(1 - prob_AS*self.infectivity_aware)
                else:
                    prob_r = self.r_prob(neighbour)
                    prob_AS = self.probability_AS(neighbour)
                    prob_US = self.probability_US(neighbour)
                    probability_temp = ((1-prob_r)*(1 - prob_AS) + prob_r*(1-prob_US))
                    probability_physical = probability_physical*(1 - probability_temp*self.infectivity_aware)
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

        if len(self.I) > 0:
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
        else:
            for node in nodes:
                new_S.append(node)
                new_U.append(node)

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
    s = nx.utils.powerlaw_sequence(1000, 2.5) #100 nodes, power-law exponent 2.5
    physical_layer = nx.expected_degree_graph(s, selfloops=False)
    s = nx.utils.powerlaw_sequence(1000, 2.4) #100 nodes, power-law exponent 2.5
    hidden_layer = nx.expected_degree_graph(s, selfloops=False)
    _lambda = 0.1
    rho = 0.2
    hidden_transition_prob = 0.6
    physical_transition_prob = 0.4
    factor = 0.15
    infectivity = 0.1

    # sim = MarkovModel(hidden_layer, physical_layer, hidden_transition_prob, physical_transition_prob,
    #                                                 infectivity, _lambda, factor, rho=rho)
    # sim.run()
    # print(sim.I_t)
    # print(np.mean(np.array(sim.I_t)/nodes_number))


    i_probs = []
    a_probs = []
    infectivities = []
    sim = MarkovModel(hidden_layer, physical_layer, hidden_transition_prob, physical_transition_prob,
                                                    infectivity, _lambda, factor, rho=rho, tmax=100)
    for infectivity in progressbar.progressbar(np.linspace(0, 1, 20)):
        sim.set_infectivity(infectivity)
        infectivities.append(infectivity)
        sim.run()
        i_probs.append(np.mean(sim.I_t)/nodes_number)
        a_probs.append(np.mean(sim.A_t)/nodes_number)
        sim.init_simulation()
    plt.plot(infectivities, i_probs)
    print(i_probs)
    plt.plot(infectivities, a_probs)
    plt.show()