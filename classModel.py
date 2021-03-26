import random
import progressbar
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class MarkovModel():

    def __init__(self, physical_layer, hidden_layer, hidden_transition_prob, physical_transition_prob,
        infectivity, factor, _lambda, initial_infecteds=None,
        rho = None, tmin=0, tmax=60, epsilon = 1e-5):

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
        self.probability_A = (1 - self.physical_transition_prob*self.hidden_transition_prob)

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
        len_nodes = len(self.physical_nw.nodes())
        for node in self.physical_nw.nodes():
            if node in self.initial_infecteds:
                self.hidden_nw.nodes[node]['status'] = 'A'
                self.physical_nw.nodes[node]['status'] = 'I'
            else:
                self.hidden_nw.nodes[node]['status'] = 'U'
                self.physical_nw.nodes[node]['status'] = 'S'

        self.S_t = [len_nodes - len(self.initial_infecteds)]
        self.U_t = [len_nodes - len(self.initial_infecteds)]
        self.A_t = [len(self.initial_infecteds)]
        self.I_t = [len(self.initial_infecteds)]

    def set_infectivity(self, infectivity):
        self.infectivity_unaware = infectivity
        self.infectivity_aware = factor*infectivity

    def markov_chain_AI(self):
        change_hidden = np.random.choice(['U', 'A'], p=[self.hidden_transition_prob, (1 - self.hidden_transition_prob)])
        change_physical = np.random.choice(['S', 'I'], p=[self.physical_transition_prob, (1 - self.physical_transition_prob)])
        if change_hidden == 'U' and change_physical == 'I':
            return 'AI'
        else:
            return change_hidden + change_physical

    def r_prob_second(self, node):
        probability_hidden = 1
        for neighbour in self.hidden_nw.neighbors(node):
            if self.physical_nw.nodes[neighbour]['status'] == 'I' and self.hidden_nw.nodes[neighbour]['status'] == 'A':
                probability_hidden = probability_hidden*(1 - self.probability_A*self._lambda)
            elif self.physical_nw.nodes[neighbour]['status'] == 'S' and self.hidden_nw.nodes[neighbour]['status'] == 'A':
                prob_US = self.probability_US(neighbour)
                probability_temp = (1 - self.hidden_transition_prob + (1 - prob_US)*self.hidden_transition_prob)
                probability_hidden = probability_hidden*(1 - probability_temp*self._lambda)
        return probability_hidden

    def r_prob(self, node):
        probability_hidden = 1
        for neighbour in self.hidden_nw.neighbors(node):
            if self.physical_nw.nodes[neighbour]['status'] == 'I' and self.hidden_nw.nodes[neighbour]['status'] == 'A':
                probability_hidden = probability_hidden*(1 - self.probability_A*self._lambda)
            elif self.physical_nw.nodes[neighbour]['status'] == 'S' and self.hidden_nw.nodes[neighbour]['status'] == 'A':
                prob_US = self.probability_US(neighbour)
                probability_temp = (1 - self.hidden_transition_prob + (1 - prob_US)*self.hidden_transition_prob)
                probability_hidden = probability_hidden*(1 - probability_temp*self._lambda)
            else:
                prob_US = self.probability_US(neighbour)
                prob_r = self.r_prob_second(neighbour)
                probability_temp = (1 - prob_r*prob_US)
                probability_hidden = probability_hidden*(1 - probability_temp*self._lambda)
        return probability_hidden

    def probability_AS(self, node):
        _sum = (1 - len(list(self.physical_nw.neighbors(node)))*self.epsilon*self.infectivity_aware)
        return _sum

    def probability_US(self, node):
        _sum = (1 - len(list(self.physical_nw.neighbors(node)))*self.epsilon*self.infectivity_unaware)
        return _sum

    def markov_chain_AS(self, node):
        change_hidden = np.random.choice(['U', 'A'],p=[self.hidden_transition_prob, (1 - self.hidden_transition_prob)])
        probability = 1
        if change_hidden == 'U':
            for neighbour in self.physical_nw.neighbors(node):
                if self.physical_nw.nodes[neighbour]['status'] == 'I' and self.hidden_nw.nodes[neighbour]['status'] == 'A':
                   probability = probability*(1 - self.probability_AI*self.infectivity_unaware)
                elif self.physical_nw.nodes[neighbour]['status'] == 'S' and self.hidden_nw.nodes[neighbour]['status'] == 'A':
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
                if self.physical_nw.nodes[neighbour]['status'] == 'I' and self.hidden_nw.nodes[neighbour]['status'] == 'A':
                    probability = probability*(1 - self.probability_AI*self.infectivity_aware)
                elif self.physical_nw.nodes[neighbour]['status'] == 'S' and self.hidden_nw.nodes[neighbour]['status'] == 'A':
                    prob_AS = self.probability_AS(neighbour)
                    probability = probability*(1 - prob_AS*self.infectivity_aware)
                else:
                    prob_r = self.r_prob(neighbour)
                    prob_AS = self.probability_AS(neighbour)
                    prob_US = self.probability_US(neighbour)
                    probability_temp = ((1-prob_r)*(1 - prob_AS) + prob_r*(1-prob_US))
                    probability = probability*(1 - probability_temp*self.infectivity_aware)
        change_physical = np.random.choice(['S', 'I'],p=[probability, (1 - probability)])
        if change_hidden == 'U' and change_physical == 'I':
            return 'AI'
        else:
            return change_hidden + change_physical

    def markov_chain_US(self, node):
        probability_hidden = 1
        probability_physical = 1
        for neighbour in self.hidden_nw.neighbors(node):
            if self.physical_nw.nodes[neighbour]['status'] == 'I' and self.hidden_nw.nodes[neighbour]['status'] == 'A':
                probability_hidden = probability_hidden*(1 - self.probability_A*self._lambda)
            elif self.physical_nw.nodes[neighbour]['status'] == 'S' and self.hidden_nw.nodes[neighbour]['status'] == 'A':
                prob_AS = self.probability_AS(neighbour)
                probability_temp = (1 - self.hidden_transition_prob + (1 - prob_AS)*self.hidden_transition_prob)
                probability_hidden = probability_hidden*(1 - probability_temp*self._lambda)
            else:
                prob_r = self.r_prob(neighbour)
                prob_US = self.probability_US(neighbour)
                probability_temp = (1 - prob_r*prob_US)
                probability_hidden = probability_hidden*(1 - probability_temp*self._lambda)
        change_hidden = np.random.choice(['U', 'A'],p=[probability_hidden, (1 - probability_hidden)])
        if change_hidden == 'U':
            for neighbour in self.physical_nw.neighbors(node):
                if self.physical_nw.nodes[neighbour]['status'] == 'I' and self.hidden_nw.nodes[neighbour]['status'] == 'A':
                    probability_physical = probability_physical*(1 - probability_physical*self.infectivity_unaware)
                elif self.physical_nw.nodes[neighbour]['status'] == 'S' and self.hidden_nw.nodes[neighbour]['status'] == 'A':
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
                if self.physical_nw.nodes[neighbour]['status'] == 'I' and self.hidden_nw.nodes[neighbour]['status'] == 'A':
                    probability_physical = probability_physical*(1 - self.probability_AI*self.infectivity_aware)
                elif self.physical_nw.nodes[neighbour]['status'] == 'S' and self.hidden_nw.nodes[neighbour]['status'] == 'A':
                    prob_AS = self.probability_AS(neighbour)
                    probability_physical = probability_physical*(1 - prob_AS*self.infectivity_aware)
                else:
                    prob_r = self.r_prob(neighbour)
                    prob_AS = self.probability_AS(neighbour)
                    prob_US = self.probability_US(neighbour)
                    probability_temp = ((1-prob_r)*(1 - prob_AS) + prob_r*(1-prob_US))
                    probability_physical = probability_physical*(1 - probability_temp*self.infectivity_aware)
        change_physical = np.random.choice(['S', 'I'],p=[probability_physical, (1 - probability_physical)])
        if change_hidden == 'U' and change_physical == 'I':
            return 'AI'
        else:
            return change_hidden + change_physical

    def hidden_chain(self, nodes):

        infected_nodes = list(filter(lambda x: x[1] == 'I', self.physical_nw.nodes(data='status')))
        if len(infected_nodes) > 0:
            s_count = 0
            i_count = 0
            a_count = 0
            u_count = 0

            for node in nodes:
                if self.physical_nw.nodes[node]['status'] == 'I' and self.hidden_nw.nodes[node]['status'] == 'A':
                    status = self.markov_chain_AI()
                elif self.physical_nw.nodes[node]['status'] == 'S' and self.hidden_nw.nodes[node]['status'] == 'A':
                    status = self.markov_chain_AS(node)
                elif self.physical_nw.nodes[node]['status'] == 'S' and self.hidden_nw.nodes[node]['status'] == 'U':
                    status = self.markov_chain_US(node)
                else:
                    status = 'AI'

                if status[1] == 'S':
                    s_count = s_count + 1
                    self.physical_nw.nodes[node]['status'] = 'S'
                else:
                    i_count = i_count + 1
                    self.physical_nw.nodes[node]['status'] = 'I'

                if status[0] == 'U':
                    u_count = u_count + 1
                    self.hidden_nw.nodes[node]['status'] = 'U'
                else:
                    a_count = a_count + 1
                    self.hidden_nw.nodes[node]['status'] = 'A'

            self.S_t.append(s_count)
            self.I_t.append(i_count)
            self.U_t.append(u_count)
            self.A_t.append(a_count)
        else:
            self.S_t.append(len(nodes))
            self.I_t.append(0)
            self.U_t.append(len(nodes))
            self.A_t.append(0)

    def run(self):
        all_nodes = list(self.physical_nw.nodes())
        for i in progressbar.progressbar(range(self.tmin, self.tmax)):
            self.hidden_chain(all_nodes)
            self.times.append(i)


def generate_graph(nodes):
    while True:  
        s=[]
        while len(s)<nodes:
            nextval = int(nx.utils.powerlaw_sequence(1, 2.5)[0]) #100 nodes, power-law exponent 2.5
            if nextval!=0:
                s.append(nextval)
        if sum(s)%2 == 0:
            break
    G = nx.configuration_model(s)
    G=nx.Graph(G) # remove parallel edges
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
    nodes_number=1000
    physical_layer = generate_graph(nodes_number)
    hidden_layer = physical_layer.copy()
    for i in range(400):
        hidden_layer = random_edge(hidden_layer)
    _lambda = 0.15
    rho = 0.2
    hidden_transition_prob = 0.6
    physical_transition_prob = 0.4
    factor = 1e-4
    infectivity = 0.1

    sim = MarkovModel(physical_layer, hidden_layer, hidden_transition_prob, physical_transition_prob,
                                                    infectivity, factor, _lambda, rho=rho)
    sim.run()
    print(sim.I_t)
    print(np.mean(np.array(sim.I_t)/nodes_number))


    # i_probs = []
    # a_probs = []
    # infectivities = []
    # sim = MarkovModel(physical_layer, hidden_layer, hidden_transition_prob, physical_transition_prob,
    #                                                 infectivity, factor, _lambda, rho=rho)
    # for infectivity in progressbar.progressbar(np.linspace(0, 1, 20)):
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