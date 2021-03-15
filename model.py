import random
import numpy as np
import networkx as nx

def markov_chain_AI(transition_prob_hidden, transition_prob_physical):
    change_hidden = np.random.choice(['U', 'A'],replace=True,p=[transition_prob_hidden, (1 - transition_prob_hidden)])
    change_physical = np.random.choice(['S', 'I'],replace=True,p=[transition_prob_physical, (1 - transition_prob_physical)])
    if change_hidden == 'U' and change_physical == 'I':
        return 'AI'
    else:
        return change_hidden + change_physical

def probability_AS(physical_layer, transition_prob_hidden, transition_prob_physical, factor, I_t, A_t, infectivity, node):
    probability = 1
    probability_AI = (1 - transition_prob_hidden)*(1 - transition_prob_physical) + transition_prob_hidden*(1 - transition_prob_physical)
    for neighbour in physical_layer.neighbors(node):
        if neighbour in I_t and neighbour in A_t:
            probability = probability*(1 - probability_AI*factor*infectivity)
    return probability

def probability_US(physical_layer, transition_prob_hidden, transition_prob_physical, I_t, A_t, infectivity, node):
    probability = 1
    probability_AI = (1 - transition_prob_hidden)*(1 - transition_prob_physical) + transition_prob_hidden*(1 - transition_prob_physical)
    for neighbour in physical_layer.neighbors(node):
        if neighbour in I_t and neighbour in A_t:
            probability = probability*(1 - probability_AI*infectivity)
    return probability  

def markov_chain_AS(transition_prob_hidden, transition_prob_physical, physical_layer, infectivity, factor, S_t, I_t, A_t, node):
    change_hidden = np.random.choice(['U', 'A'],replace=True,p=[transition_prob_hidden, (1 - transition_prob_hidden)])
    probability_AI = (1 - transition_prob_hidden)*(1 - transition_prob_physical) + transition_prob_hidden*(1 - transition_prob_physical)
    probability = 1
    if change_hidden == 'U':
        for neighbour in physical_layer.neighbors(node):
            if neighbour in I_t and neighbour in A_t:
                probability = probability*(1 - probability_AI*infectivity)
            elif neighbour in S_t and neighbour in A_t:
                probability = probability*(1 - probability_AS(physical_layer, transition_prob_hidden, transition_prob_physical, factor, I_t, A_t, infectivity, neighbour))
    else:
        for neighbour in physical_layer.neighbors(node):
            if neighbour in I_t and neighbour in A_t:
                probability = probability*(1 - probability_AI*factor*infectivity)
            elif neighbour in S_t and neighbour in A_t:
                probability = probability*(1 - probability_US(physical_layer, transition_prob_hidden, transition_prob_physical, I_t, A_t, infectivity, neighbour))
    change_physical = np.random.choice(['S', 'I'],replace=True,p=[probability, (1 - probability)])
    if change_hidden == 'U' and change_physical == 'I':
        return 'AI'
    else:
        return change_hidden + change_physical

def markov_chain_US(transition_prob_hidden, transition_prob_physical, _lambda, hidden_layer, physical_layer, infectivity, factor, S_t, I_t, A_t, node):
    probability_hidden = 1
    probability_physical = 1
    probability_AI = (1 - transition_prob_hidden)*(1 - transition_prob_physical) + transition_prob_hidden*(1 - transition_prob_physical)
    probability_A = probability_AI + (1 - transition_prob_hidden)*transition_prob_physical
    for neighbour in hidden_layer.neighbors(node):
        if neighbour in I_t and neighbour in A_t:
            probability_hidden = probability_hidden*(1 - probability_A*_lambda)
        elif neighbour in S_t and neighbour in A_t:
            probability_hidden = probability_hidden*((1 - transition_prob_hidden) + transition_prob_hidden*probability_AS(physical_layer, transition_prob_hidden, transition_prob_physical, factor, I_t, A_t, infectivity, neighbour))
    change_hidden = np.random.choice(['U', 'A'],replace=True,p=[probability_hidden, (1 - probability_hidden)])
    if change_hidden == 'U':
        for neighbour in physical_layer.neighbors(node):
            if neighbour in I_t and neighbour in A_t:
                probability_physical = probability_physical*(1 - probability_physical*infectivity)
            elif neighbour in S_t and neighbour in A_t:
                probability_physical = probability_physical*transition_prob_hidden
    else:
        for neighbour in physical_layer.neighbors(node):
            if neighbour in I_t and neighbour in A_t:
                probability_physical = probability_physical*(1 - probability_AI*factor*infectivity)
            elif neighbour in S_t and neighbour in A_t:
                probability_physical = probability_physical*(1 - probability_US(physical_layer, transition_prob_hidden, transition_prob_physical, I_t, A_t, infectivity, neighbour))
    change_physical = np.random.choice(['S', 'I'],replace=True,p=[probability_physical, (1 - probability_physical)])
    if change_hidden == 'U' and change_physical == 'I':
        return 'AI'
    else:
        return change_hidden + change_physical

def hidden_chain(physical_layer, hidden_layer, S_t, I_t, U_t, A_t, nodes, transition_prob_hidden, transition_prob_physical, infectivity, factor, _lambda):
    S = []
    I = []
    U = []
    A = []
    for node in I_t:
        if node in A_t and node in I_t:
            status = markov_chain_AI(transition_prob_hidden, transition_prob_physical)
        elif node in A_t and node in S_t:
            status = markov_chain_AS(transition_prob_hidden, transition_prob_physical, physical_layer, infectivity, factor, S_t, I_t, A_t, node)
        elif node in U_t and node in S_t:
            status = markov_chain_US(transition_prob_hidden, transition_prob_physical, _lambda, hidden_layer, physical_layer, infectivity, factor, S_t, I_t, A_t, node)
        else:
            status = 'AI'

        if status[1] == 'S':
            S.append(node)
        else:
            I.append(node)

        if status[0] == 'U':
            U.append(node)
        else:
            A.append(node)

        for neighbour in physical_layer.neighbors(node):
            if neighbour not in list(set(S + I + U + A)):
                if neighbour in A_t and neighbour in I_t:
                    status = markov_chain_AI(transition_prob_hidden, transition_prob_physical)
                elif neighbour in A_t and neighbour in S_t:
                    status = markov_chain_AS(transition_prob_hidden, transition_prob_physical, physical_layer, infectivity, factor, S_t, I_t, A_t, neighbour)
                elif neighbour in U_t and neighbour in S_t:
                    status = markov_chain_US(transition_prob_hidden, transition_prob_physical, _lambda, hidden_layer, physical_layer, infectivity, factor, S_t, I_t, A_t, neighbour)
                else:
                    status = 'AI'

                if status[1] == 'S':
                    S.append(neighbour)
                else:
                    I.append(neighbour)

                if status[0] == 'U':
                    U.append(neighbour)
                else:
                    A.append(neighbour)

    return S, I, U, A

def run(physical_nw, hidden_nw, transition_prob_hidden, transition_prob_physical,
        infectivity, factor, _lambda, initial_infecteds=None,
        rho = None, tmin=0, tmax=100):

    #Get initial infected nodes
    if initial_infecteds is None:
        if rho is None:
            initial_number = 1
        else:
            initial_number = int(round(physical_nw.order()*rho))
        initial_infecteds=random.sample(physical_nw.nodes(), initial_number)
    elif physical_nw.has_node(initial_infecteds):
        initial_infecteds=[initial_infecteds]

    times = [tmin]
    all_nodes = list(physical_nw.nodes())
    S = list(set(all_nodes) - set(initial_infecteds)) #Sustainable
    s_times = [len(S)]
    I = initial_infecteds[:] #Infected
    i_times = [len(I)]
    U = list(set(all_nodes) - set(initial_infecteds)) #Unaware
    u_times = [len(U)]
    A = initial_infecteds[:] #Aware
    a_times = [len(A)]

    while tmin < tmax:
        S, I, U, A = hidden_chain(physical_nw, hidden_nw, S, I,
                                U, A, all_nodes, transition_prob_hidden,
                                transition_prob_physical, infectivity, factor, _lambda)
        tmin = tmin+1
        times.append(tmin)
        s_times.append(len(S))
        i_times.append(len(I))
        u_times.append(len(U))
        a_times.append(len(A))
    
    return s_times, i_times, u_times, a_times, times

if __name__ == "__main__":
    nodes_number=1000
    hidden_layer = nx.scale_free_graph(nodes_number)
    physical_layer = nx.scale_free_graph(nodes_number,  alpha=0.45, beta=0.5, gamma=0.05)
    _lambda = 0.1
    rho = 0.2
    transition_prob_hidden = 0.6
    transition_prob_physical = 0.4
    infectivity = 0.3
    factor = 0.15
    s_times, i_times, u_times, a_times, times = run(hidden_layer, physical_layer, transition_prob_hidden, transition_prob_physical,
                                                    infectivity, _lambda, factor, rho=rho)
    print(i_times)