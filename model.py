import random
import numpy as np

def markov_chain_AI(transition_prob_hidden, transition_prob_physical):
    change_hidden = np.random.choice(['U', 'A'],replace=True,p=[transition_prob_hidden, (1 - transition_prob_hidden)])
    change_physical = np.random.choice(['S', 'I'],replace=True,p=[transition_prob_physical, (1 - transition_prob_physical)])
    if change_hidden == 'U' and change_physical == 'I':
        return 'AI'
    else:
        return change_hidden + change_physical

def markov_chain_AS(transition_prob_hidden, physical_layer, infectivity, factor, I_t, A_t, node):
    change_hidden = np.random.choice(['U', 'A'],replace=True,p=[transition_prob_hidden, (1 - transition_prob_hidden)])
    probability = 1
    if change_hidden == 'U':
        for neighbour in physical_layer.neighbors(node):
            if neighbour in I_t and neighbour in A_t:
                probability = probability*(1 - infectivity)
    else:
        for neighbour in physical_layer.neighbors(node):
            if neighbour in I_t and neighbour in A_t:
                probability = probability*(1 - factor*infectivity)
    change_physical = np.random.choice(['S', 'I'],replace=True,p=[probability, (1 - probability)])
    if change_hidden == 'U' and change_physical == 'I':
        return 'AI'
    else:
        return change_hidden + change_physical

def markov_chain_US(_lambda, hidden_layer, physical_layer, infectivity, factor, I_t, A_t, node):
    probability_hidden = 1
    probability_physical = 1
    for neighbour in hidden_layer.neighbors(node):
        if neighbour in A_t:
            probability_hidden = probability_hidden*(1 - _lambda)
    change_hidden = np.random.choice(['U', 'A'],replace=True,p=[probability_hidden, (1 - probability_hidden)])
    if change_hidden == 'U':
        for neighbour in physical_layer.neighbors(node):
            if neighbour in I_t and neighbour in A_t:
                probability_physical = probability_physical*(1 - infectivity)
    else:
        for neighbour in physical_layer.neighbors(node):
            if neighbour in I_t and neighbour in A_t:
                probability_physical = probability_physical*(1 - factor*infectivity)
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
    for node in nodes:
        if node in A_t and node in I_t:
            status = markov_chain_AI(transition_prob_hidden, transition_prob_physical)
        elif node in A_t and node in S_t:
            status = markov_chain_AS(transition_prob_hidden, physical_layer, infectivity, factor, I_t, A_t, node)
        elif node in U_t and node in S_t:
            status = markov_chain_US(_lambda, hidden_layer, physical_layer, infectivity, factor, I_t, A_t, node)
        else:
            status = 'AI'

        if status != 'AI':
            S.append(node)
            if status[0] == 'U':
                U.append(node)
            else:
                A.append(node)
        else:
            A.append(node)
            I.append(node)
    return S, I, U, A

def run(physical_nw, hidden_nw, transition_prob_hidden, transition_prob_physical,
        infectivity, factor, _lambda, initial_infecteds=None,
        rho = None, tmin=0, tmax=100):

    #Get initial infected nodes
    if initial_infecteds is None:
        if rho is None:
            initial_number = 1
        else:
            initial_number = int(round(G.order()*rho))
        initial_infecteds=random.sample(G.nodes(), initial_number)
    elif physical_nw.has_node(initial_infecteds):
        initial_infecteds=[initial_infecteds]

    times = [tmin]
    S = list(set(physical_nw.order()) - set(initial_infecteds[:])) #Sustainable
    s_times = [len(S)]
    I = initial_infecteds[:] #Infected
    i_times = [len(I)]
    U = list(set(physical_nw.order()) - set(initial_infecteds[:])) #Unaware
    u_times = [len(U)]
    A = initial_infecteds[:] #Aware
    a_times = [len(A)]

    while tmin < tmax:
        S, I, U, A = hidden_chain(physical_nw, hidden_nw, S, I,
                                U, A, list(physical_nw.order()), transition_prob_hidden,
                                transition_prob_physical, infectivity, factor, _lambda)
        tmin = tmin+1
        times.append(tmin)
        s_times.append(len(S))
        i_times.append(len(I))
        u_times.append(len(U))
        a_times.append(len(A))
    
    return s_times, i_times, u_times, a_times, times