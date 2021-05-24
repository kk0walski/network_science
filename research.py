import os
import csv
import json
import signal
import progressbar
from multiprocessing import Pool
import numpy as np
import pandas as pd
import networkx as nx
from dictonaryModel import random_edge, MarkovModel, multiplex_network, multiplex_network_from_file, from_file

def read_network(network):
    with open('networks/' + network + '.json') as json_file:
        data = json.load(json_file)
        reasult = {int(k):v for k,v in data.items()}
    return reasult

def create_network(nodes):
    if type(nodes) == str:
        physical_layer = from_file(nodes)
        nodes_number = len(physical_layer.nodes())
        hidden_layer = physical_layer.copy()
        for i in range(400):
            hidden_layer = random_edge(hidden_layer)
        return multiplex_network_from_file(nodes_number, physical_layer, hidden_layer)
    else:
        nodes_number = nodes
        physical_layer = nx.barabasi_albert_graph(nodes_number, 5)
        hidden_layer = physical_layer.copy()
        for i in range(400):
            hidden_layer = random_edge(hidden_layer)
        return multiplex_network(nodes_number, physical_layer, hidden_layer)

def experiment(parameters):
    filepath, column_names, tree_level, network_name, physical_prob, hidden_prob, infectivity, _lambda, media = parameters
    multiplex_network = read_network(network_name)
    nodes_number = len(multiplex_network.keys())
    model = MarkovModel(nodes_number, multiplex_network,  rho=0.2)
    model.set_level(tree_level)
    model.set_factor(0.01)
    model.set_physical_trans_prob(physical_prob)
    model.set_hidden_trans_prob(hidden_prob)
    model.set_infectivity(infectivity)
    model.set_lambda(_lambda)
    model.set_media(media)
    reasult = {"nodes": nodes_number, "rho": 0.2,
        "network": network_name, "beta": infectivity, 
        "lambda": _lambda, "factor": 0.01,
        "physical_infectivity": physical_prob,
        "hidden_infectivity": hidden_prob, "media": media,
        "aware": 0, "infected": 0
        }
    model.init_simulation()
    model.run()
    reasult["infected"] = np.mean(np.array(model.I_t) / nodes_number)
    reasult["aware"] = np.mean(np.array(model.A_t) / nodes_number)
    try:
        with open(filepath, "a") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=column_names, lineterminator='\n')
            writer.writerow(reasult)
    except IOError:
        print("I/O error")

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def do_research(filepath, tree_level):
    column_names = ["nodes", "rho", "network", "beta", "lambda", "media", "factor", "physical_infectivity", "hidden_infectivity", "aware", "infected"]
    if not os.path.isfile(filepath):
        try:
            with open(filepath, "a", newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=column_names)
                writer.writeheader()
        except IOError:
            print("I/O error")
    
    parameters = []
    for network in ["hiv", "school"]:
        for media in [0]: #np.round(np.linspace(0,1,11), 2):
            for physical_prob in np.round(np.linspace(0,1,11), 2):
                for hidden_prob in np.round(np.linspace(0,1,11), 2):
                    for infectivity in np.round(np.linspace(0,1,11), 2):
                        for _lambda in np.round(np.linspace(0,1,11), 2):
                            parameters.append((filepath, column_names, tree_level, network, physical_prob, hidden_prob, infectivity, _lambda, media))

    print("Parametry wygenerowane")

    with Pool(initializer=init_worker) as pool:
        for _ in progressbar.progressbar(pool.imap_unordered(experiment, parameters, chunksize=250)):
            pass


if __name__ == "__main__":
    do_research("experiment14.csv", 2)