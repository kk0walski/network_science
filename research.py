import os
import csv
import progressbar
import numpy as np
import pandas as pd
import networkx as nx
from dictonaryModel import random_edge, MarkovModel, multiplex_network, multiplex_network_from_file, from_file

def do_research(filepath):
    column_names = ["nodes", "rho", "network", "beta", "lambda", "factor", "physical_infectivity", "hidden_infectivity", "aware", "infected"]
    if not os.path.isfile(filepath):
        try:
            with open(r"experiment7.csv", "a", newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=column_names)
                writer.writeheader()
        except IOError:
            print("I/O error")

    for nodes_number in [100, 1000]:
        physical_layer = nx.barabasi_albert_graph(nodes_number, 5)
        hidden_layer = physical_layer.copy()
        for i in range(400):
            hidden_layer = random_edge(hidden_layer)
        network = multiplex_network(nodes_number, physical_layer, hidden_layer)
        model = MarkovModel(nodes_number, network,  rho=0.2)
        model.set_level(2)
        model.set_factor(0.01)
        for physical_prob in np.round(np.linspace(0, 1, 11), 2):
            model.set_physical_trans_prob(physical_prob)
            for hidden_prob in np.round(np.linspace(0, 1, 11), 2):
                model.set_hidden_trans_prob(hidden_prob)
                for infectivity in progressbar.progressbar(np.round(np.linspace(0,1,11), 2)):
                    model.set_infectivity(infectivity)
                    for _lambda in np.round(np.linspace(0,1,11), 2):
                        model.set_lambda(_lambda)
                        reasult = {"nodes": nodes_number, "rho": 0.2,
                            "network": "barabassi", "beta": infectivity, 
                            "lambda": _lambda, "factor": 0.01,
                            "physical_infectivity": physical_prob,
                            "hidden_infectivity": hidden_prob,
                            "aware": 0, "infected": 0
                            }
                        model.init_simulation()
                        model.run()
                        reasult["infected"] = np.mean(np.array(model.I_t) / nodes_number)
                        reasult["aware"] = np.mean(np.array(model.A_t) / nodes_number) 
                        try:
                            with open(r"experiment7.csv", "a") as csvfile:
                                writer = csv.DictWriter(csvfile, fieldnames=column_names)
                                writer.writerow(reasult)
                        except IOError:
                            print("I/O error")

if __name__ == "__main__":
    do_research("experiment7.csv")