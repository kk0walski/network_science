# import powerlaw
# import networkx as nx
# import matplotlib.pyplot as plt

# for _seed in range(1000):
#     g = nx.barabasi_albert_graph(1000, 6, seed=779)
#     degrees = {}
#     for node in g.nodes():
#         key = len(list(g.neighbors(node)))
#         degrees[key] = degrees.get(key, 0) + 1

#     max_degree = max(degrees.keys(), key=int)
#     num_nodes = []
#     for i in range(1, max_degree + 1):
#         num_nodes.append(degrees.get(i, 0))

#     fit = powerlaw.Fit(num_nodes)
#     print(fit.power_law.alpha)
#     if abs(2.5 - fit.power_law.alpha) < 0.5:
#         print(_seed)
#         break
# nx.draw(g)
# plt.show()

import powerlaw
import collections
import matplotlib.pyplot as plt
import networkx as nx

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

G = nx.powerlaw_cluster_graph(1000, 5, p=0.001)
lista = list(filter(lambda x: G.degree[x] < 24, G.nodes()))
lista = lista[0:200]
print(max(node in G for node in lista))

print(lista[0:200])

# degrees = {}
# for node in G.nodes():
#     key = len(list(G.neighbors(node)))
#     degrees[key] = degrees.get(key, 0) + 1

# max_degree = max(degrees.keys(), key=int)
# num_nodes = []
# for i in range(1, max_degree + 1):
#     num_nodes.append(degrees.get(i, 0))

# fit = powerlaw.Fit(num_nodes)
# print(fit.power_law.alpha)


# degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
# degreeCount = collections.Counter(degree_sequence)
# deg, cnt = zip(*degreeCount.items())

# fig, ax = plt.subplots()
# plt.bar(deg, cnt, width=0.80, color="b")

# plt.title("Degree Histogram")
# plt.ylabel("Count")
# plt.xlabel("Degree")
# ax.set_xticks([d + 0.4 for d in deg])
# ax.set_xticklabels(deg)

# # draw graph in inset
# plt.axes([0.4, 0.4, 0.5, 0.5])
# Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
# pos = nx.spring_layout(G)
# plt.axis("off")
# nx.draw_networkx_nodes(G, pos, node_size=20)
# nx.draw_networkx_edges(G, pos, alpha=0.4)
# plt.show()