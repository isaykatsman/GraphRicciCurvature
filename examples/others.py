import sys

from GraphRicciCurvature.FormanRicci import formanCurvature
from GraphRicciCurvature.OllivierRicci import ricciCurvature
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def get_edge_curvatures(graph):
    ollivier_curvatures = []
    forman_curvatures = []
    for _, _, attrs in graph.edges(data=True):
        if 'ricciCurvature' in attrs:
            ollivier_curvatures.append(attrs['ricciCurvature'])
        if 'formanCurvature' in attrs:
            forman_curvatures.append(attrs['ricciCurvature'])

    return np.array(ollivier_curvatures), np.array(forman_curvatures)


def plot_curvatures(curvatures, name):
    plt.hist(curvatures, bins=20)
    plt.xlabel('Curvature')
    plt.ylabel('Edges')
    plt.savefig('{}_curvatures.png'.format(name))
    plt.close()


def full_graph():
    graph = nx.complete_graph(100)
    graph = ricciCurvature(graph, alpha=0.5, method='ATD')
    graph = formanCurvature(graph)

    o_curvatures, f_curvatures = get_edge_curvatures(graph)
    plot_curvatures(o_curvatures, 'full_ollivier')
    plot_curvatures(f_curvatures, 'full_forman')


def tree():
    graph = nx.Graph()
    graph.add_edge(0, 1)
    next_id = 2
    for _ in range(1234):
        u = np.random.randint(0, next_id)
        graph.add_edge(u, next_id)
        next_id += 1

    graph = ricciCurvature(graph, alpha=0.5, method='ATD')
    graph = formanCurvature(graph)
    o_curvatures, f_curvatures = get_edge_curvatures(graph)
    plot_curvatures(o_curvatures, 'tree_ollivier')
    plot_curvatures(f_curvatures, 'tree_forman')


def main():
    full_graph()
    tree()


if __name__ == '__main__':
    sys.exit(main())
