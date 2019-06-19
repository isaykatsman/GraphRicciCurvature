import collections
import sys

from GraphRicciCurvature.FormanRicci import formanCurvature
from GraphRicciCurvature.OllivierRicci import ricciCurvature
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from tqdm import tqdm


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

    graph = ricciCurvature(graph, alpha=0.5, method='OTD')
    o_curvatures, _ = get_edge_curvatures(graph)
    plot_curvatures(o_curvatures, 'tree_ollivier')

    # skip the last layer
    curvatures = []
    for _, v, attrs in graph.edges(data=True):
        if graph.degree[v] > 1:
            curvatures.append(attrs['ricciCurvature'])
    plot_curvatures(curvatures, 'tree_no_leaves_ollivier')


def plot_sphere(X, name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2])
    plt.savefig('{}.png'.format(name))
    plt.close()


def plot_degree_distribution(graph, name):
    degrees = sorted([d for n, d in graph.degree()], reverse=True)
    counts = collections.Counter(degrees)
    deg, cnt = zip(*counts.items())
    plt.bar(deg, cnt)
    plt.savefig('{}_degrees.png'.format(name))
    plt.close()


def sphere():
    n_samples = 2000
    n = 3
    edges_percent = 0.2

    X = np.ndarray(shape=(n_samples, n))
    for i in range(n_samples):
        x = np.random.multivariate_normal(np.zeros(n), np.eye(n))
        X[i] = x / np.linalg.norm(x)
    plot_sphere(X, 'sphere')

    inner_prods = X @ X.T
    dists = np.arccos(np.clip(inner_prods, -1, 1))
    print('Finished computing distances')

    dists_vec = dists[np.triu_indices(n_samples, 1)]
    n_edges = int((edges_percent * 0.5 * n_samples * (n_samples - 1)) / 100)
    neigh_threshold = np.partition(dists_vec, n_edges)[n_edges]
    print('Number of edges: {}, neigh threshold: {}'.format(
            n_edges, neigh_threshold))

    graph = nx.Graph()
    graph.add_edges_from(np.argwhere(dists < neigh_threshold))
    graph.remove_edges_from(graph.selfloop_edges())

    graph = ricciCurvature(graph, alpha=0.5, method='OTD')
    o_curvatures, _ = get_edge_curvatures(graph)
    plot_curvatures(o_curvatures, 'sphere_ollivier')


def regular_sphere():
    # Algorithm from https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
    n_samples = 1000
    a = 4 * np.pi / n_samples
    d = np.sqrt(a)
    m_nu = int(np.pi / d)
    d_nu = np.pi / m_nu
    d_phi = a / d_nu

    X = []
    for m in range(m_nu):
        nu = np.pi * (m + 0.5) / m_nu
        m_phi = int(2 * np.pi * np.sin(nu) / d_phi)
        for n in range(m_phi):
            phi = 2 * np.pi * n / m_phi
            x = np.array([
                    np.sin(nu) * np.cos(phi),
                    np.sin(nu) * np.sin(phi),
                    np.cos(nu)
            ])
            X.append(x)
    n_samples = len(X)
    print('Number of points: ', n_samples)

    # plot it to make sure it looks 'regular'
    X = np.array(X)
    plot_sphere(X, 'regular_sphere')

    # compute distances
    inner_prods = X @ X.T
    dists = np.arccos(np.clip(inner_prods, -1, 1))
    print('Finished computing distances')

    # search for the smallest distance which gives a single connected component
    low = 0.1
    high = 1.0
    while high - low > 1e-5:
        mid = (low + high) / 2
        graph = nx.Graph()
        graph.add_edges_from(np.argwhere(dists < mid))
        graph.remove_edges_from(graph.selfloop_edges())

        if nx.number_connected_components(graph) > 1:
            low = mid
        else:
            high = mid
    assert nx.number_connected_components(graph) == 1

    # degree distribution; should be small
    plot_degree_distribution(graph, 'regular_sphere')

    # curvature
    graph = ricciCurvature(graph, alpha=0.5, method='OTD')
    graph = formanCurvature(graph)
    o_curvatures, f_curvatures = get_edge_curvatures(graph)
    plot_curvatures(o_curvatures, 'regular_sphere_ollivier')
    plot_curvatures(f_curvatures, 'regular_sphere_forman')


def cycle():
    g = nx.cycle_graph(10)
    g = ricciCurvature(g, alpha=0.5, method='OTD')
    g = formanCurvature(g)
    o_curvatures, f_curvatures = get_edge_curvatures(g)
    print(o_curvatures, f_curvatures)


def balanced_tree():
    branching = 3
    depth = 5
    g = nx.balanced_tree(branching, depth)
    print('Number of edges: ', g.number_of_edges())

    g = ricciCurvature(g, alpha=0.5, method='OTD')
    o_curvatures, f_curvatures = get_edge_curvatures(g)
    plot_curvatures(o_curvatures, 'balanced_tree_ollivier')

    # this shows that the positively curved edges are on the last layer
    for limit in range(1, depth + 1):
        curvatures = []
        for u, v in nx.bfs_edges(g, 0, depth_limit=limit):
            curvatures.append(g[u][v]['ricciCurvature'])
        plot_curvatures(np.array(curvatures), '{}_balanced'.format(limit))


def main():
    # full_graph()
    tree()
    # sphere()
    # regular_sphere()
    # cycle()
    # balanced_tree()


if __name__ == '__main__':
    sys.exit(main())
