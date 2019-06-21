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
    plt.hist(curvatures)
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
    fig = plt.figure(figsize=(15, 15))
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


def search_smallest_dist_for_connected_graph(dists):
    # search for the smallest distance which gives a single connected component
    low = 0.01
    high = 10.0
    while high - low > 1e-5:
        mid = (low + high) / 2
        graph = nx.Graph()
        graph.add_edges_from(np.argwhere(dists < mid))
        graph.remove_edges_from(graph.selfloop_edges())

        if nx.number_connected_components(graph) > 1:
            low = mid
        else:
            high = mid

    print('Neighbor threshold distance: ', high)
    graph = nx.Graph()
    graph.add_edges_from(np.argwhere(dists < high))
    graph.remove_edges_from(graph.selfloop_edges())
    assert nx.number_connected_components(graph) == 1

    return graph


def largest_smallest_connecting_distance(dists):
    dists_copy = dists.copy()
    np.fill_diagonal(dists_copy, np.inf)
    min_dists = np.min(dists_copy, axis=1)
    neigh_threshold = np.max(min_dists)

    graph = nx.Graph()
    # multiply by some threshold to make sure it is connected; note that the
    # above formula doesn't guarantees connectedness
    graph.add_edges_from(np.argwhere(dists < 1.3 * neigh_threshold))
    graph.remove_edges_from(graph.selfloop_edges())
    assert nx.number_connected_components(graph) == 1

    return graph


def regular_sphere(toy=False):
    if toy:
        # yapf: disable
        X = [
            [0, 0, 1], # north pole
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], # equator
            [0, 0, -1], # south pole
        ]
        # yapf: enable
    else:
        # Algorithm from https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
        n_samples = 100
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
    print('Number of points: ', len(X))

    # plot it to make sure it looks 'regular'
    X = np.array(X)
    plot_sphere(X, 'regular_sphere')

    # compute distances
    inner_prods = X @ X.T
    dists = np.arccos(np.clip(inner_prods, -1, 1))
    print('Finished computing distances')

    # search for the smallest distance which gives a single connected component
    #   graph = search_smallest_dist_for_connected_graph(dists)
    # even better: use the largest-smallest connecting distance
    graph = largest_smallest_connecting_distance(dists)

    # degree distribution; should be small
    plot_degree_distribution(graph, 'regular_sphere')

    # curvature
    graph = ricciCurvature(graph, alpha=0.5, method='OTD')
    graph = formanCurvature(graph)
    o_curvatures, f_curvatures = get_edge_curvatures(graph)
    if toy:
        print(o_curvatures, f_curvatures)
    else:
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


def erdos_renyi():
    n = 1000
    p = 0.01

    g = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.rand() < p:
                g.add_edge(i, j)
    print('Number of edges: ', g.number_of_edges())
    print('Number of connected components: ', nx.number_connected_components(g))

    g = ricciCurvature(g, alpha=0.5, method='OTD')
    g = formanCurvature(g)
    o_curvs, f_curvs = get_edge_curvatures(g)
    plot_curvatures(o_curvs, 'o_gnp')
    plot_curvatures(f_curvs, 'f_gnp')


def hypercube():
    g = nx.hypercube_graph(5)
    g = ricciCurvature(g, alpha=0.5, method='OTD')
    g = formanCurvature(g)
    o_curvs, f_curvs = get_edge_curvatures(g)
    print(o_curvs, f_curvs)


def grid():
    g = nx.grid_graph([5, 5, 5], periodic=True)
    g = ricciCurvature(g, alpha=0.5, method='OTD')
    g = formanCurvature(g)
    o_curvs, f_curvs = get_edge_curvatures(g)
    plot_curvatures(o_curvs, 'o_grid')
    plot_curvatures(f_curvs, 'f_grid')


def small_sphere():
    g = nx.Graph()
    g.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 5),
                      (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)])
    g = ricciCurvature(g, alpha=0.5, method='OTD')
    g = formanCurvature(g)
    o_curvs, f_curvs = get_edge_curvatures(g)
    print(o_curvs, f_curvs)


def main():
    # full_graph()
    # tree()
    # sphere()
    # regular_sphere()
    # cycle()
    # balanced_tree()
    # erdos_renyi()
    # hypercube()
    # grid()
    # small_sphere()
    regular_sphere(toy=True) # should be the same as `small_sphere()`


if __name__ == '__main__':
    sys.exit(main())
