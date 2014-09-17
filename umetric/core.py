import numpy as _np
import scipy.spatial.distance as _distance
import scipy.cluster.hierarchy as _hierarchy
import networkx as _nx
import itertools as _itertools


def hierarchical_clustering_to_dendrogram(clustering):
    """Converts a scipy hierarchical clustering encoding matrix to a networkx
    tree object where each node has the 'distance' attribute."""
    root = _hierarchy.to_tree(clustering)

    tree = _nx.DiGraph()
    tree.add_node(root.id, distance=root.dist)

    if root.left:
        queue = [(root, root.left), (root, root.right)]

    while queue:
        parent, child = queue.pop(0)

        tree.add_edge(parent.id, child.id)
        tree.node[child.id]['distance'] = float(child.dist)

        if child.left:
            queue.append((child, child.left))

        if child.right:
            queue.append((child, child.right))

    return tree


def dendrogram_to_ultrametric(dendrogram):
    """Returns the ultrametric corresponding to the dendrogram."""
    t = dendrogram.copy()

    leaf_nodes = [x for x in t if t.out_degree(x) == 0]

    n = sum(1 for _ in leaf_nodes)
    ultrametric = _np.zeros((n,n))

    for u in leaf_nodes:
        t.node[u]['leafy_ancestors'] = set([u])

    for u in list(_nx.dfs_postorder_nodes(t)):
        if t.out_degree(u) == 0:
            continue

        leafy_ancestors = []
        for child in t.successors(u):
            leafy_ancestors.append(t.node[child]['leafy_ancestors'])
            t.remove_node(child)

        d = t.node[u]['distance']

        for i in range(len(leafy_ancestors)):
            for j in range(len(leafy_ancestors)):
                if i == j:
                    continue
                
                left_set = leafy_ancestors[i]
                right_set = leafy_ancestors[j]

                for x,y in _itertools.product(left_set, right_set):
                    ultrametric[x][y] = ultrametric[y][x] = d

        ancestors = set()
        for x in leafy_ancestors:
            ancestors |= x

        t.node[u]['leafy_ancestors'] = ancestors

    return ultrametric


def ultrametric_to_dendrogram(ultrametric):
    """Computes the equivalent dendrogram of the ultrametric."""
    condensed_distances = _distance.squareform(ultrametric)
    slc = _hierarchy.linkage(condensed_distances, method='single')
    return hierarchical_clustering_to_dendrogram(slc)


def linkage_ultrametric(distances, method="single"):
    """Computes the ultrametric corresponding to the dendrogram produced by
    the given linkage method."""
    condensed_distances = _distance.squareform(distances)
    clustering = _hierarchy.linkage(condensed_distances, method=method)
    dendrogram = hierarchical_clustering_to_dendrogram(clustering)
    return dendrogram_to_ultrametric(dendrogram)


def non_ultrametric_triples(d):
    """Returns all 3-tuples (i,j,k) for which d[i,j] < min(d[i,k], d[j,k]).
    Assumes a symmetric input matrix."""
    n = d.shape[0]
    grid = _np.mgrid[:n, :n, :n]
    inds = _np.column_stack(x.flatten() for x in grid)
    
    inds = inds[inds[:,0] < inds[:,1]]
    inds = inds[inds[:,0] < inds[:,2]]
    inds = inds[inds[:,1] < inds[:,2]]
    
    i,j,k = inds.T
    
    d_ij = d[i,j]
    d_ik = d[i,k]
    d_jk = d[j,k]
    
    non_um_inds = d_ij < _np.min(_np.column_stack((d_ik, d_jk)), axis=1)
    non_um_inds = _np.logical_and(non_um_inds, d_ik != d_jk)
    return inds[non_um_inds]


def is_ultrametric(m):
    """Given a metric matrix m, verifies the three-point condition."""
    n = m.shape[0]

    if not _np.allclose(_np.diag(m), 0):
        return False

    if not _np.allclose(m, m.T):
        return False

    if non_ultrametric_triples(m).shape[0] > 0:
        return False
    
    return True


def enforce_ultrametricity(m, maxiters=10):
    """Given an metric matrix m that is close to being an ultrametric, 
    enforces ultrametricity by ensuring that for any set of points (i,j,k)
    for which d_ij < min(d_ik, d_jk), we set d_ik = d_jk."""
    q = m.copy()

    triples = non_ultrametric_triples(q)
    n = 0
    while triples.shape[0] > 0:

        if n >= maxiters:
            raise RuntimeError("Exceeded maximum number of iterations.""")

        i,j,k = triples.T
        v = _np.min(_np.column_stack((q[i,k], q[j,k])), axis=1)
        q[i,k] = q[j,k] = v

        q = _np.triu(q, k=1)
        q = q + q.T

        triples = non_ultrametric_triples(q)

        n += 1

    return q


    
