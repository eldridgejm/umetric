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

    return _distance.squareform(ultrametric)


def ultrametric_to_dendrogram(ultrametric):
    """Computes the equivalent dendrogram of the ultrametric."""
    slc = _hierarchy.linkage(ultrametric, method='single')
    return hierarchical_clustering_to_dendrogram(slc)


def linkage_ultrametric(distances, method="single"):
    """Computes the ultrametric corresponding to the dendrogram produced by
    the given linkage method. 'distances' is a condensed metric."""
    clustering = _hierarchy.linkage(distances, method=method)
    dendrogram = hierarchical_clustering_to_dendrogram(clustering)
    return dendrogram_to_ultrametric(dendrogram)


def condensed_indices(n, ix):
    """Converts the indices from (i,j) to their location in the condensed 
    distance array."""
    ix = _np.asarray(ix)
    i = ix[:,0]
    j = ix[:,1]
    
    return n*(n-1)/2 - (n-i)*(n-i-1)/2 + (j-i-1)


def triangle_inequality_indices(n):
    """Generates all triples of indices (i,j,k) such that it must be
    that d_ij <= d_ik + d_jk."""
    # generate all pairs
    pairs = _np.column_stack(_np.triu_indices(n, k=1))
    
    ix,k = _np.indices((pairs.shape[0], n)).reshape((2,-1))
    i,j = pairs[ix].T
    
    triple_idx = (i != k) & (j != k)
    
    i = i[triple_idx]
    j = j[triple_idx]
    k = k[triple_idx]
    
    return i,j,k


def non_ultrametric_triples(m):
    """Returns triples of indices in the condensed metric for which it is not 
    ultrametric."""
    n = number_of_points(m.shape[0])
    
    # generate all indices to check
    i,j,k = triangle_inequality_indices(n)
    
    # convert these to condensed matrix indices
    ij = condensed_indices(n, i, j)
    ik = condensed_indices(n, i, k)
    jk = condensed_indices(n, j, k)
    
    lix = (m[ik] != m[jk]) & (m[ij] <= _np.minimum(m[ik], m[jk]))
    return ij[lix], ik[lix], jk[lix]


def is_ultrametric(m):
    ij, ik, jk = non_ultrametric_triples(m)
    return ij.shape[0] == 0


def number_of_points(n):
    """Given a number of pairwise distances, returns the number of points
    in the dissimilarity."""
    return int(1 + _np.sqrt(1 + 8*n))/2


def condensed_indices(n, i, j, upper=False):
    """Returns the index of the (i,j) point in the condensed metric array."""
    i = _np.asarray(i)
    j = _np.asarray(j)
    
    if not upper:
        i_old = i
        j_old = j
        
        cond = i<j
        i = _np.where(cond, i_old, j_old)
        j = _np.where(cond, j_old, i_old)

    return n*(n-1)/2 - (n-i)*(n-i-1)/2 + (j-i-1)


def cophenetic_correlation(actual, approximate):
    x = _np.asarray(actual)
    y = _np.asarray(approximate)

    return 1 - _np.sum((x - y)**2) / _np.sum((x - x.mean())**2)
