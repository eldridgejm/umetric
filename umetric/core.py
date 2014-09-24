import numpy as _np
import scipy.spatial.distance as _distance
import scipy.cluster.hierarchy as _hierarchy
import networkx as _nx
import itertools as _itertools


def hierarchical_clustering_to_dendrogram(clustering):
    """Converts an array representing a clustering to a dendrogram.

    Args:
        clustering (ndarray): A hierarchical clustering matrix, in the form
            returned by scipy.hierarchical.linkage.

    Returns:
        (networkx.DiGraph): A dendrogram. Each node in the dendrogram has the
        'distance' attribute, which is the threshold at which its children
        are merged in the clustering.
    """
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
    """Returns the ultrametric corresponding to the dendrogram.

    Args:
        dendrogram (networkx.DiGraph): The dendrogram to be converted. Each
        node should have a 'distance' attribute, which denotes the threshold
        at which the children are merged.
        
    Returns:
        (ndarray): The ultrametric as a condensed distance matrix.
    """
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
    """Computes the equivalent dendrogram of the ultrametric.
    
    Args:
        ultrametric (ndarray): The ultrametric as a condensed distance 
            matrix.
            
    Returns:
        (networkx.DiGraph): The dendrogram.

    Works by computing the single-linkage clustering of the ultrametric.
    """
    slc = _hierarchy.linkage(ultrametric, method='single')
    return hierarchical_clustering_to_dendrogram(slc)


def linkage_ultrametric(distances, method="single"):
    """Computes the ultrametric corresponding to the dendrogram produced by
    the given linkage method.
    
    Args:
        distances (ndarray): A condensed distance matrix representing a 
            dissimilarity.
        method (str, optional): The linkage method to use. Must be one of the
            methods used by ``scipy.cluster.hierarchical.linkage``. Defaults 
            to 'single'.

    Returns:
        (ndarray): The ultrametric as a condensed distance matrix.
    """
    clustering = _hierarchy.linkage(distances, method=method)
    dendrogram = hierarchical_clustering_to_dendrogram(clustering)
    return dendrogram_to_ultrametric(dendrogram)


def triangle_inequality_indices(n):
    """Generates the indices to check for the triangle inequality.

    Args:
        n (int): The number of points in the metric.

    Returns:
        i,j,k (ndarray): 3 ndarrays of equal size, so that if m is the metric
        matrix, ``m[i[x], j[x]] <= m[i[x], k[x]] + m[j[x], k[x]]`` must hold 
        for all x in [1,...,len(i)]. If all of these hold, m is metric.

    All (n choose 3) triples of points must obey the triangle 
    inequality. Given a (potentially) metric :math:`n \\times n` matrix, not
    all (n choose 3) triples of entries need to be checked -- some are
    redundant. For example, we need to ensure that 
    :math:`d_{12} \\leq d_{13} + d_{23}`, but do not need to check 
    :math:`d_{12} \\leq d_{23} + d_{13}` or 
    :math:`d_{21} \\leq d_{13} + d_{23}`, etc., by symmetry.

    For every pair :math:`(i,j)`, we need to check that 
    :math:`d_{ij} \\leq d_{ik} + d_{kj}` for every :math:`k \\neq i`, 
    :math:`k \\neq j`. This function generates exactly those indices necessary
    to verify metricity.
    """
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
    ultrametric.

    Args:
        m (ndarray): The condensed metric array.

    Returns:
        i,j,k (ndarray): Arrays of indices for which 
        ``m[i[x]] <= min(m[j[x]], m[k[x]])`` and ``m[j[x]] != m[k[x]]``.

    If :math:`m_{ij} \\leq \\min(m_{ik}, m_{jk})`, and 
    :math:`m_{ik} \\neq m_{jk}`, then the triple violates the ultrametric
    inequality. This function looks at the minimal set of distances necessary
    to ensure ultrametricity, and verifies the inequality for each of them.

    Note that because equality is checked for, this function may return triples
    which are practically ultrametric, but differ due to floating point issues.
    """
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


def cophenetic_correlation(x, y):
    x = _np.asarray(x)
    y = _np.asarray(y)

    d = _np.sqrt(_np.sum((x - x.mean())**2) * _np.sum((y - y.mean())**2))
    n = _np.sum((x - x.mean())*(y - y.mean()))
    
    return n/d

def rammal_ultrametricity(m):
    """Computes the Rammal ultrametricity degree."""
    msd = linkage_ultrametric(m, method='single')
    return _np.sum(m - msd) / _np.sum(m)
