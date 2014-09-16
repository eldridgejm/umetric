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


def is_ultrametric(m):
    """Given a metric matrix m, verifies the three-point condition. Naive
    implementation takes O(n choose 3) time."""
    n = m.shape[0]

    if not _np.allclose(_np.diag(m), 0):
        return False
    
    for  i in xrange(n):
        for j in xrange(n):
            if m[i,j] > _np.min(_np.max(_np.column_stack((m[i], m[j])), 
                                axis=0)):
                return False
            
    return True

