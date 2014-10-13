import numpy as _np
import networkx as _nx

def min_max_l_infinity(profile):
    """Minimizes the max l-infinity distance from the consensus to the
    ultrametrics in the profile."""
    sup = _np.max(profile, axis=0)
    inf = _np.min(profile, axis=0)
    return sup - 1./2*_np.max(sup - inf)

def best_in_profile(profile, cost_function):
    """Applies the cost function to each ultrametric in the profile, and 
    returns the best. The cost function must return a nxn consensus ultrametric
    and return a scalar value representing its cost."""
    profile_iter = (row for row in profile)
    return min(profile_iter, key=cost_function)

def sum_of_squared_errors(profile, consensus):
    """Computes the sum of squared errors between the profile and the 
    consensus."""
    return _np.sum((profile - consensus)**2)

def strict_consensus_clusters(dendrograms):
    """Clusters appearing in the strict tree."""
    clusters = (all_clusters(dendrogram) for dendrogram in dendrograms)
    common = reduce(operator.and_, clusters)
    return sorted(common, key=len, reverse=True)

def majority_rule_clusters(dendrograms):
    """Clusters appearing in the majority rule tree."""
    clusters = (all_clusters(dendrogram) for dendrogram in dendrograms)
    
    cluster_count = {}
    
    for dendrogram in clusters:
        for cluster in dendrogram:
            if not cluster in cluster_count:
                cluster_count[cluster] = 1
            else:
                cluster_count[cluster] += 1

    half = len(dendrograms)/2
    
    return sorted([x for x,y in cluster_count.items() if y > half], key=len)

def tree_from_clusters(clusters):
    """Rebuilds a tree from clusters."""
    clusters = sorted(clusters, key=len, reverse=True)
    
    # the first entry is the root, and contains all leaf nodes
    clusters.extend((x,) for x in clusters[0])
    tree = _nx.DiGraph()
    root = clusters.pop(0)
    tree.add_node(root)
    
    while clusters:
        node = clusters.pop(0)
        # find the tightest possible parent node
        candidates = [x for x in tree if set(x).issuperset(node)]

        try:
            parent = min(candidates, key=len)
        except ValueError:
            raise ValueError("No candidates. Perhaps the input clusters were "
                             "not compatible?")

        tree.add_edge(parent, node)
        
    return tree
