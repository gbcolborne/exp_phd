#! -*- coding: utf-8 -*-
""" 
This module contains a class used to create a k-nearest-neighbour graph. 

Classes:
KNNGraph -- A class to create a k-nearest-neighbour graph.
"""

import sys
import numpy as np
import networkx

class KNNGraph:
    """ 
    A class to create a k-nearest-neighbour graph.
    
    Methods:
    __init__ -- initialize graph from a distance matrix
    """

    def __init__(self, dist, gtype, k):
        """ 
        Create k-nearest-neighbour graph from a distance matrix.

        Arguments:
        dist -- a distance matrix
        gtype -- type of graph (directed, symmetric or mutual)
        k -- k (number of nearest neighbours used for computing graph)

        Returns:
        graph -- a Networkx graph object
        """
        # Check args
        if gtype not in ['symmetric', 'mutual', 'directed']:
            msg = 'Error: Unrecognized graph type "{}".'.format(gtype)
            sys.exit(msg)
        self.gtype = gtype
        self.k = int(k)
        self.nb_nodes = dist.shape[0]
        # Make sure distance matrix is symmetric
        if dist.shape[1] != dist.shape[0]:
            msg = 'Error: distance matrix is not square.'
            sys.exit(msg)
        if not (dist.transpose() == dist).all():
            msg = 'Error: distance matrix is not symmetric.'
            sys.exit(msg)
        # Set diagonal to infinity
        np.fill_diagonal(dist, np.inf)            
        # Get edges of graph, i.e. a list of (node, neighbour) tuples.
        if gtype == 'symmetric' or gtype == 'directed':
            n1 = range(self.nb_nodes) * self.k
            n2 = dist.argsort()[:,:self.k].T.flatten().tolist()
        elif gtype == 'mutual':
            # Create matrix in which cell ij will be True if w_j is a
            # knn of w_i
            is_knn = np.zeros((self.nb_nodes, self.nb_nodes),dtype=bool) 
            is_knn[(range(self.nb_nodes) * self.k), dist.argsort()[:,:k].T.flatten()] = 1
            # Multiply by transpose so that cell ij is True only if
            # w_j is a knn of w_i and vice-versa.
            is_knn = is_knn * is_knn.T 
            n1, n2 = np.where(np.triu(is_knn,k=1))
        # Get edge weights
        weights = dist[n1,n2]
        # Create graph
        if gtype == 'directed':
            self.graph = networkx.DiGraph(name='k-NN graph')
        elif gtype == 'symmetric' or gtype == 'mutual':
            self.graph = networkx.Graph(name='k-NN graph')
        nodes = range(dist.shape[0])
        self.graph.add_nodes_from(nodes)
        edges = zip(n1, n2, [{'weight':w} for w in weights])
        self.graph.add_edges_from(edges)


        
    
    
