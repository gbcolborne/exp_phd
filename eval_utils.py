#! -*- coding: utf-8 -*-
""" 
A module containing functions used to evaluate distributional semantic
models.

Functions:
compute_MAP_from_dist_mat -- compute MAP from a distance matrix on
  evaluation data
evaluate_graph -- compute accuracy of neighbourhoods in graph on
  evaluation data
process_ref -- process file containing evaluation data (semantic
  relations)
"""

import codecs, sys
from collections import defaultdict
import numpy as np
import networkx

def evaluate_graph(graph, eval_data, word_to_ID):
    """ 
    Compute accuracy of neighbourhoods in graph on evaluation data. 

    Arguments:
    graph -- a NetworkX graph object
    eval_data -- a dictionary that maps words to a list of semantically
      related words
    word_to_ID -- a dictionary that maps words to their identifier
      (row and column number in the distance matrix)
    """
    if not isinstance(graph, networkx.Graph) and not isinstance(graph, networkx.DiGraph):
        sys.exit('Error: graph is not a NetworkX graph or digraph object.')
    precision = []
    recall = []
    for w in eval_data.keys():
        # Get neighbours
        nbrs = graph.neighbors(word_to_ID[w])
        # Get related terms
        rt = [word_to_ID[t] for t in eval_data[w]]
        # Compute metrics 
        if len(nbrs) == 0:
            p,r = 0.,0.
        else:
            # Compute number of true positives
            tp = len(set(nbrs).intersection(rt))
            # Compute metrics
            p = float(tp)/len(nbrs)
            r = float(tp)/len(rt)
        precision.append(p)
        recall.append(r)
    # Compute averages
    nb_cases = len(eval_data)
    mean_precision = sum(precision)/nb_cases
    mean_recall = sum(recall)/nb_cases
    return mean_precision, mean_recall


def compute_MAP_from_dist_mat(dist, eval_data, word_to_ID, dist_is_sim=False):
    """
    Compute MAP from a word-word distance matrix on evaluation data.

    Arguments:
    dist -- a word-word distance matrix (or similarity matrix if
      dist_is_sim is True)
    eval_data -- a dictionary that maps words to a list of semantically
      related words
    word_to_ID -- a dictionary that maps words to their identifier
      (row and column number in the distance matrix)
    dist_is_sim -- if True, dist will be considered a similarity
      matrix rather than a distance matrix

    Notes:
    Diagonal of dist will be set to inf (or 0 if dist_is_sim).
    """
    if dist_is_sim:
        np.fill_diagonal(dist, 0)
    else:
        np.fill_diagonal(dist, np.inf)
    AP = []
    for q in eval_data:
        # Get ID of query and related terms
        if q not in word_to_ID:
            print u'WARNING: "{}" not in word_to_ID.'.format(q),
            print u' AP for this query will be considered to be 0.'
            AP.append(0)
            continue
        q_id = word_to_ID[q]
        # Compute similarity rank of all target words wrt query
        if dist_is_sim:
            nbr_ranks = dist[q_id].argsort()[::-1].argsort() + 1
        else:
            nbr_ranks = dist[q_id].argsort().argsort() + 1
        # Get rank of all related terms
        rt_ranks = []
        for rt in eval_data[q]:
            if rt not in word_to_ID:
                print u'WARNING: "{}" not in word_to_ID.'.format(rt),
                print u' Precision for this related term will be considered to be 0.'
                continue
            rt_id = word_to_ID[rt]
            rt_ranks.append(nbr_ranks[rt_id])
        precisions = np.arange(1,len(rt_ranks)+1,dtype=float)/sorted(rt_ranks)
        # Compute average precision
        ap_q = precisions.sum()/len(eval_data[q])
        # Append to list of AP values
        AP.append(ap_q)
    MAP = np.mean(AP)
    return MAP

def process_ref(filename, target_words=None):
    """
    Process file containing evaluation data (semantic relations)

    Each line in this file should contain one relation in the
    following format, where relation should be 'QSYN', 'ANTI', 'HYP',
    'DRV' or 'FRM': word1,pos1,word2,pos2,relation. The POS can be
    left empty.

    The first line of the file (header) is skipped.

    Arguments:
    filename -- path of the file containing the evaluation data
    target_words -- a list of target words. If provided, relations
      containing a word which is not in this list will be discarded.

    Returns: a dictionary that maps each of the 5 relations (as well
    as "TOUTES" for all relations combined except for FRM) to a
    dictionary which maps queries to a list of related terms for that
    relation. It also maps each POS to a dictionary that maps words to
    related terms (excluding FRM relations) that have the same POS.
    """
    if target_words and type(target_words) != set:
        # Use a set rather than a list for faster lookup
        target_words = set(target_words)
    relations = []
    with codecs.open(filename, encoding='utf-8') as f:
        # Skip the header
        f.readline()
        for line in f:
            w1, p1, w2, p2, rel = line.strip().split(',')
            relations.append((w1,p1,w2,p2,rel))
    ref = {}
    ref['TOUTES'] = defaultdict(list)
    for (w1,p1,w2,p2,rel) in relations:
        if target_words:
            w1_missing = w1 not in target_words
            w2_missing = w2 not in target_words
            if w1_missing or w2_missing:
                print u'- ATTENTION : Relation ({},{}) exclue.'.format(w1,w2),
                if w1_missing and w2_missing:
                    print u'Les deux mots sont absents des mots-cibles.'
                elif w1_missing:
                    print u'Le 1er mot est absent des mots-cibles.'
                else:
                    print u'Le 2e mot est absent des mots-cibles.'
                continue
        if rel not in ref:
            ref[rel] = defaultdict(list)
        ref[rel][w1].append(w2)
        if rel != 'FRM':
            ref['TOUTES'][w1].append(w2)
        if p1 == p2 and rel != 'FRM':
            if p1 not in ref:
                ref[p1] = defaultdict(list)
            ref[p1][w1].append(w2)
    return ref
