#! -*- coding:utf-8 -*-
""" Evaluate count-based distributional semantic models. """
import sys, os, codecs, argparse
from sklearn.metrics.pairwise import pairwise_distances
import Corpus, CoocTensor, KNNGraph, eval_utils

def join_strings_and_append_to_file(strings, filename):
    """ 
    Join strings in list using comma as delimiter, append to file at
    path provided.
    """

    with open(filename, 'a') as f:
        f.write(','.join(strings)+'\n')

if __name__ == "__main__":
    dsc = ("Évaluer différentes paramétrisations de l'analyse distributionnelle "
           "au moyen des données de référence.")
    parser = argparse.ArgumentParser(description=dsc)
    parser.add_argument('-s', '--exclude_stop', action="store_true",
                        help='Exclure les mots vides des mots-cibles')
    parser.add_argument('lang', choices=['EN', 'FR'])
    parser.add_argument('corpus', help="Chemin du corpus (fichier texte, une phrase par ligne)")
    parser.add_argument('output', help="Chemin du dossier qui contiendra les résultats")
    args = parser.parse_args()

    # Check args
    if os.path.isdir(args.output):
        sys.exit('ERREUR: Le dossier {} existe déjà.'.format(args.output))
    PATH_OUT_MAP = os.path.abspath(args.output) + '/res_MAP.csv'
    PATH_OUT_GRAPH = os.path.abspath(args.output) + '/res_graph.csv'
    os.mkdir(os.path.abspath(args.output))

    # Check that the evaluation data are where they should be
    PATH_REF = 'data/ref_{}.csv'.format(args.lang)
    if not os.path.isfile(PATH_REF):
        e = "Erreur: le fichier {} n'existe pas.".format(PATH_REF)
        sys.exit(e)

    # Load stop words if necessary
    stopwords = None
    if args.exclude_stop:
        stopwords = set()
        path = 'data/stop_{}.txt'.format(args.lang)
        with codecs.open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if len(line) and line[0] != '#':
                    stopwords.add(line)

    # Prepare corpus, compute vocab
    print u'\nLecture du corpus et calcul du vocabulaire...'
    corpus = Corpus.Corpus(args.corpus)
    print u'Terminée. Le corpus contient {} mots.'.format(corpus.size)

    # Select target words
    print u'\nIdentification des mots-cibles...'
    nb_targets = 10000
    targets = []
    for word in corpus.most_freq_words(apply_filters=True, stopwords=stopwords):
        targets.append(word)
        if len(targets) == nb_targets:
            break
    target2id = dict([(w,i) for (i,w) in enumerate(targets)])
    print u'Terminée. {} mots-cibles identifiés.'.format(len(targets))

    # Process evaluation data
    print u'\nLecture des données de référence...'
    ref = eval_utils.process_ref(PATH_REF, target_words=targets)
    print u'Terminée.'
    print ''
    print '=' * 25
    print u'Catégorie    Nb relations'
    for r in sorted(ref.keys(), key=len):
        nb_rels = sum(len(v) for v in ref[r].itervalues())
        print '{0: <13}{1}'.format(r, nb_rels)
    print '=' * 25

    # Transform corpus into list of word IDs
    print u'\nPréparation du corpus...'
    corpus = corpus.list_word_IDs(target2id, OOV_ID=-1)
    print u'Terminée.'

    # Define parameter values to be tested
    max_win_size = 10
    win_sizes = range(max_win_size,0,-1)
    win_types = ['G&D', 'G+D']
    win_shapes = ['rect', 'tri']
    weighting_schemes = [('None', 'None'), ('None', 'log'), ('MI', 'None'), 
                         ('MI2', 'None'), ('MI3', 'None'), ('local-MI', 'log'), 
                         ('z-score', 'sqrt'), ('t-score', 'sqrt'), ('simple-ll', 'log')]
    model_params = []
    for wt in win_types:
        for ws in win_shapes:
            for cs in win_sizes:
                for (we,tr) in weighting_schemes:
                    model_params.append((wt,ws,cs,we,tr))

    # Define graph parameter values
    gtypes = ['directed', 'symmetric', 'mutual']
    kvals = [1, 2, 4, 8, 16, 32, 64, 128]
    graph_params = []
    for gtype in gtypes:
        for k in kvals:
            graph_params.append((gtype, k))

    # Compute (word, context, position) cooccurrence frequency tensor
    print u'\nCalcul du tenseur de cooccurrence...'
    tensor = CoocTensor.CoocTensor(corpus, max_win_size)
    print u'Terminé.'

    # Write header in output files
    ref_rels = sorted(ref.keys(), key=len)
    param_names = ['win_type','win_shape','win_size','weighting_scheme']
    header = param_names[:] 
    header += ['MAP({})'.format(rel) for rel in ref_rels]
    join_strings_and_append_to_file(header, PATH_OUT_MAP)     
    param_names.append('graph_type')
    param_names.append('k')
    header = param_names[:]
    header.append('nb_edges')
    for rel in ref_rels:
        header.append('P({})'.format(rel))
        header.append('R({})'.format(rel))
    join_strings_and_append_to_file(header, PATH_OUT_GRAPH)     

    # Start experiment
    num_models = len(model_params)
    num_graphs = num_models * len(graph_params)
    model_counter = 0
    graph_counter = 0
    for (wt, ws, cs, we, tr) in model_params:
        model_counter += 1

        # Build model
        msg = u'\nConstruction du modèle {} de {}...'.format(model_counter,num_models)
        msg += u'\n  - type de fenêtre = {}'.format(wt)
        msg += u'\n  - taille de fenêtre = {}'.format(cs)
        msg += u'\n  - forme de fenêtre = {}'.format(ws)
        msg += u"\n  - mesure d'association = {}".format(we)
        msg += u'\n  - transformation = {}'.format(tr)
        print msg
        model = tensor.to_matrix(cs, wt, ws, we, tr)

        # Compute MAP on all data sets
        print u'Évaluation du modèle...'
        dist = pairwise_distances(model, metric='cosine')
        MAP_strings = []
        for rel in ref_rels:
            MAP = eval_utils.compute_MAP_from_dist_mat(dist, ref[rel], target2id)
            MAP_strings.append('{:.4f}'.format(MAP))
        print u'Terminée.'

        # Write results of model evaluation (MAP) 
        param_strings = [wt, ws, str(cs), '{}+{}'.format(we, tr)]
        join_strings_and_append_to_file(param_strings + MAP_strings, PATH_OUT_MAP)     
        
        # Compute and evaluate graphs
        for (gtype,k) in graph_params:
            graph_counter += 1
            print u'Construction du graphe {} de {}...'.format(graph_counter, num_graphs)            
            print u'- type de graphe = {}'.format(gtype)
            print u'- k = {}'.format(k)
            # Compute graph
            graph = KNNGraph.KNNGraph(dist, gtype=gtype, k=k).graph
            # Initialize list of result strings
            result_strings = []
            result_strings.append(str(graph.number_of_edges()))
            # Compute metrics on each of the sets (and subsets) of reference data
            print u'Évaluation du graphe...'
            for rel in ref_rels:
                p,r = eval_utils.evaluate_graph(graph, ref[rel], target2id)
                result_strings.append('{:.4f}'.format(p))
                result_strings.append('{:.4f}'.format(r))
            print 'Terminée.'
            param_strings = [wt, ws, str(cs), '{}+{}'.format(we, tr), gtype, str(k)]
            join_strings_and_append_to_file(param_strings + result_strings, PATH_OUT_GRAPH)     

    print u'\n\n\nVoir les résultats dans {}.\n'.format(os.path.abspath(args.output))
