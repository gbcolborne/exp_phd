#! -*- coding:utf-8 -*-
"""
Classes for (word,context,position) cooccurrence frequency tensors and
(word,context) cooccurrence frequency matrices.

Classes:
CoocMatrix -- sparse (word,context) cooccurrence frequency matrix
CoocTensor -- (word, context, position) cooccurrence frequency tensor
  containing a sparse (word,context) cooccurrence frequency matrix for
  each position
"""

import sys
from collections import defaultdict
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, hstack, isspmatrix

class CoocMatrix:
    """
    Sparse (word,context) cooccurrence frequency matrix.

    Methods:
    __init__ -- initialize using a precomputed frequency matrix
    get_expected_cooc_freq -- compute expected cooccurrence frequencies
      from observed cooccurrence frequencies
    weight -- weight cooccurrence frequencies using an association measure.
    transform -- apply transformation to cooccurrence frequencies.
    """

    def __init__(self, freq):
        """
        Initialize using a pre-computed matrix of observed cooccurrence frequencies.

        Arguments:
        freq -- a sparse matrix containing observed (word, context)
          cooccurrence frequencies
        """

        if not isspmatrix(freq):
            sys.exit('ERROR: matrix must be sparse.')
        self.freq = freq
        # Set attribute is_raw to True. It will be set to false if the
        # frequencies are weighted or transformed.
        self.is_raw = True

    def weight(self, measure):
        """
        Compute association measure from observed cooccurrence frequencies.

        Given a matrix of observed cooccurrence frequencies, compute
        association scores using one of the simple association
        measures defined in (Evert, 2007, ch. 4): MI, MI^k, local-MI,
        simple-LL, t-score, z-score.

        Return matrix of association scores.

        Arguments:
        am -- association measure ('MI', 'MI2', 'MI2', 'local-MI',
          't-score', 'z-score', 'simple-ll' or 'None')
        """

        if measure not in ['None', 'MI', 'MI2', 'MI3', 'local-MI', 't-score', 
                      'z-score', 'simple-ll']:
            sys.exit('ERROR: {} not a valid measure.'.format(measure))
        E = self.get_expected_cooc_freq()
        if measure in ['MI', 'MI2', 'MI3']:
            if measure == 'MI':
                k = 1
            elif measure == 'MI2':
                k = 2
            elif measure == 'MI3':
                k = 3
            self.freq.data = np.log2(self.freq.data**k / E.data)
        elif measure == 'local-MI':
            self.freq.data = self.freq.data * np.log2(self.freq.data / E.data)
        elif measure == 't-score':
            self.freq.data = (self.freq.data - E.data) / np.sqrt(self.freq.data)
        elif measure == 'z-score':
            self.freq.data = (self.freq.data - E.data) / np.sqrt(E.data)
        elif measure == 'simple-ll':
            freq_is_low = np.where(self.freq.data<E.data)
            self.freq.data = 2*(self.freq.data*np.log(self.freq.data/E.data)-(self.freq.data-E.data))
            # Make values where O < E negative (because simple-LL
            # returns positive values both where O>>E and where O<<E).
            self.freq.data[freq_is_low] *= -1
        # Make non-negative
        self.freq.data = self.freq.data.clip(min=0)
        self.freq.eliminate_zeros()
        # Set is_raw to False to signal that the frequencies have been weighted.
        self.is_raw = False

    def transform(self, transformation):
        """ 
        Apply transformation to cooccurrence frequencies. 

        Arguments:
        transformation -- name of transformation ('log', 'sqrt', 'sigm' or 'None')
        """

        if transformation not in ['None','log', 'sqrt', 'sigm']:
            sys.exit('ERROR: {} not a valid transformation.'.format(transformation))
        if transformation == 'sqrt':
            self.freq = self.freq.sqrt()
        elif transformation == 'log':
            self.freq = self.freq.log1p()
        elif transformation == 'sigm':
            self.freq.data = 1.0 / (1.0 + np.exp(-1.0 * self.freq.data))
        # Set is_raw to False to signal that the frequencies have been weighted.
        self.is_raw = False

    def get_expected_cooc_freq(self):
        """
        Compute expected cooccurrence frequencies from observed
        cooccurrence frequencies.

        Given a matrix of observed cooccurrence frequencies (O), compute
        expected cooccurrence frequencies (E).  This method of computing E
        is different than the method described by Evert (2007, ch. 4). It
        uses the row and column sums of O, whereas Evert's version is
        based on the marginal (unigram) frequencies of words, the number
        of positions in the context window, and the number of tokens in
        the corpus.

        Return sparse matrix containing expected cooccurrence frequencies.
        """
        if not self.is_raw:
            msg = 'ERROR: expected cooccurrence frequencies should be computed '
            msg += 'from the raw cooccurrence frequencies (before they have ' 
            msg += 'been weighted or transformed).'
            sys.exit(msg)
        # Get column sums and row sums
        csums = np.array(self.freq.sum(0))[0]
        rsums = np.array(self.freq.sum(1)).reshape(1,self.freq.shape[0])[0]
        # Get sum of all values
        sum_all_freqs = csums.sum()
        # Get indices of non-zero values in O
        nzr, nzc = self.freq.nonzero()
        # Make copy of O
        E = self.freq.copy()
        # Compute E
        E.data = rsums[nzr] * csums[nzc] / sum_all_freqs
        return E

class CoocTensor:
    """ 
    A (word, context, position) cooccurrence frequency tensor.

    The tensor contains a sparse (word, context) cooccurrence
    frequency matrix for each position in a fixed-width context
    window.

    Note: For now, this class assumes that the vocabulary of contexts
    is the same as the vocabulary of words.

    Methods: 
    __init__ -- scan a corpus and compute tensor for a given window
      size
    __getitem__ -- get matrix for a given position
    compute_slices -- scan a corpus and compute tensor for a given
      window size
    sum_slices -- compute sum of slices (matrices) for a given set of
      positions
    to_matrix -- generate a sparse (word, context) cooccurrence
      frequency matrix for a given context window
    check_corpus -- make sure the list of word identifiers provided
      has the required properties
    get_vocab_size -- get the size of the vocabulary (number of target
    words)
    """

    def __init__(self, corpus, win_size):
        """
        Scan corpus and compute (word, context position) cooccurrence
        frequency tensor for a given window size.

        The tensor contains a sparse (word, context) cooccurrence
        frequency matrix for each position (signed distance from the
        target word) in a fixed-width context window. For instance,
        each value M_ij in the matrix M for position -1 would indicate
        the frequency at which context j was observed one word to the
        left of word i.

        Arguments:
        corpus -- a list containing an integer representation of every
          word in a corpus (-1 for OOV)
        win_size -- width of the context window (number of words)
        """

        self.corpus = corpus
        self.check_corpus()
        self.corpus_size = len(self.corpus)
        self.vocab_size = self.get_vocab_size()
        # Create list of positions (signed distances from the target word)
        self.win_size = win_size
        self.positions = range(1,self.win_size+1)
        self.positions = np.asarray([-x for x in self.positions[::-1]] + self.positions)
        self.compute_slices()
        self.slice_shape = (self.vocab_size, self.vocab_size)


    def __getitem__(self, key):
        """ Given a position, return cooccurrence matrix for this position """
        return self.slices[key]

    def compute_slices(self):
        """ Compute cooccurrence matrices for each position in a context window. """
        self.slices = {}
        for p in self.positions:
            print 'Calcul de la matrice de cooccurrence pour la position {}...'.format(p)
            start = max(p,0)
            stop = min(self.corpus_size, self.corpus_size+p)
            contexts = self.corpus[start:stop]
            padding = [-1] * abs(p)
            if p < 0:
                contexts = np.hstack((padding,contexts))
            else:
                contexts = np.hstack((contexts,padding))
            contexts = np.asarray(contexts)
            # Given the list of words and the list of contexts at
            # position p, extract the indices where neither the word
            # nor its context are OOV.
            not_OOV = (self.corpus != -1) * (contexts != -1)
            # Compute cooccurrence frequencies
            coocs = zip(self.corpus[not_OOV], contexts[not_OOV])
            cooc_freqs = defaultdict(int)
            for (f, c) in coocs:
                cooc_freqs[(f, c)] += 1
            # Create sparse cooccurrence frequency matrix
            data = []
            row = []
            col = []
            for ((r,c), v) in cooc_freqs.items():
                data.append(v)
                row.append(r)
                col.append(c)
            pslice = coo_matrix((data,(row,col)), shape=(self.vocab_size, self.vocab_size), 
                                dtype=float)
            self.slices[p] = pslice.tocsr()

    def sum_slices(self, positions, win_shape):
        """
        Compute sum of slices (matrices) for a given set of positions.

        Given a word-word-position cooccurrence tensor, a set of positions 
        (signed distances from focus word), and a window shape (rect or tri), 
        return weighted sum of slices for specified positions.

        Return sparse matrix containing sum of slices.

        Arguments:
        positions -- a set of positions (signed distances from the target word)
        win_shape -- shape of context window ('rect' or 'tri'), i.e. a
          function that determines the weight of a cooccurrence based
          on the distance between the target word and the context
          word.
        """

        if win_shape not in ['rect', 'tri']:
            sys.exit('ERROR: {} not a valid shape.'.format(win_shape))
        # Initialize matrix
        mat = csr_matrix(self.slice_shape, dtype=float)
        # Compute weights
        if win_shape == 'rect':
            weights = [1. for p in positions]
        elif win_shape == 'tri':
            weights = [1./abs(p) for p in positions]
        # Compute weighted sum of slices
        for i in range(len(positions)):
            p = positions[i]
            w = weights[i]
            mat = mat + (self.slices[p] * w)
        return mat

    def to_matrix(self, win_size, win_type, win_shape, am='None', tr='None'):
        """
        Generate a sparse (word, context) cooccurrence frequency
        matrix for a given context window.

        Given a window size, type, and shape, generate a cooccurrence
        frequency matrix by matricizing the tensor and creating an
        instance of the CoocMatrix class. An association-based
        weighting scheme and/or a transformation can also be applied.
        
        Return sparse cooccurrence matrix.

        Arguments:
        win_size -- width of the context window (cannot be greater
          than the width used to compute the tensor)
        win_type -- type (or direction) of the context window ('G' for
          left, 'D' for right, 'G+D' or 'G&D')
        win_shape -- shape of the context window ('rect' or 'tri'),
          i.e. a function that determines the weight of a cooccurrence
          based on the distance between the target word and the
          context word.
        am -- association measure used for weighting ('MI', 'MI2',
          'MI2', 'local-MI', 't-score', 'z-score', 'simple-ll' or
          'None')
        tr -- transformation applied to the weighted cooccurrence
          frequencies ('log', 'sqrt', 'sigm' or 'None')
        """

        if win_size > self.win_size:
            sys.exit('ERROR: win_size is greater than the window size of the tensor.')
        if win_type not in ['G', 'D', 'G+D', 'G&D']:
            sys.exit('ERROR: {} not a valid window type.'.format(win_type))
        if win_shape not in ['rect', 'tri']:
            sys.exit('ERROR: {} not a valid window shape.'.format(win_shape))
        # Matricize
        if win_type == 'G':
            positions = range(-win_size, 0)
            mat = self.sum_slices(positions, win_shape)
        elif win_type == 'D':
            positions = range(1,win_size+1)
            mat = self.sum_slices(positions, win_shape)
        elif win_type == 'G+D':
            positions = range(-win_size, 0) + range(1,win_size+1)
            mat = self.sum_slices(positions, win_shape)
        elif win_type == 'G&D':
            left_positions = range(-win_size, 0)
            right_positions = range(1,win_size+1)
            mat = (hstack((self.sum_slices(left_positions, win_shape), 
                           self.sum_slices(right_positions, win_shape)), format='csr'))
        # Create instance of CoocMatrix class
        mat = CoocMatrix(mat)
        # Weight
        mat.weight(am)
        # Transform
        mat.transform(tr)
        # Return the attribute containing the actual matrix
        return mat.freq
    
    def check_corpus(self):
        """
        Make sure the corpus is well-formed.

        The corpus should be a list containing non-negative integer
        representations of words (and -1 for out-of-vocabulary words
        if necessary). All integers between 0 and the highest word
        identifier should be present in the corpus, so that the (word,
        context) cooccurrence frequency matrices we will build contain
        no empty rows or columns.

        Raise a ValueError if the corpus is not well-formed.
        """

        if type(self.corpus) != np.ndarray:
            self.corpus = np.asarray(self.corpus, dtype=int)
        # The minimum value in the corpus should be 0 or -1.
        sorted_uniq_IDS = sorted(set(self.corpus))
        smallest_ID = sorted_uniq_IDS[0]
        if smallest_ID == -1:
            sorted_uniq_IDS = sorted_uniq_IDS[1:]
        elif smallest_ID < -1:
            msg = 'ERROR: smallest word ID ({}) should be 0 or -1 (for OOV).'.format(smallest_ID)
            msg += ' Make sure word IDs are assigned sequentially, starting at 0,'
            msg += ' and only to words that are actually in the corpus.'
            raise ValueError(msg)
        # Add 1 to the highest word ID to get the size of the vocabulary
        vocab_size = max(self.corpus) + 1
        # Make sure the word IDs are consecutive (to avoid making the
        # cooccurrence matrices larger than necessary).
        missing_IDS = set(range(vocab_size)).difference(sorted_uniq_IDS)
        if len(missing_IDS):
            msg = 'ERROR: The following word IDs do not occur in the corpus: '
            msg += ', '.join([str(i) for i in missing_IDS])
            msg += '. Make sure word IDs are assigned sequentially, starting at 0,'
            msg += ' and only to words that are actually in the corpus.'
            raise ValueError(msg)

    def get_vocab_size(self):
        """ 
        Get highest word identifier and add 1 to obtain the size of
        the vocabulary (assuming the corpus is well-formed).

        Return the size of the vocabulary.
        """

        return max(self.corpus) + 1
