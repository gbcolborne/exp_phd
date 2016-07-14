#! -*- coding: utf-8 -*-
""" This module contains a class called Corpus to handle corpora. """
import codecs, re, collections

class Corpus:
    """
    A class to handle corpora.

    Given the path of a corpus (single text file containing one
    sentence per line), this class can be used to compute the
    vocabulary of the corpus, generate the sentences it contains, and
    carry out other corpus-related tasks.

    Methods:
    __init__ -- initialize and compute vocab
    make_vocab -- map words to their frequency in the corpus
    most_freq_words -- generate words in reverse order of frequency
    stream_sents -- generate sentences from the corpus 
    list_words -- return corpus as one long list of words
    list_word_IDs -- return corpus as a list of word identifiers given
      a mapping of words to identifiers
    """

    def __init__(self, path_corpus):
        """ Initialize given the path of a corpus, compute vocab. """
        self.path_corpus = path_corpus
        self.make_vocab()
        self.size = sum(self.vocab.values())

    def stream_sents(self):
        """ 
        Generate sentences from the corpus. 

        For each sentence (line) in the corpus, split at whitespace,
        and yield list of tokens.
        """

        with codecs.open(self.path_corpus, 'r', encoding='utf-8') as f:
            for line in f:
                yield line.split()

    def list_words(self):
        """ Return corpus as one long list of words. """
        words = []
        for sent in self.stream_sents():
            words += sent
        return words

    def list_word_IDs(self, word_to_ID, OOV_ID=-1):
        """ 
        Return corpus as a list of word identifiers.

        Given a mapping of words to identifiers, replace each word in
        the corpus by its identifier (or OOV_ID if unknown).

        Return a list of word identifiers.
        """
        vocab = set(word_to_ID.keys())
        corpus = []
        for sent in self.stream_sents():
            for word in sent: 
                if word in vocab:
                    corpus.append(word_to_ID[word]) 
                else: 
                    corpus.append(OOV_ID) 
        return corpus

    def make_vocab(self):
        """ Map words to their frequency in the corpus. """
        self.vocab = collections.defaultdict(int)
        for sent in self.stream_sents():
            for word in sent:
                self.vocab[word] += 1

    def most_freq_words(self, apply_filters=False, stopwords=None):
        """ Generate words in reverse order of frequency.
        
        Arguments:

        apply_filters -- if True, filters are applied to return only
          words that satisfy certain criteria (e.g. they contain only
          letters, digits or hyphens).
        stopwords -- a set of stop words which will be discarded
        """

        if stopwords and type(stopwords) != set:
            stopwords = set(stopwords)
        if not stopwords:
            stopwords = set()
        if apply_filters:
            # Define illegal characters: all punctuation except the
            # hyphen, as well as all characters in the Latin-1
            # encoding which are neither a letter nor a digit.
            illegal_chars = (ur'[\x21\x22\x23\x24\x25\x26\x27\x28\x29\x2a\x2b\x2c\x2e\x2f\x3a\x3b\x3c'
                ur'\x3d\x3e\x3f\x40\x5b\x5c\x5d\x5e\x5f\x60\x7b\x7c\x7d\x7e\xa0\xa1\xa2\xa3\xa4\xa5'
                ur'\xa6\xa7\xa8\xa9\xaa\xab\xac\xad\xae\xaf\xb0\xb1\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9'
                ur'\xba\xbb\xbc\xbd\xbe\xbf\xc1\xc3\xc4\xc5\xcc\xcd\xd0\xd2\xd3\xd5\xd7\xd8\xda\xdd'
                ur'\xde\xdf\xe1\xe3\xe4\xe5\xec\xed\xf0\xf2\xf3\xf5\xf7\xf8\xfa\xfd\xfe\xff]')
            illegal_chars_pattern = re.compile(illegal_chars)
            for word in sorted(self.vocab, key=self.vocab.__getitem__, reverse=True):
                if not len(word):
                    continue
                elif word in stopwords:
                    continue
                elif re.search(illegal_chars_pattern, word):
                    continue
                elif word[0] == '-':
                    continue
                elif word[-1] == '-':
                    continue
                else:
                    yield word
        else:
            for word in sorted(self.vocab, key=self.vocab.__getitem__, reverse=True):
                yield word
            
