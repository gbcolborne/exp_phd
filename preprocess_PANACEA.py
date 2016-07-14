#! -*- coding:utf-8 -*-
""" Preprocessing script for the PANACEA corpus. """
import sys, os, re, argparse, codecs, unicodedata
from xml.dom.minidom import parseString
from unidecode import unidecode
import treetaggerwrapper as ttw

def replace_oelig(string):
    """ Replace oe ligatures (which are not present in Latin-1). """
    string = re.sub(ur"\u0153", "oe", string)
    string = re.sub(ur"\u0152", "OE", string)
    return string 

def replace_quotes(string):
    """ Replace quotes that are absent from Latin-1 with Latin-1 equivalents. """
    string = re.sub(ur"[\u2018\u2019]", "'", string)
    string = re.sub(ur"[\u201C\u201D]", '"', string)
    return string

def normalize_chars(string, lang):
    """ 
    Normalize characters in string.

    For English, we simply map all characters to an ASCII character
    using unidecode.

    For French, we start by getting the normal form (C) of the unicode
    string, then replace characters that are not present in the
    Latin-1 encoding but that we want to keep (o+e ligatures, certains
    quotation marks). Finally, we convert the string to Latin-1,
    discarding characters that are not in Latin-1, and convert back to
    unicode.
    """
    if lang.lower()=='en':
        return unidecode(string)
    elif lang.lower()=='fr':
        string = unicodedata.normalize('NFC', string)
        string = replace_oelig(string)
        string = replace_quotes(string)
        string = string.encode('latin-1', errors='ignore')
        string = string.decode('latin-1')
        return string
    else: 
        print 'ERROR: "{}" not a recognized language.'.format(lang)
        return None

def lemma_is_uncertain(string):
    """ 
    Check if a string contains a pipe, which is used by TreeTagger to
    indicate that it is unsure of the lemma of some token.
    """

    pattern = ".+\|.+"
    if re.search(pattern,string):
        return True
    else:
        return False

def get_treetagger_triple(string):
    """ 
    Split a single line from TreeTagger's output to obtain a
    (word,pos,lemma) triple.
    """ 

    elems = string.split('\t') 
    # Lines that don't contain exactly 2 tabs are ignored.  These are
    # usually lines containing a single <repdns> or <repurl> element
    # which TreeTagger uses to indicate that it has replaced the token
    # in the previous (token, pos, lemma) triple with a special symbol
    # (e.g. dns-remplacé). The replaced text is in the "text"
    # attribute of the repdns or repurl element.
    if len(elems) == 3:
        return elems
    else:
        return None

def lemmatize(tagger, string):
    """ 
    Use TreeTagger to tokenize a string (at sentence and word level) and lemmatize.

    Arguments:
    tagger -- an instance of treetaggerwrapper.TreeTagger
    string -- a string
    """

    sentences = []
    current_sent = []
    lines = tagger.TagText(string)
    for line in lines:
        triple = get_treetagger_triple(line)
        if triple:
            word, pos, lemma = triple
            if lemma == u'<unknown>' or lemma_is_uncertain(lemma):
                token = word.lower()
            else:
                token = lemma.lower()
            # Replace o+e ligatures
            token = replace_oelig(token)
            if pos == 'SENT':
                if len(current_sent):
                    current_sent.append(token)
                    sentences.append(current_sent)
                    current_sent = []
                else:
                    # The sentence is empty. Ignore the punctuation mark.
                    pass
            else:
                current_sent.append(token)
    # Make sure that the current sentence is empty, or else add to output.
    if len(current_sent):
        sentences.append(current_sent)
    return sentences

def sent_tokenize(tagger, string):
    """ 
    Use treetagger to tokenize a string at sentence and word level.

    Arguments:
    tagger -- an instance of treetaggerwrapper.TreeTagger
    string -- a string
    """

    sentences = []
    current_sent = []
    lines = tagger.TagText(string)
    for line in lines:
        triple = get_treetagger_triple(line)
        if triple:
            word, pos, lemma = triple
            token = word.lower()
            # Replace o+e ligatures
            token = replace_oelig(token)
            if pos == 'SENT':
                if len(current_sent):
                    current_sent.append(token)
                    sentences.append(current_sent)
                    current_sent = []
                else:
                    # The sentence is empty. Ignore the punctuation mark.
                    pass
            else:
                current_sent.append(token)
    # Make sure that the current sentence is empty, or else add to output.
    if len(current_sent):
        sentences.append(current_sent)
    return sentences


if __name__ == "__main__":
    dsc = (u'Prétraitement du corpus PANACEA (français ou anglais) :'
           u' extraction du contenu textuel, normalisation de caractères,'
           u' segmentation et lemmatisation (facultative).')
    parser = argparse.ArgumentParser(description=dsc)
    parser.add_argument('-l', '--lemmatize', action="store_true", required=False,
                        help=(u"Lemmatiser au moyen de TreeTagger. Sinon, seules "
                              u"la normalisation et la segmentation sont appliquées."))
    parser.add_argument('lang', choices=['EN', 'FR'])
    parser.add_argument('input', help=u'Chemin du répertoire contenant les fichiers XML.')
    parser.add_argument('output', help=u'Chemin du fichier de sortie.')
    args = parser.parse_args()

    # Process args
    corpus_dir = args.input
    if corpus_dir[-1] != '/':
        corpus_dir += '/'
    if os.path.isfile(args.output):
        sys.exit(u'ERREUR : Le fichier {} existe déjà.'.format(args.output))

    # Get paths of XML files in corpus
    filenames = [x for x in os.listdir(corpus_dir) if x[-3:] == 'xml']

    # Initialize tagger
    tagger = ttw.TreeTagger(TAGLANG=args.lang.lower(), TAGDIR='/usr/local/TreeTagger',
                            TAGINENC='utf-8', TAGOUTENC='utf-8') 

    # Apply preprocessing
    MIN_TOKENS_PER_DOC = 50
    for i in range(len(filenames)):
        # Load file
        docname = filenames[i]    
        docpath = corpus_dir + docname
        with codecs.open(docpath, 'r', encoding='utf-8') as f:
            doctext = f.read()
        xmldoc = parseString(doctext.encode('utf-8'))
        paras = []                     
        processed_paras = []
        nb_tokens = 0

        # Get title
        title_stmt = xmldoc.getElementsByTagName('titleStmt')[0]
        title_node = title_stmt.getElementsByTagName('title')[0].firstChild
        if title_node:
            title = title_node.data
            paras.append(title)

        # Get paragraphs
        for p_elem in xmldoc.getElementsByTagName('p'):
            p_txt = p_elem.firstChild.data
            p_type = p_elem.getAttribute('crawlinfo')
            if p_type not in ['boilerplate', 'ooi-lang', 'ooi-length']:
                paras.append(p_txt)

        # Lemmatize or simply tokenize paragraphs (including title)
        for p in paras:
            p = normalize_chars(p, args.lang)
            if args.lemmatize:
                sents = lemmatize(tagger, p)
            else:
                sents = sent_tokenize(tagger, p)
            for s in sents:
                nb_tokens += len(s)
            processed_paras.append(sents)

        # Write doc if it is long enough
        if nb_tokens >= MIN_TOKENS_PER_DOC:
            with codecs.open(args.output, 'a', encoding='utf-8') as f:
                for para in processed_paras:
                    for sent in para:
                        f.write(' '.join(sent)+'\n')
                    f.write('\n')
        else:
            print u'ATTENTION : Le fichier {} a été exclu (trop court).'.format(docname)
        if i % 10 == 0:
            print u'{} fichiers traités...'.format(i)
    print u'{} fichiers traités. Terminé.\n'.format(len(filenames))



