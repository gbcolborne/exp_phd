# Évaluation de modèles sémantiques distributionnels

Ce dossier contient des ressources et des programmes que j'ai
développés dans le cadre de mon doctorat. L'objectif de ma recherche
était de développer un cadre méthodologique basé sur la sémantique
distributionnelle pour l'analyse des relations lexicales à partir de
corpus spécialisés. À cette fin, j'ai évalué des modèles sémantiques
distributionnels au moyen de données de référence que j'ai extraites
de dictionnaires spécialisés. De cette façon, j'ai exploré comment le
choix et le paramétrage optimaux du modèle distributionnel sont
influencés par différents facteurs, tels que le genre de relations
lexicales qu'on souhaite identifier, la langue traitée ou la partie du
discours des requêtes utilisées pour évaluer le modèle.

## Code


* `exp_AD.py` et `exp_W2V.py` : programmes qui construisent des
  modèles distributionnels (analyse distributionnelle et word2vec
  respectivement) et les évaluent sur les données de
  référence. **ATTENTION** : Le programme `exp_AD.py` exige beaucoup
  de mémoire (du moins si le nombre de mots-cibles est élevé).

* `Corpus.py` : module utilisé par `exp_AD.py` et `exp_W2V.py` pour
  générer les phrases que contient le corpus, identifier les
  mots-cibles en fonction de leur fréquence dans le corpus et d'autres
  critères, etc.

* `KNNGraph.py` : module utilisé par `exp_AD.py` et `exp_W2V.py` pour
  construire des graphes de *k* plus proches voisins.

* `CoocTensor.py` : module utilisé par `exp_AD.py` pour construire des
  matrices de cooccurrence.

* `eval_utils.py` : module utilisé par `exp_AD.py` et `exp_W2V.py`
  pour traiter les données de référence et calculer les mesures
  d'évaluation.

* `preprocess_PANACEA.py` : programme qui extrait le contenu textuel
  du corpus PANACEA (français ou anglais) et applique différentes
  opérations de prétraitement.

## Ressources

* data/ref_FR.txt et data/ref_EN.txt : les données de référence
  utilisées par `exp_AD.py` et `exp_W2V.py` pour évaluer les
  modèles. Les données de référence ont été extraites du
  [DiCoEnviro](http://olst.ling.umontreal.ca/cgi-bin/dicoenviro/search_enviro.cgi)
  et du [Framed
  DiCoEnviro](http://olst.ling.umontreal.ca/dicoenviro/framed/index.php). Chaque
  ligne contient une paire de termes accompagnés de leur partie du
  discours ainsi que la relation à laquelle ils participent : QSYN
  (synonyme, quasi-synonyme, variante, cohyponyme ou sens voisin),
  ANTI (antonyme), HYP (hyperonyme ou hyponyme), DRV (dérivé
  syntaxique) ou FRM (terme évoquant le même cadre sémantique). Note :
  dans le cas des FRM, la partie du discours des termes n'est pas
  indiquée.

* data/stop_FR.txt et data/stop_EN.txt : listes (facultatives) de mots
  vides utilisées par `exp_AD.py` et `exp_W2V.py` pour identifier les
  mots-cibles si l'option -s est utilisée. Ces listes ont été adaptées
  de celles qu'exploite le raciniseur (*stemmer*)
  [Snowball](http://snowballstem.org/), qui se trouvent
  [ici](http://snowballstem.org/algorithms/french/stop.txt) (pour le
  français) et
  [ici](http://snowballstem.org/algorithms/english/stop.txt) (pour
  l'anglais).

## Utilisation

L'évaluation quantitative systématique décrite dans ma thèse peut être
reproduite de la façon suivante :

1. Obtenir le corpus monolingue PANACEA du domaine de l'environnement
en
[français](http://catalog.elra.info/product_info.php?products_id=1186&language=fr)
ou en
[anglais](http://catalog.elra.info/product_info.php?products_id=1184&language=fr). Les
commandes suivantes sont utilisées pour traiter le corpus français,
puis construire des modèles sur ce corpus et les évaluer sur les
données de référence; pour traiter l'anglais, remplacer l'option FR
par EN dans chacune des commandes. Le dossier contenant les fichiers
XML du corpus sera appelé PANACEA_XML.

2. `python preprocess_PANACEA.py -l FR PANACEA_XML corpus.txt`

3. `python exp_AD.py -s FR corpus.txt res_AD`

4. `python exp_W2V.py -s FR corpus.txt res_W2V`

Les dossiers res_AD et res_W2V contiendront chacun deux fichiers de
résultats en format CSV. Un premier fichier contiendra la MAP sur
différents jeux de données de référence en fonction du paramétrage du
modèle. Le second contiendra les résultats de l'évaluation de chacun
des graphes de voisinage construits à partir de chaque modèle,
c'est-à-dire la précision et le rappel sur chacun des jeux de données
en fonction du paramétrage du modèle et du graphe.

Les jeux de données comprennent un jeu pour chacune des relations
présentes dans les données de référence (QSYN, ANTI, HYP, DRV et FRM),
ainsi qu'un jeu pour chacune des parties du discours (NN, VV ou JJ),
qui contient seulement les relations entre termes de la même partie du
discours (ainsi, les DRV ne sont pas inclus dans ces jeux). Un dernier
jeu de données, appelé *TOUTES*, comprend toutes les relations excepté
les FRM.

## Configuration requise

Il faut avoir installé Python (2.x).

Pour utiliser `exp_AD.py` ou `exp_W2V.py`, il faut avoir installé les
bibliothèques NumPy, scikit-learn et NetworkX pour Python.

Pour utiliser `exp_AD.py`, il faut aussi avoir installé SciPy.

Pour utiliser `exp_W2V.py`, il faut avoir installé
[word2vec](https://code.google.com/p/word2vec/). Installer dans
/usr/local (sinon modifier PATH_W2V dans `exp_W2V.py`). Il faut aussi
avoir installé la bibliothèque
[Gensim](https://radimrehurek.com/gensim/) pour Python.

Pour utiliser `preprocess_PANACEA.py`, il faut avoir installé
[TreeTagger](http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/)
ainsi que la bibliothèque
[TreeTaggerWrapper](https://pypi.python.org/pypi/treetaggerwrapper)
pour Python. Installer TreeTagger dans /usr/local (sinon modifier
TAGDIR dans `preprocess_PANACEA.py`). Pour traiter un corpus
anglais, il faut aussi avoir installé la bibliothèque
[Unidecode](https://pypi.python.org/pypi/Unidecode) pour Python.
