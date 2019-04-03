import operator
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import math


# Déclaration de l'algorithme de similarité cosinus
def cosine_similarity(vector1, vector2):
    dot_product = vector2.multiply(vector1).sum()
    magnitude = math.sqrt(sum([val ** 2 for val in vector1])) * math.sqrt(vector2.power(2).sum())
    if not magnitude:
        return 0
    return dot_product / magnitude


# Récupération des jeux de données
fichiers = []
noms = []

# Chemin vers le dossier du corpus d'entrainement
url = './SimpleText/SimpleText_auto/'

# Copie du contenu des fichiers en mémoire
for file in os.listdir(url):
    fichiers.append(open(url + file, 'r+').read())
    noms.append(file)

# Chemin vers le dossier du corpus de test
test_path = './SimpleText/SimpleText_test/'

# Copie du contenu des fichiers de tests
test_documents = []
test_noms = []
for file in os.listdir(test_path):
    test_documents.append([open(url + file,'r+').read()])
    test_noms.append(file)

# Vectorisation des fichiers
tfidfvectorizer = TfidfVectorizer()
x = tfidfvectorizer.fit_transform(fichiers)

# Création d'une bibliothèque regroupant les recommandations
recommendations = {}

for index, document in enumerate(x.toarray()):
    # Pour chaque document, on va calculer la "compatibilité" avec les documents de tests
    for index_test, doc_test in enumerate(test_documents):
        # On vérifie que le fichier que l'on compare n'est pas le même que le document initial
        if test_noms[index_test] != noms[index]:
            # Création de la matrice creuse pour le document de test qui va être comparé
            diff_doc = tfidfvectorizer.transform(doc_test)
            if test_noms[index_test] not in recommendations:
                recommendations[test_noms[index_test]] = {}
            recommendations[test_noms[index_test]][index] = cosine_similarity(document,diff_doc)

for key,value in recommendations.items():
    # Affiche les 5 fichiers ayant le plus de ressemblances pour chaque fichier testés
    print('Le fichier ',key, ' est similaire aux fichiers  suivants :')
    for elem in sorted(value.items(), key=operator.itemgetter(1), reverse=True)[:5]:
        print(noms[elem[0]] + " Avec un score de ressemblance de  " + str(elem[1]))