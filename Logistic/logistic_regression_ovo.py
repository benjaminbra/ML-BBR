from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import itertools
import operator

# On charge le jeu de données
digits = datasets.load_digits()

# On récupère les données
donnees = digits['data']

# Récupération des données cibles générées
valeurs_cible = digits['target']
classes = set(valeurs_cible)


x_train, x_test, y_train, y_test = train_test_split(donnees, valeurs_cible, test_size=0.2)

# Initialisation de la bibliothèque des classifiers
listes_classifiers = {}

# Création des différentes combinaisons de classifers
for cible in itertools.combinations(classes, 2):
    class0 = [x_train[index] for index, value in enumerate(y_train) if value == cible[0]]
    class1 = [x_train[index] for index, value in enumerate(y_train) if value == cible[1]]

    # Création des "tableaux" à alimenter par les données d'entrainement
    value = [0] * len(class0) + [1] * len(class1)

    # Création des données d'apprentissages en récupérant les données des classes
    apprentissage = class0 + class1
    # Création du classifieur à partir des données d'entrainement et ajout dans la bibliothèque
    listes_classifiers['%s_%s' % cible] = LogisticRegression(solver='lbfgs').fit(apprentissage, value)

resultats = {}
# Test de prédictions des digits de tests avec les classifiers créés (allant de 0 à 9)
for index, value in enumerate(y_test):
    classifier_results = {}
    # On fait un contrôle avec tous les classifiers pour trouver la meilleure prédiction
    for name, classifier in listes_classifiers.items():
        c_result = classifier.predict([x_test[index]])
        # Séparation des cibles
        c_cibles = name.split('_')
        if(classifier_results.get(c_cibles[c_result[0]])):
            # Compte le nombre de fois qu'une cible est obtenu
            classifier_results[c_cibles[c_result[0]]] += 1
        else:
            # Si la cible n'a pas encore de score on l'initialise
            classifier_results[c_cibles[c_result[0]]] = 1

    # Ajout du digit le plus prédit dans les résultats
    resultats[index] = max(classifier_results.items(), key=operator.itemgetter(1))[0]

# Affichage des prédictions par rapport aux attendus
for key, elem in resultats.items():
    value = y_test[key]
    print('Attendu : ', value, ' Prédiction : ', elem)