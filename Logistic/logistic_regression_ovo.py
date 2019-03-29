from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import itertools

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
    listes_classifiers['%c_%c' % cible] = LogisticRegression.fit(apprentissage, value)

# Test de prédictions des digits de tests avec les classifiers créés (allant de 0 à 9)
for index, value in enumerate(y_test):
    result = listes_classifiers.predict([x_test[0]])
    print("Resultat : ", result)
    print("Attendu :", value)