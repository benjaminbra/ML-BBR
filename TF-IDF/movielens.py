from surprise import SVD
from surprise import Dataset
from surprise.model_selection import train_test_split
import matplotlib.pyplot as plt

# Récupération du jeu de données de MovieLens
data = Dataset.load_builtin('ml-100k')

# Récupération de l'algorithme de Décompostion en value singulière
svd = SVD()

# Séparation du jeu de données
# 20% de jeu de test, et 80% de jeu d'entrainement
data_train, data_test = train_test_split(data, test_size=0.2)

# Entrainement de l'algorithme à partir du jeu d'entrainement puis prédiction pour le jeu de test
predict = svd.fit(data_train).test(data_test)

result =[]
for prediction in predict:
    # Calcul du delta entre la prediction et la réalité
    result.append(prediction.r_ui - prediction.est)

# Affiche l'histogramme du delta entre les predictions et la réalité
plt.hist(result, 80)

plt.show()