from sklearn import datasets
from sklearn.model_selection import train_test_split
from neuronne import Neuronne
from network import Network
import datetime

# We load the data
dataset = datasets.load_digits()
data = dataset['data']
target = dataset['target']

# Split the data into 70% training data and 30% test data
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3)


nb_numbers = 10
nb_dimension = 64

ntw = Network(nb_numbers,nb_dimension)

best_score = 0

for i in range(0,3):
	start_train_time = datetime.datetime.now()

	ntw.train(x_train, y_train)

	train_time = datetime.datetime.now() - start_train_time

	start_test_time = datetime.datetime.now()

	nb_success = ntw.test(x_test, y_test)

	test_time = datetime.datetime.now() - start_test_time

	print('Nombre de succès : ' + str(nb_success) + '/' + str(len(x_test)))
	print('Le taux de précision est de ' + str((nb_success / len(x_test)) * 100) + '%')
	print('Durée d\'entrainement : ' + str(train_time))
	print('Durée de test : ' + str(test_time))

# Train
# 1 - Creer un neuronne qui va tenter de predire le poids d'une donnee

# 2 - Recuperer la prediction a partir du neuronne dont le poids est le plus eleve

# 3 - Test de la donnee
# Si la donnee est differente on reduit le poids du neuronne
# Et on augmente le poids du neuronne qui aurait du deviner la valeur


# TEST
# 1 - Creer un neuronne qui va tenter de predire le poids d'une donnee de tests


# 2 - Recuperer la prediction a partir du neuronne dont le poids est le plus eleve
# Plus un poids est eleve, plus on est proche de la realite normalement

# 3 - Test de la donnee obtenue avec la donnee de comparaison 