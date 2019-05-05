from neuronne import Neuronne

class Network:

	def __init__(self, nb_values, dimensions):

		# Creation d'autant de neuronnes que j'ai de valeur a predire
		self.neuronnes = [Neuronne(dimensions) for i in range(nb_values)]

	def train(self, train_input, train_expecting):
		# Pour chaque donnees d'entrainement, 
		# je vais recuperer le neuronne avec le poids le plus lourd
		for index, train_image in enumerate(train_input):
			weights = [neuronne.get_predicted_weight(train_image) for neuronne in self.neuronnes]
			prediction = weights.index(max(weights))

			# Je verifie que ma prediction correspond a ce qui est attendu
			# Si ce n'est pas le cas je met a jour les poids des neuronnes concernes
			if prediction != int(train_expecting[index]):
				self.neuronnes[int(prediction)].reduce(train_image)
				self.neuronnes[int(train_expecting[index])].increase(train_image)

	def test(self, test_input, test_expecting):
		#  Pour chaque données de test
		# Je vais récupérer le neuronne avec le poids le plus lourd
		nb_success = 0
		for index, test_image in enumerate(test_input):
			weights = [neuronne.get_predicted_weight(test_image) for neuronne in self.neuronnes]
			prediction = weights.index(max(weights))

			# En cas de succès, j'augmente le nombre de tests réussis
			if prediction == int(test_expecting[index]):
				nb_success += 1

		return nb_success

	def get_neuronnes(self):
		return self.neuronnes