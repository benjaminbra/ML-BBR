import numpy as np
from random import randint

class Neuronne:

	def __init__(self, dimension):
		# Initialisation du neuronne a partir de la dimension fournie.

		self.values = np.asarray([0]*dimension)

	def reduce(self, image):
		# Reduit le poids du neuronne pour la valeur donnee
		self.values -= ((image - np.min(image))/np.ptp(image)).astype(int)

	def increase(self, image):
		# Augmente le poids du neuronne pour la valeur donnee
		self.values += ((image - np.min(image))/np.ptp(image)).astype(int)

	def get_predicted_weight(self, image):
		# Retourne le poids par rapport à la valeur donnée. 
		# Effectue le produit des valeurs du neuronne avec la valeur donnee
		return np.dot(self.values, image)