import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Jeux de données
# Dans un soucis de simplicité, on va transformer les dates en nombres simples
# La date du 16-03-2019 sera 0
# La date du 03-04-2019 sera 18
# On cherche donc la valeur pour 19

# J'ai remplacé mon algorithme par celui de sklearn car
# Mon algorithme était trop lent et pas assez précis
# L'ancien algo est disponible sur controle_linear_date.py

dataset = [
    [0.0, 81682.0],
    [2.0, 81720.0],
    [4.0, 81760.0],
    [8.0, 81826.0],
    [9.0, 81844.0],
    [10.0, 81864.0],
    [11.0, 81881.0],
    [12.0, 81900.0],
    [14.0, 81933.0],
    [18.0, 82003.0]
]

X = np.array([elem[0] for elem in dataset]).reshape(-1,1)
y = np.array([elem[1] for elem in dataset]).reshape(-1,1)

lr = LinearRegression().fit(X,y)

dayWanted = 19.0
result = lr.predict(np.array(dayWanted).reshape(-1,1))
print('On cherche l''estimation du '+ str(dayWanted) +'ème jour')
print('Soit : '+ str(result[0][0]))

plt.close()
plt.scatter(X, y)
plt.plot(dayWanted, result[0][0], 'ro')
a = np.arange(0,dayWanted,1)

# Prédiction du point d'origine
p_0 = lr.predict(np.array(0.0).reshape(-1,1))

# Création de la ligne à partir de la prédiction à x=0 et x=dayWanted
b = np.arange(p_0[0][0],result[0][0],(result[0][0]-p_0[0][0])/(dayWanted))
plt.plot(a, b, color='blue', linewidth=1)

plt.show()