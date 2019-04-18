Dans le cadre de ce tp, on réalisé une comparaison des techniques afin d'observer quelle est la plus précise pour un jeu de données commun.

Afin de lancer la comparaison, il suffit d'executer main.py.

Voici des résultats que j'ai pu observer avec le jeu de données `fetch_olivetti_faces` :

Résultats avec une première tentative :
```
Random Forest Precision : 95.0% - train : 0:00:02.648477 - test : 0:00:00.016991
SVM (ovo) Precision : 91.25% - train : 0:00:01.073388 - test : 0:00:00.130927
SVM (ovr) Precision : 91.25% - train : 0:00:01.069386 - test : 0:00:00.133925
Logistic Regression OvR Precision : 98.75% - train : 0:01:35.416923 - test : 0:00:00.006997
Cart Precision : 62.5% - train : 0:00:04.411478 - test : 0:00:00.000999
```

Résultats avec une seconde tentative : 
```
Random Forest Precision : 85.0% - train : 0:00:02.740430 - test : 0:00:00.017988
SVM (ovo) Precision : 81.25% - train : 0:00:01.095373 - test : 0:00:00.141918
SVM (ovr) Precision : 81.25% - train : 0:00:01.183322 - test : 0:00:00.133947
Logistic Regression OvR Precision : 91.25% - train : 0:01:43.891968 - test : 0:00:00.005993
Cart Precision : 55.00000000000001% - train : 0:00:02.804397 - test : 0:00:00
```

On peut constater que les résultats restent globalement variable en terme de précision.
<br/> Notamment pour le Random Forest, SVM. Où l'on peut avoir plus de 10% d'écart entre chaque résultats.

La regression logistique va être plus précise, et le résultat semble relativement stable.
<br/> De plus, pour obtenir le résultat, il a fallu passer presque 2 minutes d'entrainement. Ce qui est beaucoup plus long que les autres techniques qui sont plus proches des 1 à 2 secondes.

Le Cart semble moins précis au niveau de ses prédictions.

