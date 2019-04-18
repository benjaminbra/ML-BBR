from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from TP_CART.RandomForest import RandomForest
from TP_CART.SVM import SVM
from TP_CART.LogisticRegression import LogisticRegression
from TP_CART.Cart import Cart

datasets = fetch_olivetti_faces()

x_train, entry_test, y_train, expected_test = train_test_split(datasets.data, datasets.target, test_size=0.2)


RandomForest(x_train,y_train).run(entry_test,expected_test)

SVM(x_train,y_train,'ovo').run(entry_test,expected_test)

SVM(x_train,y_train,'ovr').run(entry_test,expected_test)

LogisticRegression(x_train,y_train).run(entry_test,expected_test)

Cart(x_train,y_train).run(entry_test,expected_test)