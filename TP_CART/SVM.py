from sklearn.svm import SVC
from TP_CART.Utils import Utils
import datetime


class SVM:

    def __init__(self, x_train, y_train, shape='ovo'):
        self.x_train = x_train
        self.y_train = y_train
        self.shape = shape

    def run(self, entry_test, expected_test):
        start_train_time = datetime.datetime.now()

        svm = SVC(gamma='scale', decision_function_shape=self.shape).fit(self.x_train, self.y_train)

        train_time = datetime.datetime.now() - start_train_time

        start_test_time = datetime.datetime.now()
        predictions = svm.predict(entry_test)

        test_time = datetime.datetime.now() - start_test_time
        precision = Utils.precision(predictions, expected_test)
        print('SVM (' + self.shape + ') Precision : ' + str(precision) + '% - train : ' + str(
            train_time) + ' - test : ' + str(test_time))
