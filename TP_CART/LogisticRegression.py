from sklearn.linear_model import LogisticRegression as LR
from TP_CART.Utils import Utils
import datetime


class LogisticRegression:

    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def run(self, entry_test, expected_test):
        start_train_time = datetime.datetime.now()

        lr = LR(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=4000).fit(self.x_train, self.y_train)

        train_time = datetime.datetime.now() - start_train_time

        start_test_time = datetime.datetime.now()
        predictions = lr.predict(entry_test)

        test_time = datetime.datetime.now() - start_test_time
        precision = Utils.precision(predictions, expected_test)
        print('Logistic Regression OvR Precision : ' + str(precision) + '% - train : ' + str(
            train_time) + ' - test : ' + str(test_time))
