from sklearn.ensemble import RandomForestClassifier
from TP_CART.Utils import Utils
import datetime


class RandomForest:

    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def run(self, entry_test, expected_test):
        start_train_forest_time = datetime.datetime.now()

        forest = RandomForestClassifier(n_estimators=100).fit(self.x_train, self.y_train)

        train_forest_time = datetime.datetime.now() - start_train_forest_time

        start_test_forest_time = datetime.datetime.now()
        predictions = forest.predict(entry_test)

        test_forest_time = datetime.datetime.now() - start_test_forest_time
        forest_precision = Utils.precision(predictions, expected_test)
        print('Random Forest Precision : ' + str(forest_precision) + '% - train : ' + str(train_forest_time) + ' - test : ' + str(test_forest_time))
