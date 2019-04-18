
class Utils:

    def precision(predictions, expectations):
        good_predictions = 0
        for i in range(len(predictions)):
            if predictions[i] == expectations[i]:
                good_predictions += 1

        return good_predictions / len(predictions) * 100