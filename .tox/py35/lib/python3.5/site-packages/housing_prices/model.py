""" A lightweight wrapper for a Sci-kit learn learner """
import utils

class model:
    def __init__(self, ignore_features=None, model=None, learner_type=""):
        self.ignore_features = ignore_features
        self.model = model
        self.learner_type = learner_type

    def predict(self, features):
        features = utils.drop_features(features, self.ignore_features)
        return self.model.predict(features)
