""" A lightweight wrapper for a Sci-kit learn learner """
import warnings

class model:
    def __init__(self, ignore_features=None, model=None, learner_type=""):
        self.ignore_features = ignore_features
        self.model = model
        self.learner_type = learner_type

    def predict(self, features):
        try:
            features = features.drop(self.ignore_features, axis=1)
        except:
            warnings.warn("Could not drop ignore features")
        return self.model.predict(features)
