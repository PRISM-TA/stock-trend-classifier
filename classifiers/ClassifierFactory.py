from classifiers.BaseClassifier import BaseClassifier
from models.BaseHyperParam import BaseHyperParam

class ClassifierFactory:

    def __init__(self, training_param: BaseHyperParam, model_class: BaseClassifier):
        self.training_param = training_param
        self.model_class = model_class

    def create_classifier(self, input_size: int):
        return self.model(input_size=input_size)