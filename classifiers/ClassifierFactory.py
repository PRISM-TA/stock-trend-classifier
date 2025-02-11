from classifiers.BaseClassifier import BaseClassifier

class ClassifierFactory:

    def __init__(self, model_class: BaseClassifier):
        self.model_class = model_class

    def create_classifier(self, input_size: int):
        return self.model_class(input_size=input_size)