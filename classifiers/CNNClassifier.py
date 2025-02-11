from classifiers.BaseClassifier import BaseClassifier

import torch
from torch import nn

class CNNClassifier_V0(BaseClassifier):
    def __init__(self, input_size: int, num_classes: int = 3):
        super().__init__()
        self.model_name = "CNNv0"

        # TODO: Implement CNN
        pass

    def forward(self, x):
        pass

