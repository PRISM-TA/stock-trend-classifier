import torch
import torch.nn as nn


class BaseHyperParam:
    criterion: nn.Module # loss function to be used for optimization
    optimizer: torch.optim.Optimizer # optimizer for updating model parameters

    num_epochs: int # number of epochs for which the model will be trained

    early_stopping: bool # boolean flag indicating whether early stopping should be applied
    val_loader: torch.utils.data.DataLoader # data loader for the validation dataset, required if early stopping is enabled
    patience: int # number of epochs with no improvement after which training will be stopped if early stopping is on

    def __init__(self, 
                 criterion: nn.Module, 
                 optimizer: torch.optim.Optimizer, 
                 num_epochs: int = 100, 
                 early_stopping: bool = True, 
                 patience: int = 50):
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.early_stopping = early_stopping
        self.patience = patience
    
    def __repr__(self):
        return f"<BaseHyperParam(criterion={self.criterion}, optimizer={self.optimizer}, num_epochs={self.num_epochs}, early_stopping={self.early_stopping}, val_loader={self.val_loader}, patience={self.patience})>"