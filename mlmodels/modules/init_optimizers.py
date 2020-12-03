import torch
from mlmodels.modules.radam import RAdam
import torch.optim as optim
from transformers import AdamW
from typing import Generator


class optimizers:
    @staticmethod
    def init_optimizers(optimizer: str, model_named_parameters: Generator, learning_rate: float,
                        adam_epsilon: float, weight_decay):
        """

        @param optimizer: parameter to choose the optimizer
        @param model_named_parameters: model parameters
        @param learning_rate: learning rate
        @param adam_epsilon: adam epsilon value
        @param weight_decay: weight decay
        @return: return optimizer
        """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model_named_parameters if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {"params": [p for n, p in model_named_parameters if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        if optimizer.lower() == "adamax":
            optimizer = optim.Adamax(optimizer_grouped_parameters, lr=learning_rate,
                                     eps=adam_epsilon)
        elif optimizer.lower() == "adamw":
            optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate,
                              eps=adam_epsilon)
        elif optimizer.lower() == "adam":
            optimizer = optim.Adam(optimizer_grouped_parameters, lr=learning_rate,
                                   eps=adam_epsilon)

        elif optimizer.lower() == "radam":
            optimizer = RAdam(optimizer_grouped_parameters, lr=learning_rate,
                              eps=adam_epsilon)

        elif optimizer.lower() == "adadelta":
            optimizer = optim.Adadelta(optimizer_grouped_parameters, lr=learning_rate,
                                       eps=adam_epsilon)

        elif optimizer.lower() == "adagrad":
            optimizer = optim.Adagrad(optimizer_grouped_parameters, lr=learning_rate,
                                      eps=adam_epsilon)
        else:
            optimizer = optim.SGD(optimizer_grouped_parameters, lr=learning_rate)
        return optimizer
