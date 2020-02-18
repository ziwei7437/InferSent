
from .models import InferSent

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss


class InferSentForSequenceClassification(nn.Module):
    
    def __init__(self, config, num_labels):
        super(InferSentForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        #self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.model = InferSent(config)

    def forward(self, input_sentences, bsize=128, tokenize=False, verbose=True, labels=None):
        embeddings = self.model.encode(input_sentences, bsize=bsize, tokenize=tokenize, verbose=verbose)
        logits = self.classifer(embeddings)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


def main():
    pass
    #TODO


if __name__ == "__main__":
    main()