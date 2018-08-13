from torch import nn
import torch.nn.functional as F


# This version of CBOW has no hidden layers
class CBOW0(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW0, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(2 * context_size * embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = self.linear1(embeds)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


# CBOW with 1 hidden layer
class CBOW1(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW1, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(2 * context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

