import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import urllib3
import spacy
import matplotlib.pyplot as plt

nlp = spacy.load('en')

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10


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

def make_context_vector(context, word_to_ix):
        idxs = [word_to_ix[w] for w in context]
        return torch.tensor(idxs, dtype=torch.long)



http = urllib3.PoolManager()
r = http.request('GET', 'https://www.gutenberg.org/files/11/11-0.txt')

test_sentence = r.data.decode('utf-8')[752:20000]
doc = nlp(test_sentence)
test_sentence = []

#test_sentence[:200]
#doc

for token in doc:
    if token.pos_ not in {'SPACE', 'PUNCT', 'PART'}:
        test_sentence.append(token.lemma_)

print('Size of corpus:', len(test_sentence))

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(CONTEXT_SIZE, len(test_sentence) - CONTEXT_SIZE):
    context = []
    for j in range(-CONTEXT_SIZE, 0):
        context.append(test_sentence[i + j])
    for j in range(1, CONTEXT_SIZE+1):
        context.append(test_sentence[i + j])
    target = test_sentence[i]
    data.append((context, target))
#print(data[:5])
print('Number of words (lemmas):', len(vocab))

make_context_vector(data[0][0], word_to_ix)



model = CBOW1(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
#print(model)
losses = []
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in tqdm(range(300)):
    total_loss = 0
    for context, target in data:
        
        context_idxs = make_context_vector(context, word_to_ix)
        
        model.zero_grad()
        
        log_probs = model(context_idxs)
        loss = loss_function(log_probs,
                             torch.tensor([word_to_ix[target]],
                                          dtype=torch.long))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    losses.append(total_loss)
    
plt.plot(losses)
plt.show()

#losses[-5:]

ix_to_word = [i for i in enumerate(vocab)]
i = 0
for context, target in data:
    i += 1
    if i > 1000:
        break
    context_idxs = make_context_vector(context, word_to_ix)
    pred_idx = torch.argmax(model(context_idxs)[0])
    pred_word = ix_to_word[pred_idx]
    err = 'XXXXXXXXX' if target != pred_word[1] else ''
    print(context, target, pred_word, err)

