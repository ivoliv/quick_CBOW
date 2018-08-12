from torch import nn
import torch.optim as optim
from tqdm import tqdm
import urllib3
import spacy
from cbow_models import CBOW1
from utils import *
import sys


def read_and_setup(CORPUS_URL, CORPUS_START, CORPUS_END, 
                   CONTEXT_SIZE, EMBEDDING_DIM, device):

    # Read corpus from url

    urllib3.disable_warnings()
    http = urllib3.PoolManager()
    r = http.request('GET', CORPUS_URL)
    input_text = r.data.decode('utf-8')
    input_text = input_text[CORPUS_START:CORPUS_END]

    # spacy processing of corpus

    nlp = spacy.load('en')
    doc = nlp(input_text)

    input_text = clean_tokens(doc)
    vocab = set(input_text)
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    ix_to_word = [i for i in enumerate(vocab)]
    len_text = len(input_text)
    len_vocab = len(vocab)
    data = create_context_target(input_text, len_text, CONTEXT_SIZE)

    print('Size of corpus:', len_text)
    print('Number of words (lemmas):', len_vocab)
    sys.stdout.flush()

    model = CBOW1(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE, device)

    return model, data, word_to_ix, ix_to_word


def optimize(model, data, word_to_ix, device, EPOCHS):

    losses = []
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for _ in tqdm(range(EPOCHS)):
        total_loss = 0
        for context, target in data:

            context_idxs = make_context_vector(context, word_to_ix, device)

            model.zero_grad()

            log_probs = model(context_idxs)
            loss = loss_function(log_probs,
                                 torch.tensor([word_to_ix[target]],
                                              dtype=torch.long).to(device))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        losses.append(total_loss)
    return model, losses


def main():
    
    # Set options

    CONTEXT_SIZE = 2
    EMBEDDING_DIM = 20

    CORPUS_URL = 'https://www.gutenberg.org/files/11/11-0.txt'
    CORPUS_START = 752
    CORPUS_END = 10000
    EPOCHS = 200

    # Check GPU availability

    device = torch.device('cpu')
    if torch.cuda.is_available():
        print('CUDA found')
        device = torch.device('cuda:0')
    print('Using device:', device)

    # Read and optimize model

    model, data, word_to_ix, ix_to_word = read_and_setup(CORPUS_URL, CORPUS_START, CORPUS_END,
                                                         CONTEXT_SIZE, EMBEDDING_DIM, device)
    model, losses = optimize(model, data, word_to_ix, device, EPOCHS)

    #Outputs

    plot_losses(losses)
    text_head(100, data, model, word_to_ix, ix_to_word, device)


if __name__ == '__main__':

    main()
