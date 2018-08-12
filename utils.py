import torch


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)


def clean_tokens(doc):
    input_text = []
    for token in doc:
        if token.pos_ not in {'SPACE', 'PUNCT', 'PART'}:
            input_text.append(token.lemma_)
    return input_text


def create_context_target(input_text, len_text, CONTEXT_SIZE):
    data = []
    for i in range(CONTEXT_SIZE, len_text - CONTEXT_SIZE):
        context = []
        for j in range(-CONTEXT_SIZE, 0):
            context.append(input_text[i + j])
        for j in range(1, CONTEXT_SIZE + 1):
            context.append(input_text[i + j])
        target = input_text[i]
        data.append((context, target))
    return data


def plot_losses(losses):
    try:
        import matplotlib.pyplot as plt
        plt.plot(losses)
        plt.show()
    except:
        print('No matplotlib - supressing losses plot.')


def text_head(num, data, model, word_to_ix, ix_to_word):
    i = 0
    for context, target in data:
        i += 1
        if i > num:
            break
        context_idxs = make_context_vector(context, word_to_ix)
        pred_idx = torch.argmax(model(context_idxs)[0])
        pred_word = ix_to_word[pred_idx]
        err = 'XXXXXXXXX' if target != pred_word[1] else ''
        print(context, target, pred_word, err)
