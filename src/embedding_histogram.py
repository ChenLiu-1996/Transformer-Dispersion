import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import AlbertConfig, AlbertTokenizer, AlbertModel
from matplotlib import pyplot as plt


def compute_correlations(hidden_states):
    corrs = []
    for hs in hidden_states:
        T = hs.squeeze(0).clone().detach().requires_grad_(False)
        T = torch.nn.functional.normalize(T, dim=1)
        T2 = torch.matmul(T, T.transpose(0,1))
        corrs += [T2.flatten().cpu(),]
    return corrs

def get_random_input(dataset, tokenizer):
    l = len(dataset['train'])
    while True:
        it = torch.randint(l,(1,)).item()
        text = dataset['train'][it]['text']
        tokens = tokenizer(text, return_tensors='pt', truncation=True)
        if (tokens['input_ids'].shape[1]>300):
            break
    return tokens

def plot_histograms(dataset, tokenizer, model):

    tokens = get_random_input(dataset, tokenizer)
    print(tokenizer.batch_decode(tokens['input_ids']))
    output = model(**tokens, output_hidden_states=True)
    correls = compute_correlations(output['hidden_states'])

    if model.config.num_hidden_layers < 25:
        fig, axes = plt.subplots(5,5)

    else:
        fig, axes = plt.subplots(7,7)
    axes = axes.flatten()

    for i in range(len(correls)):
        axes[i].hist(correls[i], bins=100, density=True, histtype='step')
        axes[i].set_title(f'Layer {i}')
        axes[i].set_xlim(-.3, 1)

def plot_histograms_new(dataset, tokenizer, model, step=1):

    tokens = get_random_input(dataset, tokenizer)
    print(tokenizer.batch_decode(tokens['input_ids']))
    output = model(**tokens, output_hidden_states=True)
    correls = compute_correlations(output['hidden_states'])

    nr_plots = (model.config.num_hidden_layers + 1) // step
    if model.config.num_hidden_layers < 25:
        cols = 5
        fig, axes = plt.subplots(5,5)

    else:
        fig, axes = plt.subplots(7,7)

    axes = axes.flatten()
    for i in range(len(correls)):
        axes[i].hist(correls[i],bins=100, density=True, histtype='step')
        axes[i].set_title(f'Layer {i}')
        axes[i].set_xlim(-.3,1)

def plot_histograms_save(dataset, tokenizer, model):
    tokens = get_random_input(dataset, tokenizer)
    print(tokenizer.batch_decode(tokens['input_ids']))
    output = model(**tokens, output_hidden_states=True)
    correls = compute_correlations(output['hidden_states'])

    # Create a directory to save the plots
    os.makedirs('histograms', exist_ok=True)

    # Determine the global maximum density value
    max_density = 0
    for data in correls:
        counts, bin_edges = np.histogram(data, bins=100, density=True)
        max_density = max(max_density, max(counts))

    for i, data in enumerate(correls):
        IQR = np.percentile(data, 75) - np.percentile(data, 25)
        n = len(data)
        bin_width = 2 * IQR / n ** (1/3)
        bins = int((max(data) - min(data)) / bin_width)

        plt.figure()
        plt.hist(data, bins=bins, density=True, histtype='step', color='#3658bf', linewidth=1.5)
        plt.title(f'Layer {i}', fontsize=16)
        plt.xlim(-.3, 1.05)
        plt.ylim(0, max_density)  # Set a consistent y-axis limit

        plt.savefig(f'histogram_layer_{i}.pdf')
        plt.close()


if __name__ == '__main__':
    wikitext = load_dataset("wikitext", 'wikitext-103-v1')

    tokenizer_albert = AlbertTokenizer.from_pretrained('albert-xlarge-v2')
    model_albert = AlbertModel.from_pretrained("albert-xlarge-v2")

    plot_histograms_save(dataset=wikitext, tokenizer=tokenizer_albert, model=model_albert)

    ### Check norm of output tokens
    ## The norm is not exactly the same because the LayerNorm
    ## that is applied at the end also has trainable diagonal matrix \gamma and  vector \beta which are used as follows
    ## (on each token)
    ## \tilde x = (x - mean(x))/sqrt(var(x)) * \gamma + \beta   (here token is a row vector)

    tokens = get_random_input(dataset=wikitext, tokenizer=tokenizer_albert)
    print(tokenizer_albert.batch_decode(tokens['input_ids']))
    output = model_albert(**tokens, output_hidden_states=True)
    output['hidden_states'][24].var(2)

    config_albert = AlbertConfig.from_pretrained("albert-xlarge-v2", num_hidden_layers=48, num_attention_heads=1)
    model_albert2 = AlbertModel.from_pretrained("albert-xlarge-v2", config=config_albert)
    print(model_albert.config.num_hidden_layers)

    plot_histograms_save(dataset=wikitext, tokenizer=tokenizer_albert, model=model_albert2)

    # Decomposing AlbertModel
    model_encoder = model_albert2.encoder
    model_layer = model_encoder.albert_layer_groups[0].albert_layers[0]
    attention = model_layer.attention

    output = model_albert2(**tokens, output_hidden_states=True)
    test = torch.arange(2048).reshape(1,1,-1)
    print(attention.transpose_for_scores(test)[0,1,0,:])
    print(attention.value.weight)

