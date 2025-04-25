from typing import List
import argparse
import os
import sys
import numpy as np
import torch
from datasets import load_dataset
from transformers import AlbertConfig, AlbertTokenizer, AlbertModel
import matplotlib.pyplot as plt

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir)
from diffusion.diffusion_condensation import diffusion_condensation
from dse.dse import diffusion_spectral_entropy


def get_random_long_text_input(dataset, tokenizer, min_length: int = 300) -> dict:
    while True:
        idx = torch.randint(len(dataset['train']), (1,)).item()
        text = dataset['train'][idx]['text']
        tokens = tokenizer(text, return_tensors='pt', truncation=True)
        if tokens['input_ids'].shape[1] > min_length:
            return tokens

def plot_DSE(embeddings_by_layer: List[np.ndarray], save_path: str = None):
    fig = plt.figure(figsize=(16, 8))
    for subplot_idx, l2_normalize in enumerate([False, True]):
        if l2_normalize:
            normalize_func = normalize_numpy
        else:
            normalize_func = lambda x: x
        ax = fig.add_subplot(1, 2, subplot_idx + 1)
        for sigma, color_base in zip([1, 5, 10], ['Blues', 'Reds', 'Greens']):
            cmap = plt.get_cmap(color_base)
            for diffusion_t, cmap_idx in zip([1, 2, 5, 10], [0.4, 0.6, 0.8, 1.0]):
                entropy_arr = [diffusion_spectral_entropy(normalize_func(embeddings), gaussian_kernel_sigma=sigma, t=diffusion_t)
                               for embeddings in embeddings_by_layer]
                ax.plot(entropy_arr, marker='o', linewidth=2, color=cmap(cmap_idx), label=f'$\sigma$ = {sigma}, t = {diffusion_t}')
        ax.legend(loc='upper right', ncols=3)

        ax.set_ylim([0, ax.get_ylim()[1] * 1.2])
        ax.tick_params(axis='both', which='major', labelsize=18)
        if l2_normalize:
            ax.set_title('With L2 normalization', fontsize=18)
        else:
            ax.set_title('Without L2 normalization', fontsize=18)
        ax.set_xlabel('Layer', fontsize=18)
        ax.set_ylabel('Entropy', fontsize=18)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Embedding DSE per Layer', fontsize=24)
    fig.tight_layout(pad=2)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()
    return

def normalize_numpy(x, p=2, axis=1, eps=1e-12):
    norm = np.linalg.norm(x, ord=p, axis=axis, keepdims=True)
    return x / (norm + eps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--random-seed', type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.random_seed)

    # Load dataset and model.
    wikitext = load_dataset("wikitext", "wikitext-103-v1")
    tokenizer = AlbertTokenizer.from_pretrained('albert-xlarge-v2')
    config = AlbertConfig.from_pretrained("albert-xlarge-v2", num_hidden_layers=48, num_attention_heads=1)
    model = AlbertModel.from_pretrained("albert-xlarge-v2", config=config)

    # Run model on a random long input.
    tokens = get_random_long_text_input(wikitext, tokenizer)

    # Extract the cosine similarities among token embeddings (hidden states).
    with torch.no_grad():
        output = model(**tokens, output_hidden_states=True)

        # Effectively only take the first 2 layers.
        z_0 = output.hidden_states[0].squeeze(0)  # [seq_len, hidden_dim]
        z_0 = torch.nn.functional.normalize(z_0, dim=1, p=2).cpu().numpy()
        z_1 = output.hidden_states[1].squeeze(0)  # [seq_len, hidden_dim]
        z_1 = torch.nn.functional.normalize(z_1, dim=1, p=2).cpu().numpy()

        embeddings_by_layer = diffusion_condensation(X=z_1, random_seed=args.random_seed)
        embeddings_by_layer = [z_0] + embeddings_by_layer

    # Plot and save DSE.
    plot_DSE(
        embeddings_by_layer,
        save_path='../../visualization/diffusion/embedding_DSE_albert_xlarge_v2.png')
