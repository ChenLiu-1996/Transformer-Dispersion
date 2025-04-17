from typing import List
import argparse
import os
import math
import numpy as np
import torch
from datasets import load_dataset
from transformers import AlbertConfig, AlbertTokenizer, AlbertModel
import matplotlib.pyplot as plt


def get_random_long_text_input(dataset, tokenizer, min_length: int = 300) -> dict:
    while True:
        idx = torch.randint(len(dataset['train']), (1,)).item()
        text = dataset['train'][idx]['text']
        tokens = tokenizer(text, return_tensors='pt', truncation=True)
        if tokens['input_ids'].shape[1] > min_length:
            return tokens

def compute_cosine_similarities(embeddings: List[torch.Tensor]) -> List[np.ndarray]:
    similarities = []
    for z in embeddings:
        z = torch.nn.functional.normalize(z.squeeze(0), dim=1)
        sim = torch.matmul(z, z.T).flatten().cpu().numpy()
        similarities.append(sim)
    return similarities

def plot_similarity_histograms(similarities: List[np.ndarray],
                               save_path: str = None,
                               step: int = 1,
                               bins: int = 64,
                               xlim: tuple = (-0.3, 1.05)):
    selected = [(i, data) for i, data in enumerate(similarities) if i % step == 0]
    num_plots = len(selected)

    # Auto-determine layout (rows x cols) to be roughly square
    cols = math.ceil(math.sqrt(num_plots))
    rows = math.ceil(num_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()

    # Global y-axis limit for consistent scaling
    max_density = max(
        np.histogram(data, bins=bins, density=True)[0].max() for _, data in selected
    )

    for ax, (i, data) in zip(axes, selected):
        IQR = np.percentile(data, 75) - np.percentile(data, 25)
        bin_width = 2 * IQR / len(data) ** (1 / 3)
        optimal_bins = max(10, int((max(data) - min(data)) / bin_width))

        ax.hist(data, bins=optimal_bins, density=True, histtype='step', color='#3658bf', linewidth=1.5)
        ax.set_title(f'Layer {i}', fontsize=10)
        ax.set_xlim(*xlim)
        ax.set_ylim(0, max_density * 1.05)

    # Turn off unused axes
    for ax in axes[num_plots:]:
        ax.axis('off')

    fig.tight_layout(pad=2)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--random-seed', type=int, default=1)
    parser.add_argument('--method', type=str, default='phate')
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
        similarities = compute_cosine_similarities(output.hidden_states)

    # Plot and save histograms.
    plot_similarity_histograms(
        similarities,
        save_path='../visualization/embedding_cossim_histogram_albert_xlarge_v2.png')

    import pdb; pdb.set_trace()