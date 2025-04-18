from typing import List
import argparse
import os
import math
import numpy as np
import torch
from datasets import load_dataset
from transformers import AlbertConfig, AlbertTokenizer, AlbertModel
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_random_long_text_input(dataset, tokenizer, min_length: int = 300) -> dict:
    while True:
        idx = torch.randint(len(dataset['train']), (1,)).item()
        text = dataset['train'][idx]['text']
        tokens = tokenizer(text, return_tensors='pt', truncation=True)
        if tokens['input_ids'].shape[1] > min_length:
            return tokens

def compute_cosine_similarities(embeddings: List[torch.Tensor]) -> List[np.ndarray]:
    cossim_matrix_by_layer = []
    for z in tqdm(embeddings):
        z = torch.nn.functional.normalize(z.squeeze(0), dim=1)
        cossim_matrix = torch.matmul(z, z.T).cpu().numpy()
        cossim_matrix_by_layer.append(cossim_matrix)
    return cossim_matrix_by_layer

def plot_similarity_histograms(cossim_matrix_by_layer: List[np.ndarray],
                               save_path: str = None,
                               step: int = 1,
                               bins: int = 64):
    selected = [(i, data) for i, data in enumerate(cossim_matrix_by_layer) if i % step == 0]
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

    for ax, (i, cossim_matrix) in zip(axes, selected):
        cossim_arr = cossim_matrix.flatten()
        IQR = np.percentile(cossim_arr, 75) - np.percentile(cossim_arr, 25)
        bin_width = 2 * IQR / len(cossim_arr) ** (1 / 3)
        optimal_bins = max(10, int((max(cossim_arr) - min(cossim_arr)) / bin_width))

        ax.hist(cossim_arr, bins=optimal_bins, density=True, histtype='step', color='#d62728', linewidth=1.5)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.set_title(f'Layer {i}', fontsize=32)
        ax.set_xlim([-0.4, 1.1])
        ax.set_ylim(0, max_density * 1.1)

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
    return

def plot_probability(cossim_matrix_by_layer: List[np.ndarray], save_path: str = None):
    fig = plt.figure(figsize=(10, 8))
    for subplot_idx, threshold in enumerate([0.9, 0.95, 0.99, 1.0]):
        curr_prob = [(cossim_matrix.flatten() > threshold).sum() / len(cossim_matrix.flatten())
                     for cossim_matrix in cossim_matrix_by_layer]
        ax = fig.add_subplot(2, 2, subplot_idx + 1)
        ax.plot(curr_prob, marker='o', linewidth=2, color='#2ca02c')
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_title(fr'Cosine Similarity $\geq$ {threshold}', fontsize=18)
        ax.set_xlabel('Layer', fontsize=14)
        ax.set_ylabel('Probability', fontsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(r'P(cossim(Embedding)$\approx$1) per Layer', fontsize=24)
    fig.tight_layout(pad=2)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()
    return

def plot_entropy(cossim_matrix_by_layer: List[np.ndarray], save_path: str = None):
    fig = plt.figure(figsize=(12, 6))
    for subplot_idx, entropy_type in enumerate(['Shannon', 'von Neumann']):
        ax = fig.add_subplot(1, 2, subplot_idx + 1)
        if entropy_type == 'Shannon':
            cmap = plt.get_cmap('Greens')
            for num_bins, cmap_idx in zip([64, 256, 1024, 4096], [0.4, 0.6, 0.8, 1.0]):
                entropy_arr = [compute_entropy(cossim_matrix, entropy_type=entropy_type, num_bins=num_bins)
                               for cossim_matrix in cossim_matrix_by_layer]
                ax.plot(entropy_arr, marker='o', linewidth=2, color=cmap(cmap_idx), label=f'{num_bins} bins')
            ax.legend(loc='lower right')
        else:
            entropy_arr = [compute_entropy(cossim_matrix, entropy_type=entropy_type)
                           for cossim_matrix in cossim_matrix_by_layer]
            ax.plot(entropy_arr, marker='o', linewidth=2, color='#2ca02c')

        ax.set_ylim([0, ax.get_ylim()[1]])
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_title(f'{entropy_type} Entropy', fontsize=18)
        ax.set_xlabel('Layer', fontsize=14)
        ax.set_ylabel('Entropy', fontsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('cossim(Embedding) Entropy per Layer', fontsize=24)
    fig.tight_layout(pad=2)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()
    return

def compute_entropy(matrix: np.ndarray, entropy_type: str, num_bins: int = 256):
    assert len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1]
    if entropy_type == 'Shannon':
        vec = matrix.flatten()
        # Min-Max scale.
        vec = (vec - np.min(vec)) / (np.max(vec) - np.min(vec))
        # Binning.
        bins = np.linspace(0, 1, num_bins + 1)[:-1]
        vec_binned = np.digitize(vec, bins=bins)
        # Count probability.
        counts = np.unique(vec_binned, axis=0, return_counts=True)[1]
        prob = counts / np.sum(counts)
        # Compute entropy.
        prob = prob + np.finfo(float).eps
        entropy = -np.sum(prob * np.log2(prob))

    elif entropy_type == 'von Neumann':
        # Ensure Hermitian
        assert np.allclose(matrix, matrix.conj().T)
        # Eigen-decomposition
        eigvals = np.linalg.eigvalsh(matrix)
        # Clip small negative eigenvalues due to numerical errors
        eigvals = np.clip(eigvals, 0, np.inf)
        # Normalize to ensure trace = 1
        eigvals = eigvals / eigvals.sum()
        # Count probability.
        prob = eigvals[eigvals > 0]
        # Compute entropy.
        prob = prob + np.finfo(float).eps
        entropy = -np.sum(prob * np.log2(prob))

    return entropy


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
        cossim_matrix_by_layer = compute_cosine_similarities(output.hidden_states)

    # Plot and save histograms.
    plot_similarity_histograms(
        cossim_matrix_by_layer,
        save_path='../../visualization/embedding_cossim_histogram_albert_xlarge_v2.png')

    # Plot and save metrics (prob density, entropy, etc.).
    plot_probability(
        cossim_matrix_by_layer,
        save_path='../../visualization/embedding_cossim_probability_albert_xlarge_v2.png')
    plot_entropy(
        cossim_matrix_by_layer,
        save_path='../../visualization/embedding_cossim_entropy_albert_xlarge_v2.png')
