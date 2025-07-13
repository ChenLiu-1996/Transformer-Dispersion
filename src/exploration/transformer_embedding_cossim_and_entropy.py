from typing import List
import argparse
import os
import sys
import math
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel
from huggingface_hub import login
import matplotlib.pyplot as plt
import tempfile
from tqdm import tqdm

import_dir = '/'.join(os.path.realpath(__file__).split("/")[:-2])
sys.path.insert(0, import_dir)
from utils.text_data import get_random_long_text
from dse.dse import diffusion_spectral_entropy


def organize_embeddings(embeddings: List[torch.Tensor]) -> List[np.ndarray]:
    embeddings_by_layer = []
    for z in tqdm(embeddings):
        z = z.squeeze(0).cpu().numpy()
        embeddings_by_layer.append(z)
    return embeddings_by_layer

def compute_cosine_similarities(embeddings: List[np.ndarray]) -> List[np.ndarray]:
    cossim_matrix_by_layer = []
    for z in tqdm(embeddings):
        z = normalize(z, axis=1)
        cossim_matrix = np.matmul(z, z.T)
        cossim_matrix_by_layer.append(cossim_matrix)
    return cossim_matrix_by_layer

def plot_similarity_histograms(cossim_matrix_by_layer: List[np.ndarray],
                               save_path: str = None,
                               step: int = 1,
                               bins: int = 128):
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

def plot_similarity_heatmap(cossim_matrix_by_layer: List[np.ndarray],
                            save_path: str = None,
                            step: int = 1,
                            bins: int = 128):
    selected = [(i, data) for i, data in enumerate(cossim_matrix_by_layer) if i % step == 0]

    layer_indices, hist_data = [], []
    for (layer_idx, cossim_matrix) in selected:
        cossim_arr = cossim_matrix.flatten()
        hist, _ = np.histogram(cossim_arr, bins=bins, density=False, range=(-1, 1))
        hist_data.append(hist)
        layer_indices.append(layer_idx)
    hist_matrix = np.array(hist_data)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    im = ax.imshow(hist_matrix, aspect="auto", origin="lower", cmap='Reds', extent=[-1, 1, 0, layer_indices[-1]])

    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_title('Cosine Similarity by Layer', fontsize=28, pad=20)
    ax.set_xlabel('Cosine Similarity', fontsize=24)
    ax.set_ylabel('Layer', fontsize=24)

    fig.colorbar(im, ax=ax)

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

def plot_DSE(embeddings_by_layer: List[np.ndarray], save_path: str = None):
    fig = plt.figure(figsize=(16, 8))
    for subplot_idx, l2_normalize in enumerate([False, True]):
        if l2_normalize:
            normalize_func = normalize
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

def normalize(x, p=2, axis=1, eps=1e-3):
    norm = np.linalg.norm(x, ord=p, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)

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
    parser.add_argument('--huggingface-id', type=str, default='albert-xlarge-v2')
    parser.add_argument('--huggingface-token', type=str, default=None)
    parser.add_argument('--num-attention-heads', type=int, default=None)
    parser.add_argument('--num-hidden-layers', type=int, default=None)
    parser.add_argument('--plot-all', action='store_true')
    parser.add_argument('--random-seed', type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.random_seed)

    if args.huggingface_token is not None:
        login(token=args.huggingface_token)

    with tempfile.TemporaryDirectory() as tmp_cache:
        config_kwargs = {'cache_dir': tmp_cache}
        if args.num_attention_heads is not None:
            # NOTE: We cannot change number of attention heads for most models.
            # ALBERT is an exception.
            config_kwargs['num_attention_heads'] = args.num_attention_heads

        if args.num_hidden_layers is not None:
            # NOTE: We cannot change number of layeres for most models.
            # ALBERT model's attention blocks are identical,
            # so it can be further stacked without a problem.
            config_kwargs['num_hidden_layers'] = args.num_hidden_layers

        try:
            tokenizer = AutoTokenizer.from_pretrained(args.huggingface_id, cache_dir=tmp_cache)
            config = AutoConfig.from_pretrained(args.huggingface_id, **config_kwargs)
            model = AutoModel.from_pretrained(args.huggingface_id, config=config, cache_dir=tmp_cache)
        except Exception as e:
            print(f"Unable to process model: {args.huggingface_id}. Error occurred: {e}.")

    # Run model on a random long input.
    text = get_random_long_text('wikipedia')
    tokens = tokenizer(text, return_tensors='pt', truncation=True)

    # Extract the cosine similarities among token embeddings (hidden states).
    with torch.no_grad():
        output = model(**tokens, output_hidden_states=True)
        embeddings_by_layer = organize_embeddings(output.hidden_states)
        cossim_matrix_by_layer = compute_cosine_similarities(embeddings_by_layer)

    # Plot and save histograms.
    model_name_cleaned = '-'.join(args.huggingface_id.split('/'))
    plot_similarity_heatmap(
        cossim_matrix_by_layer,
        save_path=f'../../visualization/transformer/{model_name_cleaned}/embedding_cossim_heatmap_{model_name_cleaned}_heads_{config.num_attention_heads}.png')

    if args.plot_all:
        plot_similarity_histograms(
            cossim_matrix_by_layer,
            save_path=f'../../visualization/transformer/{model_name_cleaned}/embedding_cossim_histogram_{model_name_cleaned}_heads_{config.num_attention_heads}.png')

        # Plot and save metrics (prob density, entropy, etc.).
        plot_probability(
            cossim_matrix_by_layer,
            save_path=f'../../visualization/transformer/{model_name_cleaned}/embedding_cossim_probability_{model_name_cleaned}.png')
        plot_entropy(
            cossim_matrix_by_layer,
            save_path=f'../../visualization/transformer/{model_name_cleaned}/embedding_cossim_entropy_{model_name_cleaned}.png')
        plot_DSE(
            embeddings_by_layer,
            save_path=f'../../visualization/transformer/{model_name_cleaned}/embedding_DSE_{model_name_cleaned}.png')
